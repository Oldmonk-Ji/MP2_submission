"""Scratch question encoder and conditional executable-program language model."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from round29_programs import SYMBOLS,SYMBOL_INDEX,encode_questions,encode_programs,parse


class ProgramModel(nn.Module):
    def __init__(self,vocabulary_size):
        super().__init__()
        self.words=nn.Embedding(vocabulary_size,64,padding_idx=0)
        self.encoder=nn.GRU(64,64,batch_first=True,bidirectional=True)
        self.symbols=nn.Embedding(len(SYMBOLS),64,padding_idx=0)
        self.initial=nn.Linear(128,96)
        self.decoder=nn.GRU(64+128,96,batch_first=True)
        self.query=nn.Linear(96,128,bias=False)
        self.output=nn.Linear(96+128,len(SYMBOLS))
        self.dropout=nn.Dropout(.15)

    def encode(self,questions):
        mask=questions.ne(0);length=mask.sum(1).clamp_min(1).cpu()
        emb=self.dropout(self.words(questions))
        packed=nn.utils.rnn.pack_padded_sequence(emb,length,batch_first=True,enforce_sorted=False)
        encoded,_=self.encoder(packed)
        memory,_=nn.utils.rnn.pad_packed_sequence(encoded,batch_first=True,total_length=questions.shape[1])
        pooled=(memory*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return memory,mask,pooled

    def forward(self,questions,programs,memory=None):
        memory=self.encode(questions) if memory is None else memory
        enc,mask,pooled=memory
        if len(programs)!=len(enc):
            repeat=len(programs)//len(enc)
            enc=enc.repeat_interleave(repeat,0);mask=mask.repeat_interleave(repeat,0);pooled=pooled.repeat_interleave(repeat,0)
        previous=torch.cat([torch.full_like(programs[:,:1],SYMBOL_INDEX['BOS']),programs[:,:-1]],1)
        h,_=self.decoder(torch.cat([self.dropout(self.symbols(previous)),pooled[:,None,:].expand(-1,programs.shape[1],-1)],2),
            torch.tanh(self.initial(pooled))[None,:,:])
        attention=(self.query(h)@enc.transpose(1,2))/(128**.5)
        context=attention.masked_fill(~mask[:,None,:],-1e4).softmax(2)@enc
        logits=self.output(self.dropout(torch.cat([h,context],2)))
        logp=logits.log_softmax(2).gather(2,programs[:,:,None]).squeeze(-1)
        keep=programs.ne(0)
        nll=-(logp*keep).sum(1)/keep.sum(1).clamp_min(1)
        # Semantic content excludes grammar punctuation, but retains operators,
        # source pointers and literal digits. This rule is fixed before VAL.
        content=keep.clone()
        for token in ['EOS','LIT','END']:content &= programs.ne(SYMBOL_INDEX[token])
        semantic=(logp*content).sum(1)/content.sum(1).clamp_min(1)
        return nll,semantic,logits.argmax(2)


def trim(x):
    return x[:,:max(1,int((x!=0).sum(1).max()))]


def save(model,path):
    np.savez(Path(path),**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})


def load(path):
    path=Path(path);vocab=json.loads((path/'vocabulary.json').read_text(encoding='utf-8'))
    model=ProgramModel(len(vocab))
    with np.load(path/'model.npz',allow_pickle=False) as z:model.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    return model.eval(),vocab


def centered(scores,usable):
    denom=usable.sum(1,keepdims=True).clip(1)
    mean=(scores*usable).sum(1,keepdims=True)/denom
    return np.where(usable,scores-mean,0)


@torch.no_grad()
def predict_encoded(model,questions,programs,device='cpu',batch=64):
    model.eval();scores=[];nlls=[]
    for start in range(0,len(questions),batch):
        q=torch.as_tensor(trim(questions[start:start+batch]),device=device)
        p=programs[start:start+batch];p=torch.as_tensor(trim(p.reshape(-1,p.shape[-1])),device=device)
        loss,s,_=model(q,p);scores.append(s.cpu().numpy().reshape(-1,4));nlls.append(loss.cpu().numpy().reshape(-1,4))
    return np.concatenate(scores),np.concatenate(nlls)


def semantic_scores(bank,path):
    torch.set_num_threads(4)
    model,vocab=load(path)
    programs=[[parse(r['question'],c['text']) for c in r['candidates']] for r in bank]
    pp=encode_programs([p for row in programs for p in row]).reshape(len(bank),4,-1)
    qq=encode_questions(bank,vocab,max_length=256)
    score,_=predict_encoded(model,qq,pp)
    usable=np.asarray([[p['clean'] for p in row] for row in programs])
    return centered(score,usable)
