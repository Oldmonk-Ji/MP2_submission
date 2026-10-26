"""Program decoder with contextual quantity pointers, initialized from scratch."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from round29_model import ProgramModel as BaseModel
from round29_programs import SYMBOLS,SYMBOL_INDEX,encode_questions,encode_programs,parse
from round29_model import predict_encoded,centered,save


class ProgramModel(BaseModel):
    def __init__(self,vocabulary_size):
        super().__init__(vocabulary_size)
        self.source_projection=nn.Linear(128,64)
        self.pointer_query=nn.Linear(224,128,bias=False)

    def encode(self,questions):
        enc,mask,pooled=super().encode(questions)
        slot_ids=torch.arange(2,26,device=questions.device)
        positions=questions[:,:,None].eq(slot_ids[None,None,:])
        numbers=torch.einsum('bln,bld->bnd',positions.float(),enc)
        exists=positions.any(1)
        return enc,mask,pooled,numbers,exists

    def forward(self,questions,programs,memory=None):
        enc,mask,pooled,numbers,exists=self.encode(questions) if memory is None else memory
        if len(programs)!=len(enc):
            repeat=len(programs)//len(enc)
            enc=enc.repeat_interleave(repeat,0);mask=mask.repeat_interleave(repeat,0);pooled=pooled.repeat_interleave(repeat,0)
            numbers=numbers.repeat_interleave(repeat,0);exists=exists.repeat_interleave(repeat,0)
        previous=torch.cat([torch.full_like(programs[:,:1],SYMBOL_INDEX['BOS']),programs[:,:-1]],1)
        embedded=self.symbols(previous)
        start=SYMBOL_INDEX['N0'];is_pointer=previous.ge(start)
        number_embedding=self.source_projection(numbers)
        selected=number_embedding.gather(1,(previous-start).clamp(0,23)[:,:,None].expand(-1,-1,64))
        embedded=torch.where(is_pointer[:,:,None],selected,embedded)
        h,_=self.decoder(torch.cat([self.dropout(embedded),pooled[:,None,:].expand(-1,programs.shape[1],-1)],2),
            torch.tanh(self.initial(pooled))[None,:,:])
        attention=(self.query(h)@enc.transpose(1,2))/(128**.5)
        context=attention.masked_fill(~mask[:,None,:],-1e4).softmax(2)@enc
        combined=self.dropout(torch.cat([h,context],2))
        fixed=self.output(combined)[:,:,:start]
        pointers=(self.pointer_query(combined)@numbers.transpose(1,2))/(128**.5)
        pointers=pointers.masked_fill(~exists[:,None,:],-1e4)
        logits=torch.cat([fixed,pointers],2)
        logp=logits.log_softmax(2).gather(2,programs[:,:,None]).squeeze(-1)
        keep=programs.ne(0);nll=-(logp*keep).sum(1)/keep.sum(1).clamp_min(1)
        content=keep.clone()
        for token in ['EOS','LIT','END']:content &= programs.ne(SYMBOL_INDEX[token])
        semantic=(logp*content).sum(1)/content.sum(1).clamp_min(1)
        return nll,semantic,logits.argmax(2)


def load(path):
    path=Path(path);vocab=json.loads((path/'vocabulary.json').read_text(encoding='utf-8'))
    model=ProgramModel(len(vocab))
    with np.load(path/'model.npz',allow_pickle=False) as z:model.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    return model.eval(),vocab


def semantic_scores(bank,path):
    torch.set_num_threads(4);model,vocab=load(path)
    programs=[[parse(r['question'],c['text']) for c in r['candidates']] for r in bank]
    pp=encode_programs([p for row in programs for p in row]).reshape(len(bank),4,-1)
    qq=encode_questions(bank,vocab,max_length=256);score,_=predict_encoded(model,qq,pp)
    usable=np.asarray([[p['clean'] for p in row] for row in programs])
    return centered(score,usable)
