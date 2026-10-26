"""Scratch recurrent text alignment, original official embeddings only."""
import re,json,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from round33_text import names
from round29_programs import quantities

TOKEN=re.compile(r"\d[\d,]*(?:\.\d+)?|[a-z]+(?:['’][a-z]+)?|[^\w\s]",re.I)
NUMBER=re.compile(r'^\d[\d,]*(?:\.\d+)?$')


def tokens(text,question):
    entities=names(question);source=Counter(q['value'] for q in quantities(question));out=[]
    from fractions import Fraction
    for raw in TOKEN.findall(text):
        t=raw.lower().replace('’',"'");numeric=[0.]*6
        if NUMBER.fullmatch(t):
            try:
                f=Fraction(t.replace(',',''));v=min(float(f),1e12);key=str(f)
                numeric=[1.,math.log1p(abs(v))/10,v/(1+abs(v)),float(v.is_integer()),float(key in source),float(source[key]>1)]
            except (ValueError,OverflowError):pass
            t='<number>'
        else:
            stem=t.removesuffix("'s")
            if stem in entities:t=entities[stem]+('_possessive' if stem!=t else '')
        out.append((t,numeric))
    return out or [('<unk>',[0.]*6)]


def vocabulary(bank,limit=6000):
    count=Counter()
    for r in bank:
        for s in [r['question']]+[c['text'] for c in r['candidates']]:count.update(t for t,n in tokens(s,r['question']))
    fixed=['<pad>','<unk>','<number>']+[f'entity{i}{s}' for i in range(16) for s in ['', '_possessive']]
    words=sorted((t for t in count if t not in fixed),key=lambda t:(-count[t],t))[:limit-len(fixed)]
    return {w:i for i,w in enumerate(fixed+words)}


def encode(bank,arrays,vocab):
    data=dict(q=np.zeros((len(bank),192),np.int64),qn=np.zeros((len(bank),192,6),np.float32),
        c=np.zeros((len(bank),4,320),np.int64),cn=np.zeros((len(bank),4,320,6),np.float32),e=np.asarray(arrays['embeddings'],np.float32))
    stats=dict(question_truncated=0,candidate_truncated=0,question_tokens=0,candidate_tokens=0,question_retained=0,candidate_retained=0)
    for i,r in enumerate(bank):
        for j,s in enumerate([r['question']]+[c['text'] for c in r['candidates']]):
            k='q' if j==0 else 'c';kind='question' if j==0 else 'candidate';limit=192 if j==0 else 320
            ts=tokens(s,r['question']);stats[kind+'_tokens']+=len(ts)
            if len(ts)>limit:ts=ts[:limit//2]+ts[-limit//2:];stats[kind+'_truncated']+=1
            stats[kind+'_retained']+=len(ts)
            index=i if j==0 else (i,j-1);data[k][index][:len(ts)]=[vocab.get(t,1) for t,n in ts];data[k+'n'][index][:len(ts)]=[n for t,n in ts]
    return data,stats


def batch(data,ix,device):
    out={k:torch.as_tensor(v[ix],device=device) for k,v in data.items()}
    for key in ['q','c']:
        length=max(1,int(out[key].ne(0).sum(-1).max()));out[key]=out[key][...,:length];out[key+'n']=out[key+'n'][...,:length,:]
    return out


class TextModel(nn.Module):
    def __init__(self,vocab_size,text_enabled=True):
        super().__init__();self.text_enabled=text_enabled
        self.words=nn.Embedding(vocab_size,48,padding_idx=0);self.numeric=nn.Linear(6,48,bias=False)
        self.encoder=nn.GRU(48,48,batch_first=True,bidirectional=True)
        self.compare=nn.Sequential(nn.Linear(384,48),nn.ReLU(),nn.Dropout(.2))
        self.utility=nn.Sequential(nn.Linear(2144,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        self.register_buffer('mean',torch.zeros(1024));self.register_buffer('scale',torch.ones(1024))

    def encode_text(self,ids,numeric):
        mask=ids.ne(0);length=mask.sum(1).clamp_min(1)
        x=nn.functional.dropout(self.words(ids)+self.numeric(numeric),.2,self.training)
        p=nn.utils.rnn.pack_padded_sequence(x,length.cpu(),batch_first=True,enforce_sorted=False)
        h,_=self.encoder(p);h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=ids.shape[1])
        return h,mask

    def forward(self,q,qn,c,cn,e):
        n=e.shape[0]
        if self.text_enabled:
            qh,qm=self.encode_text(q,qn);ch,cm=self.encode_text(c.reshape(n*4,-1),cn.reshape(n*4,-1,6))
            qh=qh.repeat_interleave(4,0);qm=qm.repeat_interleave(4,0)
            attention=(ch@qh.transpose(1,2))/(96**.5);aligned=attention.masked_fill(~qm[:,None,:],-1e4).softmax(2)@qh
            h=self.compare(torch.cat([ch,aligned,ch-aligned,ch*aligned],2))
            average=(h*cm[:,:,None]).sum(1)/cm.sum(1,keepdim=True).clamp_min(1)
            maximum=h.masked_fill(~cm[:,:,None],-1e4).amax(1);text=torch.cat([average,maximum],1).reshape(n,4,96)
        else:text=e.new_zeros(n,4,96)
        normalized=((e-self.mean)/self.scale).clamp(-10,10)
        return self.utility(torch.cat([normalized,normalized-normalized.mean(1,keepdim=True),text],2)).squeeze(-1)


@torch.no_grad()
def predict_data(model,data,device='cpu',batch_size=24):
    model.eval();out=[]
    for start in range(0,len(data['q']),batch_size):out.append(model(**batch(data,slice(start,start+batch_size),device)).cpu().numpy())
    return np.concatenate(out)


def load(path):
    path=Path(path);cfg=json.loads((path/'config.json').read_text());vocab=json.loads((path/'vocabulary.json').read_text(encoding='utf-8'))
    model=TextModel(len(vocab),cfg['text_enabled'])
    with np.load(path/'model.npz',allow_pickle=False) as z:model.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    return model.eval(),vocab
