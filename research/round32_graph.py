"""Contextual quantity roles and pair operations, without generation at inference."""
import json
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from round29_model import ProgramModel
from round29_programs import encode_questions,parse

OPERATORS=['ADD','SUB','RSUB','MUL','DIV','RDIV']


def labels(item,question,strict=False):
    roles=np.full(24,-100,np.int64);pairs=np.full((24,24),-100,np.int64)
    if not item.get('informative',False) or (strict and not item['clean']):return roles,pairs
    qs=item['quantities'];relations=defaultdict(set)
    def walk(p):
        if p[0]=='N':
            grad=np.zeros(24);grad[p[1]]=1
            return float(Fraction(qs[p[1]]['value'])),grad,{p[1]}
        if p[0]=='L':return float(Fraction(p[1])),np.zeros(24),set()
        a,ga,sa=walk(p[1]);b,gb,sb=walk(p[2])
        for i in sa:
            for j in sb:
                if i==j:continue
                lo,hi=sorted((i,j));op=p[0]
                if i>j and op in ('SUB','DIV'):op='R'+op
                relations[(lo,hi)].add(OPERATORS.index(op))
        if p[0]=='ADD':v,g=a+b,ga+gb
        elif p[0]=='SUB':v,g=a-b,ga-gb
        elif p[0]=='MUL':v,g=a*b,ga*b+gb*a
        else:
            if abs(b)<1e-10:raise ValueError('zero')
            v,g=a/b,(ga*b-gb*a)/(b*b)
        return v,g,sa|sb
    try:_,grad,used=walk(item['program'])
    except (ValueError,ZeroDivisionError,OverflowError):return roles,pairs
    for i,q in enumerate(qs):
        if i in used:roles[i]=1 if grad[i]>1e-8 else (2 if grad[i]<-1e-8 else 0)
        elif item['clean'] and Fraction(q['value']) not in (0,1,Fraction(1,2),Fraction(1,4),100):
            # Half/quarter may be represented as literal denominators. Do not
            # call those mentions irrelevant merely because the leaf is literal.
            roles[i]=0
    for (i,j),ops in relations.items():
        if len(ops)==1:pairs[i,j]=next(iter(ops))
    return roles,pairs


def encode_graphs(items,questions,strict=False):
    rr=[];pp=[]
    for r,q in zip(items,questions):a,b=labels(r,q,strict);rr.append(a);pp.append(b)
    return np.asarray(rr),np.asarray(pp)


class GraphModel(nn.Module):
    def __init__(self,vocabulary_size):
        super().__init__()
        self.words=nn.Embedding(vocabulary_size,64,padding_idx=0)
        self.encoder=nn.GRU(64,64,batch_first=True,bidirectional=True)
        self.role=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,3))
        self.pair=nn.Sequential(nn.Linear(640,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,6))
        self.bias=nn.Parameter(torch.zeros(()))
        self.register_buffer('role_prior',torch.full((3,),1/3))
        self.register_buffer('pair_prior',torch.full((6,),1/6))

    def forward(self,q):
        mask=q.ne(0);length=mask.sum(1).clamp_min(1)
        emb=nn.functional.dropout(self.words(q),p=.2,training=self.training)
        packed=nn.utils.rnn.pack_padded_sequence(emb,length.cpu(),batch_first=True,enforce_sorted=False)
        h,_=self.encoder(packed);h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=q.shape[1])
        slots=q[:,:,None].eq(torch.arange(2,26,device=q.device)[None,None,:])
        numbers=torch.einsum('bln,bld->bnd',slots.float(),h)
        positions=torch.arange(q.shape[1],device=q.device)[None,:]
        tail=mask & positions.ge((length-20).clamp_min(0)[:,None])
        target=(h*tail[:,:,None]).sum(1)/tail.sum(1,keepdim=True).clamp_min(1)
        t=target[:,None,:].expand(-1,24,-1)
        role=self.role(torch.cat([numbers,t],2)).log_softmax(2)
        a=numbers[:,:,None,:].expand(-1,-1,24,-1);b=numbers[:,None,:,:].expand(-1,24,-1,-1)
        pair=self.pair(torch.cat([a,b,a-b,a*b,target[:,None,None,:].expand(-1,24,24,-1)],3)).log_softmax(3)
        return role,pair

    def score(self,logp,role_targets,pair_targets):
        role,pair=logp
        if role_targets.ndim==3:
            k=role_targets.shape[1];role=role[:,None,:,:].expand(-1,k,-1,-1);pair=pair[:,None,:,:,:].expand(-1,k,-1,-1,-1)
        rp=role-self.role_prior.log();pp=pair-self.pair_prior.log()
        rm=role_targets.ge(0);pm=pair_targets.ge(0)
        rs=rp.gather(-1,role_targets.clamp_min(0)[...,None]).squeeze(-1)
        ps=pp.gather(-1,pair_targets.clamp_min(0)[...,None]).squeeze(-1)
        return (rs*rm).sum(-1)/rm.sum(-1).clamp_min(1)+(ps*pm).sum((-1,-2))/pm.sum((-1,-2)).clamp_min(1)+self.bias


def load(path):
    path=Path(path);vocab=json.loads((path/'vocabulary.json').read_text(encoding='utf-8'));model=GraphModel(len(vocab))
    with np.load(path/'model.npz',allow_pickle=False) as z:model.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    return model.eval(),vocab


@torch.no_grad()
def predict_encoded(model,q,roles,pairs,device='cpu',batch=64):
    from round29_model import trim
    out=[]
    model.eval()
    for start in range(0,len(q),batch):
        qq=torch.as_tensor(trim(q[start:start+batch]),device=device)
        rr=torch.as_tensor(roles[start:start+batch],device=device);pp=torch.as_tensor(pairs[start:start+batch],device=device)
        out.append(model.score(model(qq),rr,pp).cpu().numpy())
    return np.concatenate(out)
