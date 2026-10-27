"""Input-only utility features for a text-supported correction of round235."""
import numpy as np
from common import parse_answer

def normalized(s):
    return (s-s.mean(1,keepdims=True))/np.maximum(s.std(1,keepdims=True),1e-6)

def entropy(s):
    p=np.exp(s-s.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    return -(p*np.log(np.maximum(p,1e-9))).sum(1)/np.log(4.)

def build(bank,s,strict,text,old,mode):
    n=len(bank);row=np.arange(n)
    parsed=[[parse_answer(c['text']) for c in r['candidates']] for r in bank]
    same=np.asarray([[[a is not None and a==b for b in ps] for a in ps] for ps in parsed],bool)
    same|=np.eye(4,dtype=bool)[None]
    counts=same.sum(2).astype(np.float32)
    valid=np.asarray([[a is not None for a in ps] for ps in parsed],bool)
    different=~same[row,old]
    def diff(a):return a-a[row,old,None]
    def repeat(a):return np.broadcast_to(a[:,None],(n,4))
    sn,nn,tn=normalized(s),normalized(text),normalized(strict)
    ds,dn,dt=diff(s),diff(text),diff(strict)
    dsn,dnn,dtn=diff(sn),diff(nn),diff(tn)
    rank=(s[:,:,None]<s[:,None,:]).sum(2).astype(np.float32)
    oldtop=(s==s.max(1,keepdims=True));ntop=(text==text.max(1,keepdims=True));ttop=(strict==strict.max(1,keepdims=True))
    gate=(~oldtop[row,old]).astype(np.float32)
    topgap=np.sort(s,axis=1)[:,-1]-np.sort(s,axis=1)[:,-2]
    ngap=np.sort(text,axis=1)[:,-1]-np.sort(text,axis=1)[:,-2]
    mean=np.einsum('nij,nj->ni',same.astype(np.float32),text)/counts
    maximum=np.where(same,text[:,None,:],-np.inf).max(2)
    bmean=np.einsum('nij,nj->ni',same.astype(np.float32),s)/counts
    nstd=np.maximum(text.std(1,keepdims=True),1e-6);bstd=np.maximum(s.std(1,keepdims=True),1e-6)
    named=[('base_delta',ds),('base_normalized_delta',dsn),('text_delta',dn),('text_normalized_delta',dnn),
        ('strict_delta',dt),('strict_normalized_delta',dtn),('text_group_mean_delta',diff(mean)/nstd),
        ('text_group_max_delta',diff(maximum)/nstd),('base_group_mean_delta',diff(bmean)/bstd),
        ('answer_group_size_delta',diff(counts)/4),('base_rank',rank/3),('old_base_rank',repeat(rank[row,old]/3)),
        ('base_top',oldtop),('text_top',ntop),('strict_top',ttop),('old_not_raw_top',repeat(gate)),
        ('base_top_gap',repeat(topgap)/bstd),('text_top_gap',repeat(ngap)/nstd),
        ('text_spread',repeat(np.log1p(text.std(1)))),('base_spread',repeat(np.log1p(s.std(1)))),
        ('base_entropy',repeat(entropy(s))),('text_entropy',repeat(entropy(text))),
        ('text_strict_agree',ntop&ttop),('text_strict_product',np.tanh(dnn)*np.tanh(dtn)),
        ('text_base_product',np.tanh(dnn)*np.tanh(dsn)),('text_under_base_uncertainty',dnn/(1+repeat(topgap))),
        ('gate_raw_alternative',repeat(gate)*oldtop),('text_group_strict_product',np.tanh(diff(mean)/nstd)*np.tanh(dtn)),
        ('old_strict_top',repeat(ttop[row,old].astype(np.float32))),('old_parse_valid',repeat(valid[row,old].astype(np.float32)))]
    features=np.stack([a.astype(np.float32) for _,a in named],2)
    eligible=valid&different&(dn>0)
    if mode=='joint':eligible&=dt>0
    elif mode=='top2':eligible&=rank<2
    elif mode!='supported':raise ValueError(mode)
    return features,eligible,[name for name,_ in named]

def choose(bank,s,strict,text,old,w,mode):
    if not bank:return np.empty(0,np.int64)
    x,eligible,_=build(bank,s,strict,text,old,mode)
    z=np.clip((x-w['mean'])/w['scale'],-6,6)
    if w['coef'].ndim==2:
        logits=z@w['coef'].T+w['intercept']
        p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        logit=p@w['classes'] # Expected accuracy gain, including neutral changes.
    else:
        logit=z@w['coef']+w['intercept']
    logit=np.where(eligible,logit,-np.inf)
    best=logit.argmax(1);row=np.arange(len(bank));result=np.asarray(old).copy()
    use=logit[row,best]>0 # Positive expected gain, or binary P(benefit)>.5; no threshold search.
    result[use]=best[use]
    return result
