"""Per-program evidence with count-normalized latent aggregation."""
from collections import defaultdict
import numpy as np
from scipy.special import expit
from round103_features import question_tokens,local_tags,matrix
from round101_templates import canonical

def root_features(row,packet):
    words=question_tokens(row['question']);features=[];counts=[]
    for roots in packet:
        programs={}
        for p in roots:
            try:normal=canonical(p);programs[repr(normal)]=normal
            except (ValueError,ZeroDivisionError):continue
        counts.append(len(programs))
        for p in programs.values():
            f=defaultdict(float)
            for tag,v in local_tags(p).items():
                f[tag]+=v
                for word,w in words.items():f[tag+'|'+word]+=v*w*.2
            f['supported']=1.;features.append(dict(f))
    return features,counts

def pool(scores,ptr,mode='latent'):
    count=np.diff(ptr);owner=np.repeat(np.arange(len(count)),count);result=np.zeros(len(count),dtype=np.float64)
    if not len(scores):return result,np.empty(0),owner
    if mode=='mean':
        weight=1/count[owner];result=np.bincount(owner,weights=scores*weight,minlength=len(count));return result,weight,owner
    assert mode=='latent'
    maximum=np.full(len(count),-np.inf);np.maximum.at(maximum,owner,scores);weight=np.exp(scores-maximum[owner]);total=np.bincount(owner,weights=weight,minlength=len(count));keep=count>0
    result[keep]=maximum[keep]+np.log(total[keep]/count[keep]);weight/=total[owner]
    return result,weight,owner

def predict(x,ptr,w,mode='latent'):return pool(np.asarray(x@w).ravel(),ptr,mode)[0].reshape(-1,4)

def pairs(y):
    pos=[];neg=[];weight=[]
    for i,row in enumerate(y):
        pp=np.flatnonzero(row);nn=np.flatnonzero(row==0)
        for a in pp:
            for b in nn:pos.append(4*i+a);neg.append(4*i+b);weight.append(1/(len(pp)*len(nn)))
    return np.array(pos,dtype=int),np.array(neg,dtype=int),np.array(weight)

def loss_gradient(w,x,ptr,pair,mode='latent'):
    scores,root_weight,owner=pool(np.asarray(x@w).ravel(),ptr,mode);pos,neg,pw=pair;delta=scores[pos]-scores[neg]
    loss=2*np.dot(pw,np.logaddexp(0,-delta))+.5*np.dot(w,w);d=-2*pw*expit(-delta)
    cg=np.bincount(pos,weights=d,minlength=len(scores))-np.bincount(neg,weights=d,minlength=len(scores));gradient=np.asarray(x.T@(cg[owner]*root_weight)).ravel()+w
    return float(loss),gradient

def subset(x,ptr,questions):
    candidates=(questions[:,None]*4+np.arange(4)).ravel();count=np.diff(ptr)[candidates];rows=np.concatenate([np.arange(ptr[i],ptr[i+1]) for i in candidates]);return x[rows],np.r_[0,np.cumsum(count)]

def checks():
    from scipy import sparse
    rng=np.random.default_rng(107107);x=sparse.csr_matrix(rng.normal(size=(11,7)));ptr=np.array([0,2,3,3,5,7,8,10,11]);w=rng.normal(size=7)*.2;y=np.array([[1,0,1,0],[0,1,0,1]],float);pair=pairs(y)
    for mode in ('mean','latent'):
        _,g=loss_gradient(w,x,ptr,pair,mode);numeric=[]
        for j in range(len(w)):
            step=np.zeros_like(w);step[j]=1e-5;numeric.append((loss_gradient(w+step,x,ptr,pair,mode)[0]-loss_gradient(w-step,x,ptr,pair,mode)[0])/2e-5)
        assert np.allclose(g,numeric,atol=2e-7,rtol=2e-6)
    s=np.array([1.,3.,2.]);assert np.allclose(pool(s,[0,3])[0],pool(np.tile(s,2),[0,6])[0]);assert pool(np.array([]),[0,0])[0][0]==0
    a=predict(x,ptr,w);xx,pp=subset(x,ptr,np.array([1,0]));assert np.allclose(predict(xx,pp,w),a[[1,0]])
    return dict(analytic_gradient_mean=True,analytic_gradient_latent=True,uniform_duplication_invariant=True,empty_evidence_zero=True,question_permutation=True)
