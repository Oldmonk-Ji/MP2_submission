"""Compact sparse question-token x numeric-evidence interactions."""
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

DIM=32768

def questions(bank):
    docs=[re.sub(r'\d+(?:[.,]\d+)*',' number ',r['question'].lower()) for r in bank]
    x=HashingVectorizer(n_features=4096,ngram_range=(1,2),alternate_sign=False,norm=None,dtype=np.float32).transform(docs).tocsr()
    x.data=1+np.log(x.data)
    return normalize(x,norm='l2')

def fit_scale(g,rows):
    mean=g[rows].mean((0,1));scale=g[rows].std((0,1)).clip(.05)
    z=np.clip((g[rows]-mean)/scale,-6,6)
    variance=np.square(z-z.mean(1,keepdims=True)).mean((0,1))
    columns=np.argsort(-variance,kind='stable')[:32].astype(np.int32)
    return dict(mean=mean,scale=scale,columns=columns)

def build(text,g,w):
    g=np.clip((g-w['mean'])/w['scale'],-6,6)[:,:,w['columns']]/np.sqrt(32)
    indices=[];values=[];ptr=[0]
    for i in range(len(g)):
        lo,hi=text.indptr[i:i+2];cols=text.indices[lo:hi];v=text.data[lo:hi]
        hashed=(cols[:,None]*82+w['columns'][None,:])%DIM
        indices.append(np.tile(hashed.ravel(),4).astype(np.int32))
        values.append((g[i,:,None,:]*v[None,:,None]).ravel().astype(np.float32))
        count=len(cols)*32
        ptr.extend(ptr[-1]+count*np.arange(1,5))
    x=sparse.csr_matrix((np.concatenate(values),np.concatenate(indices),np.asarray(ptr,np.int64)),shape=(len(g)*4,DIM))
    x.sum_duplicates();x.eliminate_zeros()
    return x
