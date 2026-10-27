"""Bounded raw-text inputs and fixed positions for a scratch Transformer."""
import numpy as np
from round241_text import fit_vocab as original_vocab,encode as original_encode,save_vocab

QLEN=64
CLEN=160
VOCAB=1728

def fit_vocab(bank,rows):
    return {k:v for k,v in original_vocab(bank,rows).items() if v<VOCAB}

def encode(bank,vocab):
    old=original_encode(bank,vocab)
    result={}
    for key,nkey,length in [('q','qn',QLEN),('c','cn',CLEN)]:
        original=old[key];numbers=old[nkey]
        shape=original.shape[:-1]
        a=np.zeros((*shape,length),np.int64);b=np.zeros((*shape,length,4),np.float32)
        flat=original.reshape(-1,original.shape[-1]);fn=numbers.reshape(-1,numbers.shape[-2],4)
        fa=a.reshape(-1,length);fb=b.reshape(-1,length,4)
        for i,row in enumerate(flat):
            count=int((row!=0).sum())
            ix=np.arange(length) if count<=length else np.r_[np.arange(length*2//3),np.arange(count-(length-length*2//3),count)]
            fa[i]=row[ix];fb[i]=fn[i,ix]
        result[key]=a;result[nkey]=b
    return result

def positions():
    length=(QLEN+CLEN)//4
    p=np.arange(length,dtype=np.float32)[:,None]
    scale=np.exp(np.arange(0,48,2,dtype=np.float32)*np.float32(-np.log(10000.)/48))
    result=np.empty((length,48),np.float32)
    result[:,0::2]=np.sin(p*scale);result[:,1::2]=np.cos(p*scale)
    return result
