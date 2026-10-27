"""Compact question-only semantic context learned from supplied TRAIN."""
import re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def hashed(bank):
    docs=[re.sub(r'\d+(?:[.,]\d+)*',' number ',b['question'].lower()) for b in bank]
    vectorizer=HashingVectorizer(n_features=4096,ngram_range=(1,2),alternate_sign=False,norm=None,dtype=np.float32)
    x=vectorizer.transform(docs).tocsr()
    x.data=1+np.log(x.data)
    return x

def raw_context(x,w):
    a=normalize(x.multiply(w['lex_idf']).tocsr(),norm='l2')
    return np.asarray(a@w['lex_pc'].astype(np.float32).T,np.float32)

def fit_context(x,rows,w):
    w=dict(w)
    count=np.asarray((x[rows]>0).sum(0)).ravel()
    w['lex_idf']=(np.log((1+len(rows))/(1+count))+1).astype(np.float32)
    a=normalize(x[rows].multiply(w['lex_idf']).tocsr(),norm='l2')
    model=TruncatedSVD(n_components=16,n_iter=7,random_state=238238)
    model.fit(a)
    w['lex_pc']=model.components_.astype(np.float16)
    h=raw_context(x,w)
    w['lex_mean']=h[rows].mean(0)
    w['lex_scale']=h[rows].std(0).clip(.001)
    return w

def context(x,w):
    return np.clip((raw_context(x,w)-w['lex_mean'])/w['lex_scale'],-6,6)
