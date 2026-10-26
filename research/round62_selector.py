"""Compact kernel ranker with training-only Nyström normalization folded into weights."""
from pathlib import Path
import json,numpy as np
from round59_selector import features
from round25_selector import scores as baseline
from round22_selector import normalized
from round4_gates import apply_gate

def compact(x,w):
    e=(x[:,:,:1024]-w['pmean'])@w['pc'].T/w['pscale'];av=e.mean(1,keepdims=True)
    return np.concatenate([e,e-av,e*av,x[:,:,1024:]],2).reshape(len(x)*4,-1)

def transform(x,w):return np.clip((compact(x,w)-w['mean'])/w['scale'],-8,8)*w['blockscale']

def kernel(x,w):
    b=w['anchors'];d=np.maximum((x*x).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*x@b.T,0)
    return np.exp(-3*d/w['median'])

def predict(bank,arrays,checkpoint):
    if not bank:return []
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());x,_=features(bank,arrays)
    with np.load(p/'model.npz') as w:s=(kernel(transform(x,w),w)@w['weight']).reshape(-1,4)
    if cfg['alpha']<1:s=(1-cfg['alpha'])*normalized(baseline(bank,arrays,p/'baseline'))+cfg['alpha']*normalized(s)
    return apply_gate(bank,s,'parse_and_repair',4).astype(int).tolist()
