"""Frozen CPU selector: learned source graph decisions -> Fraction execution -> original candidates."""
from pathlib import Path
import json
import numpy as np
from round46_graph import proposals,matrix,LIMIT
from round25_selector import scores as baseline_scores
from round22_selector import normalized
from round4_gates import apply_gate

def forward(x,w):
    h=np.maximum(x@w['w0']+w['b0'],0)
    h=np.maximum(h@w['w1']+w['b1'],0)
    return (h@w['w2']+w['b2']).reshape(-1,LIMIT)

def probability(s,mask):
    m=mask.copy();m[~m.any(1),0]=True
    v=np.where(m,s,-1e9);v-=v.max(1,keepdims=True);p=np.exp(v);return p/p.sum(1,keepdims=True)

def select(bank,bs,s,mask,matches,mode='fixed_residual'):
    base=apply_gate(bank,bs,'parse_and_repair',4)
    p=probability(s,mask)
    if mode=='execute_top':
        out=base.copy()
        for i,j in enumerate(p.argmax(1)):
            eligible=np.flatnonzero(matches[i,j]) if mask[i].any() else []
            if len(eligible):out[i]=max(eligible,key=lambda k:bs[i,k])
        return out
    mass=np.einsum('nh,nhc->nc',p,matches)
    return apply_gate(bank,bs+.25*normalized(np.log(mass.clip(1e-6,1))),'parse_and_repair',4)

def infer(bank,paths):
    scores=[];masks=[];matched=[]
    weights=[]
    for path in paths:
        with np.load(Path(path)/'model.npz',allow_pickle=False) as z:weights.append({k:z[k].copy() for k in z.files})
    for start in range(0,len(bank),32):
        rows=bank[start:start+32];packets=[proposals(r) for r in rows];x=matrix(rows,packets)
        mask=np.zeros((len(rows),LIMIT),bool);matches=np.zeros((len(rows),LIMIT,4),bool)
        for i,p in enumerate(packets):
            for j,h in enumerate(p['hypotheses']):mask[i,j]=True;matches[i,j,h['matches']]=True
        scores.append(sum(forward(x,w) for w in weights)/len(weights));masks.append(mask);matched.append(matches)
    return np.concatenate(scores),np.concatenate(masks),np.concatenate(matched)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
    bs=baseline_scores(bank,arrays,path/'baseline');s,mask,matches=infer(bank,[path/p for p in cfg['policies']])
    return select(bank,bs,s,mask,matches,cfg['mode']).astype(int).tolist()
