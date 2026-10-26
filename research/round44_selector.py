"""Offline relation-policy inference. No labels, reference files or training IDs."""
import json
from pathlib import Path
import numpy as np
from round44_programs import proposals,matrix,LIMIT
from round25_selector import scores
from round22_selector import normalized
from round4_gates import apply_gate

def infer_features(bank):
    packets=[proposals(r) for r in bank];x=matrix(bank,packets);mask=np.zeros((len(bank),LIMIT),bool);matches=np.zeros((len(bank),LIMIT,4),bool)
    for i,p in enumerate(packets):
        for j,h in enumerate(p['hypotheses']):mask[i,j]=True;matches[i,j,h['matches']]=True
    return x,mask,matches,packets

def program_scores(x,paths):
    out=np.zeros((x.shape[0]//LIMIT,LIMIT),np.float32)
    for path in paths:
        with np.load(Path(path)/'model.npz',allow_pickle=False) as w:
            h=np.maximum(x@w['w0']+w['b0'],0);h=np.maximum(h@w['w1']+w['b1'],0);out+=(h@w['w2']+w['b2']).reshape(-1,LIMIT)/len(paths)
    return out

def select(bank,base_scores,policy_scores,mask,matches,mode):
    base=apply_gate(bank,base_scores,'parse_and_repair',4);valid=mask.copy();valid[~valid.any(1),0]=True
    s=np.where(valid,policy_scores,-1e9);s=s-s.max(1,keepdims=True);p=np.exp(s);p/=p.sum(1,keepdims=True)
    if mode=='execute_top':
        choices=base.copy()
        for i,j in enumerate(s.argmax(1)):
            eligible=np.flatnonzero(matches[i,j]) if mask[i].any() else []
            if len(eligible):choices[i]=max(eligible,key=lambda k:base_scores[i,k])
        return choices
    mass=np.einsum('nh,nhc->nc',p,matches)
    return apply_gate(bank,base_scores+.25*normalized(np.log(mass.clip(1e-6,1))),'parse_and_repair',4)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
    bs=scores(bank,arrays,path/'baseline');x,mask,matches,_=infer_features(bank)
    s=program_scores(x,[path/name for name in cfg['policies']]);return select(bank,bs,s,mask,matches,cfg['mode']).astype(int).tolist()
