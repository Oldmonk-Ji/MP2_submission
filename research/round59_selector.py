"""Permutation-equivariant learned pairwise comparison of supplied candidates."""
import json
from pathlib import Path
import numpy as np
from features import prepare,matrix,metadata
from round3_features import grounding
from round25_selector import scores as base_scores
from round22_selector import normalized
from round4_gates import apply_gate

def features(bank,arrays):
    p=prepare(bank,arrays)
    x=np.concatenate([p['embedding'],matrix(p,'features_checked'),grounding(bank)],2)
    answers=metadata(bank)['answers']
    other=np.asarray([[[a!=b and a is not None and b is not None for b in row] for a in row] for row in answers],np.float32)
    return x.astype(np.float32),other

def forward(x,other,w,pair=True):
    x=np.clip((x-w['mean'])/w['scale'],-8,8)
    h=np.maximum(x@w['encw']+w['encb'],0)
    solo=(np.maximum(h@w['solow']+w['solob'],0)@w['outw']+w['outb']).squeeze(-1)
    if not pair:return solo
    left=np.broadcast_to(h[:,:,None,:],(len(h),4,4,64));right=np.broadcast_to(h[:,None,:,:],left.shape)
    z=np.concatenate([left,right,left-right,left*right],-1)
    r=(np.maximum(z@w['pairw']+w['pairb'],0)@w['poutw']+w['poutb']).squeeze(-1)
    return solo+((r-r.transpose(0,2,1))*other).sum(2)/np.maximum(other.sum(2),1)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text())
    x,o=features(bank,arrays);scores=[]
    for name in cfg['models']:
        with np.load(path/name) as w:scores.append(forward(x,o,w,cfg['pair']))
    s=normalized(np.mean(scores,axis=0))
    if cfg['alpha']<1:s=cfg['alpha']*s+(1-cfg['alpha'])*normalized(base_scores(bank,arrays,path/'baseline'))
    return apply_gate(bank,s,'parse_and_repair',4).astype(int).tolist()
