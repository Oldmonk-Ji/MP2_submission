"""Corrected arithmetic evidence, local relation decisions and fixed execution."""
from pathlib import Path
import json,numpy as np
from round49_programs import proposals
from round48_local import packet_matrix,aggregate,node_scores,LIMIT
from round46_selector import select
from round25_selector import scores as baseline_scores

def infer(bank,paths):
    weights=[]
    for path in paths:
        with np.load(Path(path)/'model.npz') as w:weights.append({k:w[k].copy() for k in w.files})
    scores=[];masks=[];matched=[]
    for start in range(0,len(bank),32):
        rows=bank[start:start+32];packets=[proposals(r) for r in rows];x,ptr=packet_matrix(rows,packets)
        scores.append(sum(aggregate(node_scores(x,w),ptr) for w in weights)/len(weights))
        mask=np.zeros((len(rows),LIMIT),bool);matches=np.zeros((len(rows),LIMIT,4),bool)
        for i,p in enumerate(packets):
            for j,h in enumerate(p['hypotheses']):mask[i,j]=True;matches[i,j,h['matches']]=True
        masks.append(mask);matched.append(matches)
    return np.concatenate(scores),np.concatenate(masks),np.concatenate(matched)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
    bs=baseline_scores(bank,arrays,path/'baseline');s,mask,matches=infer(bank,[path/p for p in cfg['policies']])
    return select(bank,bs,s,mask,matches,cfg['mode']).astype(int).tolist()
