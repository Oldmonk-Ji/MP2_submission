"""Unknown-aware executable-candidate policy; missing evidence is not falsity."""
from pathlib import Path
import json,numpy as np
from features import parse_answer
from round49_programs import proposals
from round48_local import packet_matrix,aggregate,node_scores,LIMIT
from round46_selector import select as raw_select
from round25_selector import scores as baseline_scores
from round4_gates import apply_gate

def complete(bank,matches):
    valid=np.asarray([[parse_answer(c['text']) is not None for c in r['candidates']] for r in bank],bool)
    supported=matches.any(1)
    return np.all(supported|~valid,axis=1)&valid.any(1)

def select(bank,bs,s,mask,matches,mode='fixed_residual'):
    base=apply_gate(bank,bs,'parse_and_repair',4)
    choice=raw_select(bank,bs,s,mask,matches,mode)
    return np.where(complete(bank,matches),choice,base)

def infer(bank,paths):
    weights=[]
    for path in paths:
        with np.load(Path(path)/'model.npz') as w:weights.append({k:w[k].copy() for k in w.files})
    scores=np.zeros((len(bank),LIMIT),np.float32);mask=np.zeros_like(scores,bool);matches=np.zeros((len(bank),LIMIT,4),bool)
    active=[];packets=[]
    for i,r in enumerate(bank):
        answers=[parse_answer(c['text']) for c in r['candidates']]
        if len(set(answers)-{None})<2:continue
        p=proposals(r);hs=[h for h in p['hypotheses'] if h['matches']]
        supported={j for h in hs for j in h['matches']}
        if any(a is not None and j not in supported for j,a in enumerate(answers)):continue
        # No action can produce an unavailable answer. Compacting masked-out
        # actions does not change the categorical distribution on legal actions.
        p=dict(p,hypotheses=hs);active.append(i);packets.append(p)
    for begin in range(0,len(active),32):
        ix=active[begin:begin+32];ps=packets[begin:begin+32];x,ptr=packet_matrix([bank[i] for i in ix],ps)
        scores[ix]=sum(aggregate(node_scores(x,w),ptr) for w in weights)/len(weights)
        for i,p in zip(ix,ps):
            for j,h in enumerate(p['hypotheses']):mask[i,j]=True;matches[i,j,h['matches']]=True
    return scores,mask,matches

def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
    bs=baseline_scores(bank,arrays,path/'baseline');s,mask,matches=infer(bank,[path/p for p in cfg['policies']])
    return select(bank,bs,s,mask,matches,cfg['mode']).astype(int).tolist()
