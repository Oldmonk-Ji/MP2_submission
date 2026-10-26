"""Frozen constraint gate over the existing selector; no extra pretrained model."""
import json
from pathlib import Path
import numpy as np
from round3_selector import scores as base_scores
from round4_gates import apply_gate
from features import metadata,prepare,matrix
from round4_quantities import trace
from selector import model_scores


def scores_and_choices(bank,arrays,checkpoint):
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text())
    s=base_scores(bank,arrays,path/cfg['base'])
    choices=apply_gate(bank,s,cfg['gate_kind'],cfg['margin_limit'])
    return s,choices


def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text())
    if cfg['family']=='constraint_gate':return scores_and_choices(bank,arrays,path)[1].astype(int).tolist()
    if cfg['family']=='mlp' and cfg['features']=='parser':
        base=matrix(prepare(bank,arrays),'context_checked');features=[]
        for row in bank:
            for c in row['candidates']:
                t=trace(row['question'],c['text'])
                features.append([float(t['root_found']),np.log1p(t['steps']),np.log1p(t['bad']),np.log1p(t['collisions']),
                    np.log1p(t['unexplained']),np.log1p(t['resolved']),t['boxed_parse_disagreement'],np.log1p(len(t['quantities']))])
        f=np.asarray(features,np.float32).reshape(len(bank),4,8)
        x=np.concatenate([base,f,f-f.mean(1,keepdims=True)],2)
        s=model_scores(x,path)
        return np.where(metadata(bank)['valid'],s,-1e30).argmax(1).astype(int).tolist()
    raise ValueError('unsupported frozen bundle')


def confidence_inputs(scores,meta,choice):
    n=len(choice);masked=np.where(meta['valid'],scores,-1e30)
    other=masked.copy();other[np.arange(n),choice]=-1e30
    margin=scores[np.arange(n),choice]-other.max(1)
    z=scores-scores.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
    entropy=-(p*np.log(np.maximum(p,1e-8))).sum(1)
    votes=[];distinct=[]
    for i,j in enumerate(choice):
        answers=meta['answers'][i];a=answers[j]
        votes.append(sum(v==a and v is not None for v in answers)/4)
        distinct.append(len(set(v for v in answers if v is not None)))
    return np.column_stack([scores[np.arange(n),choice],margin,scores.std(1),entropy,
        votes,distinct,meta['valid'].sum(1),(masked.argmax(1)!=choice).astype(float)])


def predict_with_confidence(bank,arrays,checkpoint):
    if not bank:return [],[]
    s,choice=scores_and_choices(bank,arrays,checkpoint);x=confidence_inputs(s,metadata(bank),choice)
    with np.load(Path(checkpoint)/'confidence_calibrator.npz',allow_pickle=False) as m:z=x@m['weight']+m['bias']
    return choice.astype(int).tolist(),(1/(1+np.exp(-np.clip(z,-50,50)))).tolist()
