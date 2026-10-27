"""NumPy-only confidence inputs, shared between calibration and deployment."""
import numpy as np


def confidence_features(s,meta):
    valid=meta['valid'];masked=np.where(valid,s,-1e6)
    choice=masked.argmax(1);top=masked[np.arange(len(s)),choice]
    ordered=np.sort(masked,axis=1)
    margin=np.where(valid.sum(1)>1,top-ordered[:,-2],0)
    exp=np.exp(masked-top[:,None]);prob=exp/exp.sum(1,keepdims=True)
    entropy=-(prob*np.log(np.maximum(prob,1e-12))).sum(1)
    votes=[];distinct=[]
    for i,answers in enumerate(meta['answers']):
        selected=answers[choice[i]]
        votes.append(sum(a==selected for a in answers)/4)
        distinct.append(len(set(a for a in answers if a is not None)))
    return np.column_stack([top,margin,s.std(1),entropy,votes,distinct,valid.sum(1)]),choice
