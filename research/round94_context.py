"""Condition quantity-pair evidence on what the question asks for."""
import re
from itertools import combinations
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from round82_relations import pair_features as original
from round46_graph import graph,STOP,singular
DIM=8192

def features(q,i,j):
    base=original(q,i,j);g=graph(q)
    names={s.lower() for s in re.findall(r'\b[A-Z][a-z]+\b',q)}-{'how','what','the','each','if','there','a'}
    cues=sorted(set('entity' if t in names else singular(t) for t in g['target'] if t not in STOP and t not in ('how','what','many','much','does','did','do','will','would','be','can','could')))[:12]
    attrs={k:v for k,v in base.items() if k.startswith(('A:','B:','pair:','same_','A_den','B_den'))}
    # Crosses preserve the operator model's ability to ask which operand belongs
    # to the target. In the old additive model, target words shift every pair's
    # class score identically regardless of operand roles.
    return {'asked='+cue+'|'+k:float(v)*.25 for cue in cues for k,v in attrs.items()}

def matrix(rows):return FeatureHasher(n_features=DIM,input_type='dict',alternate_sign=False,dtype=np.float32).transform(rows)

def build(bank):
    return matrix([features(r['question'],i,j) for r in bank for i,j in combinations(range(len(graph(r['question'])['quantities'])),2)])

def join(x,z):return sparse.hstack((x,z),format='csr')
