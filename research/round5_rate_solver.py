"""Exact-unit multiplicative constraint closure; incomplete by design.

All explicit quantities must participate once. This assumption is audited on
TRAIN; uniqueness of dimensions alone is not a semantic correctness proof.
"""
import itertools,re,math
from collections import Counter
from fractions import Fraction
import numpy as np
from round5_graph import extract
from features import parse_answer

def solve(question):
    g=extract(question);nodes=g['nodes'];t=g['target']
    if not 2<=len(nodes)<=7 or t['unit'] is None:return None
    if not any(n['per'] for n in nodes):return None
    if any(n['ordinal'] or not n['unit'] or n['unit']=='ratio' or n['value']<=0 for n in nodes):return None
    if t['goal'] in ('profit','difference','remaining'):return None
    if re.search(r'\b(?:before|after|then|already|left|remaining|discount|interest|more|less|extra|except|rest|another|additional)\b',question,re.I):return None
    target=Counter({t['unit']:1})
    if t['per']:target[t['per']]-=1
    target={k:v for k,v in target.items() if v}
    found=[]
    for signs in itertools.product((-1,1),repeat=len(nodes)):
        d=Counter();value=1.
        for n,s in zip(nodes,signs):
            d[n['unit']]+=s
            if n['per']:d[n['per']]-=s
            value*=n['value']**s
        if {k:v for k,v in d.items() if v}==target and math.isfinite(value):found.append((value,signs))
    if not found:return None
    if any(not math.isclose(v,found[0][0],rel_tol=1e-8,abs_tol=1e-8) for v,_ in found):return None
    return dict(value=found[0][0],powers=list(found[0][1]),graph=g)

def proposals(bank,scores,baseline):
    out=baseline.copy();proofs=[]
    for i,row in enumerate(bank):
        proof=solve(row['question']);proofs.append(proof)
        if proof is None:continue
        eligible=[]
        for j,c in enumerate(row['candidates']):
            a=parse_answer(c['text'])
            if a is not None and math.isclose(float(Fraction(a)),proof['value'],rel_tol=1e-6,abs_tol=1e-6):eligible.append(j)
        if eligible:out[i]=max(eligible,key=lambda j:scores[i,j])
    return out,proofs
