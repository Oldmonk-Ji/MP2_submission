"""Bounded question-only arithmetic hypotheses, not semantic proofs.

Candidates contribute only their parsed final numbers. No candidate derivation,
reference, ID, or fixed embedding enters this branch. Repeated source numbers
are kept as distinct occurrences. All features are soft learned evidence.
"""
import re
from fractions import Fraction
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from round4_quantities import TOKEN,WORDS,STOP
from features import parse_answer

DIM=2**17


def quantities(text):
    result=[]
    for m in TOKEN.finditer(text):
        raw=m.group().lower()
        try:value=float(WORDS[raw]) if raw in WORDS else float(sum(Fraction(p) for p in raw.replace(',','').split()))
        except (ValueError,ZeroDivisionError):continue
        if not np.isfinite(value) or abs(value)>1e9:continue
        left=re.findall(r'[a-z]+',text[max(0,m.start()-50):m.start()].lower())[-3:]
        right=re.findall(r'[a-z]+',text[m.end():m.end()+50].lower())[:3]
        result.append(dict(value=value,words=sorted(set(w for w in left+right if w not in STOP))))
    # Input-only bounded budget; retain early givens and final target quantities.
    return result if len(result)<=8 else result[:4]+result[-4:]


def op(a,b,k):
    if k==0:return a+b
    if k==1:return a-b
    if k==2:return a*b
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):return np.where(np.abs(b)>1e-10,a/b,np.nan)


def row_features(row,context=True):
    qs=quantities(row['question']);n=len(qs)
    vals=np.asarray([q['value'] for q in qs],np.float64)
    answers=[]
    for c in row['candidates']:
        a=parse_answer(c['text']);answers.append(float(Fraction(a)) if a is not None else np.nan)
    answers=np.asarray(answers);patterns=[[] for _ in answers]
    def add(values,descriptors):
        for k,answer in enumerate(answers):
            hits=np.flatnonzero(np.isfinite(values)&(np.abs(values-answer)<=1e-7*max(1,abs(answer))))
            for h in hits[:max(0,64-len(patterns[k]))]:patterns[k].append(descriptors(int(h)))
    for i,a in enumerate(vals):
        for k,z in enumerate(answers):
            if abs(a-z)<=1e-7*max(1,abs(z)):patterns[k].append(('copy',[(i,'copy')]))
    if n>=2:
        ii,jj=np.where(~np.eye(n,dtype=bool))
        for k in range(4):
            keep=ii<jj if k in (0,2) else np.ones(len(ii),bool)
            i,j=ii[keep],jj[keep];values=op(vals[i],vals[j],k)
            add(values,lambda h: (f'op{k}',[(int(i[h]),f'{k}L'),(int(j[h]),f'{k}R')]))
    if n>=3:
        ix=np.asarray([(i,j,k) for i in range(n) for j in range(n) for k in range(n) if len({i,j,k})==3])
        for child in range(4):
            keep=ix[:,0]<ix[:,1] if child in (0,2) else np.ones(len(ix),bool)
            i,j,k=ix[keep].T;inner=op(vals[i],vals[j],child)
            for root in range(4):
                values=op(inner,vals[k],root)
                add(values,lambda h: (f'op{root}_child{child}',[(int(i[h]),f'{root}L{child}L'),
                    (int(j[h]),f'{root}L{child}R'),(int(k[h]),f'{root}R')]))
                if root in (1,3):
                    values=op(vals[k],inner,root)
                    add(values,lambda h: (f'op{root}_rchild{child}',[(int(k[h]),f'{root}L'),
                        (int(i[h]),f'{root}R{child}L'),(int(j[h]),f'{root}R{child}R')]))
    words=[w for w in re.findall(r'[a-z]+',row['question'].lower()) if w not in STOP]
    cue=sorted(set(words)|{a+'_'+b for a,b in zip(words,words[1:])})[:80]
    target=sorted(set(re.findall(r'[a-z]+',re.split(r'[.!]',row['question'])[-1].lower()))-STOP)
    result=[]
    for pats in patterns:
        f={'has_program':float(bool(pats)),'program_count':float(np.log1p(len(pats)))}
        atoms=set()
        for name,roles in pats:
            atoms.add('shape:'+name)
            for i,role in roles:
                atoms.add('position:'+role+':'+str(i))
                if context:
                    for w in qs[i]['words']:atoms.add('role:'+role+':'+w)
            if context:
                for w in cue:atoms.add('context:'+name+':'+w)
                for w in target:atoms.add('target:'+name+':'+w)
        f.update({a:1. for a in atoms});result.append(f)
    return result


def transform(bank,context=True):
    hasher=FeatureHasher(n_features=DIM,input_type='dict',alternate_sign=True,dtype=np.float32)
    return hasher.transform(f for row in bank for f in row_features(row,context)).tocsr()
