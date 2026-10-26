"""Discriminative question x arithmetic-trace features, no hard semantic rules.
Partial trace availability is a feature, never a proof of correctness.
"""
from collections import Counter
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from round4_quantities import trace,STOP

DIM=2**17

def row_features(row,context=True):
    words=[w for w in re.findall(r'[a-z]+',row['question'].lower()) if w not in STOP]
    ctx=set(words)|{a+'_'+b for a,b in zip(words,words[1:])}
    # Restrict to short local lexical patterns; no question ID or full question key.
    ctx=sorted(ctx)[:120]
    out=[]
    for c in row['candidates']:
        t=trace(row['question'],c['text']);f={};tokens=[]
        tokens+=['root_'+str(t['root_found']),'bad_'+str(min(t['bad'],3)),'collision_'+str(min(t['collisions'],3)),'steps_'+str(min(t['steps'],6))]
        if t['root_found']:
            def walk(p,depth=0):
                if depth>12:return 'deep'
                op=p[0]
                if op=='q':return 'Q'+str(min(p[1],9))
                if op=='c':return 'K'+str(p[1]) if p[1] in (0,1,2,7,12,24,60,100) else 'K'
                a,b=walk(p[1],depth+1),walk(p[2],depth+1)
                tokens.extend(['op_'+op,'edge_'+op+'_'+p[1][0]+'_'+p[2][0]])
                return '('+a+op+b+')'
            program=walk(t['program']);tokens.extend(['program_'+program,'rootop_'+t['program'][0]])
            for k,cl in enumerate(t['classes']):
                if k>=10:break
                tokens.append('sign_Q'+str(k)+'_'+str(cl))
                if cl and k<len(t['quantities']):
                    q=t['quantities'][k]
                    for word in q['left'][-3:]+q['right'][:3]:
                        if word not in STOP:tokens.append('local_'+word+'_sign'+str(cl))
        counts=Counter(tokens)
        for token,n in counts.items():f['e:'+token]=min(n,3)
        if context:
            for word in ctx:
                for token,n in counts.items():f['x:'+word+'|'+token]=min(n,3)
        out.append(f)
    return out

def transform(bank,context=True):
    h=FeatureHasher(n_features=DIM,input_type='dict',alternate_sign=True,dtype=np.float32)
    x=h.transform(f for row in bank for f in row_features(row,context))
    return x.tocsr()
