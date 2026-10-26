"""Shared source-use and literal-presence decisions plus conditional scope."""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
import re,math,numpy as np
from round139_features import source_features,pair_features,matrix
from round46_graph import STOP
from round52_local import roles
from round76_text import EXCLUDE
from round147_algebra import normalize,relation_map,ABSENT
from round29_programs import quantities

@lru_cache(maxsize=40000)
def constant_features(q,constant):
    c=F(constant);value=str(c);f=defaultdict(float);_,target=roles(q)
    names={s.lower() for s in re.findall(r'\b[A-Z][a-z]+\b',q)}-EXCLUDE-STOP;target_names=names&set(target)
    def token(w):return ('target_entity' if w in target_names else 'other_entity') if w in names else w
    words=[token(w) for w in re.findall(r'[a-z]+',q.lower())]
    f['literal:'+value]=1.;f['sign:'+str((c>0)-(c<0))]=1.;f['integer:'+str(c.denominator==1)]=1.
    f['abs_bin:'+str(max(-4,min(4,math.floor(math.log10(float(abs(c))) if c else -4))))]=1.
    f['denominator:'+str(c.denominator if c.denominator in (1,2,4,10,100,1000) else 'other')]=1.
    for word in set(words)-STOP:
        f['question:'+word]=.5;f['literal:'+value+'|word:'+word]=.5
    for a,b in set(zip(words,words[1:])):
        f['bigram:'+a+'_'+b]=.3;f['literal:'+value+'|bigram:'+a+'_'+b]=.3
    for w in set(target)-STOP:f['literal:'+value+'|target:'+token(w)]=.5
    for cue,flag in [('percent','%' in q or bool(re.search(r'percent',q,re.I))),('money','$' in q or 'dollar' in q.lower()),('remaining',bool(re.search(r'remain|left',q,re.I)))]:f['literal:'+value+'|'+cue+':'+str(flag)]=1.
    return dict(f)

def probabilities(x,w,prefix):
    s=np.asarray(x@w[prefix+'_weight'].T)+w[prefix+'_bias'];s-=s.max(1,keepdims=True);p=np.exp(s);return p/p.sum(1,keepdims=True)

def compose(source,constant,scope):
    present=source[:,1]*constant[:,1]
    return np.concatenate([present[:,None]*scope,(1-present)[:,None]],axis=1)

def predict(scope_x,source_x,constant_x,w):
    return compose(probabilities(source_x,w,'source'),probabilities(constant_x,w,'constant'),probabilities(scope_x,w,'scope'))

def packet(row,programs):
    groups=[];constants=set();q=row['question']
    for roots in programs:
        unique={}
        for p in roots:
            try:normal=normalize(p);unique[repr(normal)]=normal
            except (ValueError,ZeroDivisionError):continue
        maps=[]
        for p in unique.values():
            mapping,cs=relation_map(p);maps.append(mapping);constants.update(cs)
        groups.append(maps)
    constants=sorted(constants,key=lambda s:(F(s),s));pairs=[(i,c) for i in range(len(quantities(q))) for c in constants]
    packets=[[[(-1 if mapping.get(pair,ABSENT) is None else mapping.get(pair,ABSENT)) for pair in pairs] for mapping in maps] for maps in groups]
    return [pair_features(q,i,c) for i,c in pairs],packets,[source_features(q,i) for i,c in pairs],[constant_features(q,c) for i,c in pairs]

def checks():
    q='Paul has 20 dollars. He spends 10 percent. How much does Paul have left?'
    assert constant_features(q,'1/100')==constant_features(q.replace('Paul','George'),'0.01')
    s=np.array([[.2,.8],[.6,.4]]);c=np.array([[.7,.3],[.7,.3]]);r=np.full((2,4),.25);p=compose(s,c,r);assert np.allclose(p.sum(1),1);assert np.allclose(p[:,4],[.76,.88])
    assert compose(np.zeros((0,2)),np.zeros((0,2)),np.zeros((0,4))).shape==(0,5)
    programs=[[('MUL',('N',0),('L','2'))],[],[('N',0)],[]];a=packet(dict(question=q),programs);b=packet(dict(question=q),[programs[i] for i in (3,0,2,1)])
    assert a[0]==b[0] and a[2:]==b[2:] and b[1]==[a[1][i] for i in (3,0,2,1)]
    return dict(name_invariance=True,equivalent_literal_features=True,probability_composition=True,empty_input=True,candidate_permutation=True)
