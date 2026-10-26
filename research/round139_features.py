"""Soft lexical and typed evidence for a source quantity and an input-derived literal."""
from collections import defaultdict,Counter
from fractions import Fraction as F
from functools import lru_cache
import re,math,numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from round46_graph import graph,STOP
from round52_local import roles
from round76_text import EXCLUDE
DIM=16384

@lru_cache(maxsize=50000)
def source_features(q,i):
    g=graph(q);rs,target=roles(q);r=rs[i];node=g['nodes'][i];f=defaultdict(float)
    names={s.lower() for s in re.findall(r'\b[A-Z][a-z]+\b',q)}-EXCLUDE-STOP
    target_names=names&set(target)
    def token(w):return ('target_entity' if w in target_names else 'other_entity') if w in names else w
    for k in ('kind','family','state'):f['source:'+k+':'+r[k]]=1.
    owner=r['owner'].lower();f['source:explicit_owner:'+str(owner in names)]=1.;f['source:owner_target:'+str(owner in target_names)]=1.
    f['source:unit_target:'+str(r['unit'] in target)]=1.;f['source:rate:'+str(bool(r['den']))]=1.
    for k,v in r['verbs'].items():f['source:'+k+':'+str(v)]=1.
    for side,words in [('L',node['left'][-5:]),('R',node['right'][:5])]:
        for w in words:
            if w not in STOP:f[side+':'+token(w)]+=.5
        for a,b in zip(words,words[1:]):f[side+'2:'+token(a)+'_'+token(b)]+=.3
    for w in target[-12:]:
        if w not in STOP:f['target:'+token(w)]+=.3
    for w in set(re.findall(r'[a-z]+',q.lower()))-STOP:f['question:'+token(w)]+=.1
    return dict(f)

def pair_features(q,i,constant):
    c=F(constant);value=str(c);f=defaultdict(float,source_features(q,i));g=graph(q);rs,target=roles(q);r=rs[i]
    f['literal:'+value]=1.;f['literal:sign:'+str((c>0)-(c<0))]=1.;f['literal:integer:'+str(c.denominator==1)]=1.
    f['literal:abs_bin:'+str(max(-4,min(4,math.floor(math.log10(float(abs(c))) if c else -4))))]=1.
    f['literal:denominator:'+str(c.denominator if c.denominator in (1,2,4,10,100,1000) else 'other')]=1.
    f['literal:equals_source_magnitude:'+str(abs(c)==abs(F(g['quantities'][i]['value'])))]=1.
    for k in ('kind','family','state'):f['literal:'+value+'|'+k+':'+r[k]]=1.
    f['literal:'+value+'|unit_target:'+str(r['unit'] in target)]=1.
    for cue,flag in [('percent','%' in q or bool(re.search(r'percent',q,re.I))),('money','$' in q or 'dollar' in q.lower()),('remaining',bool(re.search(r'remain|left',q,re.I)))]:f['literal:'+value+'|'+cue+':'+str(flag)]=1.
    return dict(f)

def matrix(rows):
    return FeatureHasher(n_features=DIM,input_type='dict',alternate_sign=False,dtype=np.float32).transform(rows) if rows else sparse.csr_matrix((0,DIM),dtype=np.float32)

def probabilities(x,w):
    s=np.asarray(x@w['weight'].T)+w['bias'];s-=s.max(1,keepdims=True);p=np.exp(s);return p/p.sum(1,keepdims=True)

def checks():
    q='Paul has 20 dollars. He spends 10 percent. How much does Paul have left?'
    a=pair_features(q,1,'1/100');b=pair_features(q.replace('Paul','George'),1,'0.01');assert a==b
    assert matrix([]).shape==(0,DIM)
    assert 'source:explicit_owner:False' in a
    return dict(equivalent_fraction_features=True,name_renaming_invariant=True,unresolved_pronoun_not_named_owner=True,empty_matrix_supported=True)
