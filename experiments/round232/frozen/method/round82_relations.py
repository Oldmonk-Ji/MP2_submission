"""Read-once, monotone-chain quantity-pair operations, inspired by Roy & Roth 2015.

No claim of gold semantics for automatically aligned TRAIN references. Repeated
source quantities and unsupported operators are explicitly outside this head.
"""
from collections import Counter,defaultdict
from fractions import Fraction
import math,re
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from round44_programs import locations
from round46_graph import graph,STOP
from round52_local import roles
LABELS=['ADD','SUB','SUB_REVERSE','MUL','DIV','DIV_REVERSE'];DIM=16384

def normalize(p):
    op=p[0]
    if op in ('N','L'):return (op,p[1])
    if op not in ('ADD','SUB','MUL','DIV'):raise ValueError('unsupported operator')
    positive=[];negative=[];chain={'ADD','SUB'} if op in ('ADD','SUB') else {'MUL','DIV'};join='ADD' if op in ('ADD','SUB') else 'MUL';opposite='SUB' if join=='ADD' else 'DIV'
    def collect(node,sign):
        if node[0] in chain:
            collect(node[1],sign);collect(node[2],-sign if node[0]==opposite else sign)
        else:(positive if sign>0 else negative).append(normalize(node))
    collect(p,1)
    def combine(xs):
        xs=sorted(xs,key=repr)
        if not xs:return ('L','0' if join=='ADD' else '1')
        ans=xs[0]
        for x in xs[1:]:ans=(join,ans,x)
        return ans
    return (opposite,combine(positive),combine(negative)) if negative else combine(positive)

def relations(p):
    ns=[v[1] for _,v in locations(p) if v[0]=='N']
    if len(set(ns))!=len(ns):return None
    try:p=normalize(p)
    except ValueError:return None
    out={}
    def walk(node):
        op=node[0]
        if op=='N':return [node[1]]
        if op=='L':return []
        a=walk(node[1]);b=walk(node[2])
        for i in a:
            for j in b:
                label=op+('_REVERSE' if op in ('SUB','DIV') and i>j else '')
                out[tuple(sorted((i,j)))]=LABELS.index(label)
        return a+b
    walk(p);return out

def pair_features(q,i,j):
    assert i<j;g=graph(q);rs,target=roles(q);qs=g['quantities'];a,b=g['nodes'][i],g['nodes'][j];ra,rb=rs[i],rs[j];f=defaultdict(float)
    names={s.lower() for s in re.findall(r'\b[A-Z][a-z]+\b',q)}-{'how','what','the','each','if','there','a'}
    def token(t):return 'entity' if t in names else t
    def add(s,v=1.):f[s]+=v
    for side,n,r in [('A',a,ra),('B',b,rb)]:
        for k in ('kind','state','family'):add(side+':'+k+':'+r[k])
        add(side+':rate:'+str(bool(r['den'])));add(side+':unit_target:'+str(r['unit'] in target));add(side+':owner_target:'+str(r['owner'].lower() in target and r['owner']!='?'))
        for k,v in r['verbs'].items():add(side+':'+k+':'+str(v))
        for direction,ts in [('left',n['left'][-5:]),('right',n['right'][:5])]:
            for t in ts:
                if t not in STOP:add(side+':'+direction+':'+token(t),.5)
            for x,y in zip(ts,ts[1:]):add(side+':'+direction+':'+token(x)+'_'+token(y),.3)
    for k in ('family','kind','state'):add('pair:'+k+':'+ra[k]+':'+rb[k])
    for k in ('unit','owner','den'):add('same_'+k+':'+str(ra[k]==rb[k] and ra[k] not in ('?','')))
    add('A_den_B_unit:'+str(bool(ra['den']) and ra['den']==rb['unit']));add('B_den_A_unit:'+str(bool(rb['den']) and rb['den']==ra['unit']))
    add('same_sentence:'+str(a['sentence_start']==b['sentence_start']));add('distance_bin:'+str(min(j-i,5)))
    va,vb=Fraction(qs[i]['value']),Fraction(qs[j]['value']);add('value_order:'+str((va>vb)-(va<vb)))
    between=re.findall(r'[a-z]+',q[qs[i]['end']:qs[j]['start']].lower());between=between if len(between)<=20 else between[:10]+between[-10:]
    for t in between:
        if t not in STOP:add('between:'+token(t),.3)
    for t in g['target'][-12:]:
        if t not in STOP:add('target:'+token(t),.3)
    return dict(f)

def matrix(fs):return FeatureHasher(n_features=DIM,input_type='dict',alternate_sign=False,dtype=np.float32).transform(fs)

def probabilities(x,w):
    s=np.asarray(x@w['weight'].T)+w['bias'];s-=s.max(1,keepdims=True);p=np.exp(s);return p/p.sum(1,keepdims=True)
