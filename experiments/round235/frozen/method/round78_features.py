"""Whole-expression signed quantity influence and heuristic dimensional consistency.

Exact program structure fixes scope and direction; noun roles remain fallible
lexical hypotheses. No direction is imposed as an unconditional correct label.
"""
import numpy as np
from fractions import Fraction
from round51_programs import candidate_roots
from round44_programs import locations,canonical
from round46_graph import graph
from round52_local import roles
from round56_local import abstract

def dual(p,values):
    n=len(values);op=p[0]
    if op=='N':
        d=np.zeros(n);d[p[1]]=1;return values[p[1]],d
    if op=='L':return float(Fraction(p[1])),np.zeros(n)
    a,da=dual(p[1],values);b,db=dual(p[2],values)
    if op=='ADD':return a+b,da+db
    if op=='SUB':return a-b,da-db
    if op=='MUL':return a*b,b*da+a*db
    if op=='DIV':
        if abs(b)<1e-12:raise ValueError('zero denominator')
        return a/b,(b*da-a*db)/(b*b)
    raise ValueError('non-smooth or unsupported operation')

def detail(p,rs,target,values):
    ns=[v for _,v in locations(p)];d,owners,error=abstract(p,rs)
    adds=[v for v in ns if v[0] in ('ADD','SUB')];known=bad=0
    for node in adds:
        a,_,_=abstract(node[1],rs);b,_,_=abstract(node[2],rs)
        if a is not None and b is not None:known+=1;bad+=int(a!=b)
    out=[float(d is not None),float(error),len(owners),known/max(1,len(adds)),bad/max(1,len(adds)),
         float(d==()),float(any(k in target and v==1 for k,v in d or ())),float(any(v<0 for _,v in d or ()))]
    try:
        value,deriv=dual(p,values);elastic=np.clip(np.asarray(values)*deriv/max(abs(value),1),-10,10)
        if not np.isfinite(elastic).all():raise ValueError('nonfinite')
        out.append(1.)
    except (ValueError,ZeroDivisionError,OverflowError):elastic=np.zeros(len(values));out.append(0.)
    groups=[list(range(len(rs)))]+[[i for i,r in enumerate(rs) if r['family']==k] for k in ('money','time','fraction','count')]
    groups += [[i for i,r in enumerate(rs) if r['state']==k] for k in ('before','after')]
    groups += [[i for i,r in enumerate(rs) if r['verbs'][k]] for k in ('loss','gain')]
    groups += [[i for i,r in enumerate(rs) if r['unit'] in target],[i for i,r in enumerate(rs) if r['den']]]
    for ix in groups:
        a=elastic[ix];out.extend([float(bool(ix)),float(np.maximum(a,0).sum()),float(np.maximum(-a,0).sum()),float(np.abs(a).max()) if len(a) else 0.])
    return out

def extract(row):
    q=row['question'];g=graph(q);nodes=g['nodes'];n=len(nodes);target=set(g['target']);rs,rt=roles(q);values=[float(Fraction(v['value'])) for v in g['quantities']]
    subsets=[set(range(n)),{i for i,v in enumerate(nodes) if v['kind']=='quantity'},
        {i for i,v in enumerate(nodes) if v['kind']=='percent'},
        {i for i,v in enumerate(nodes) if v['kind'] in ('fraction','fraction_denominator','tenths_numerator')},
        {i for i,v in enumerate(nodes) if v['kind']=='ordinal'},
        {i for i,v in enumerate(nodes) if v['state']=='before'},
        {i for i,v in enumerate(nodes) if v['state']=='after'},
        {i for i,v in enumerate(nodes) if v['unit'] in target}]
    features=[];extra=[]
    for c in row['candidates']:
        roots={repr(canonical(p)):p for p,k in candidate_roots(q,c['text'],g)};use=[];stats=[];details=[]
        for p in roots.values():
            ns=[v for _,v in locations(p)];leaves=[v[1] for v in ns if v[0]=='N'];used=set(leaves);use.append(used)
            stats.append([len(ns),len(leaves)-len(used),sum(v[0]=='L' for v in ns)]+[sum(v[0]==op for v in ns) for op in ('ADD','SUB','MUL','DIV')])
            details.append(detail(p,rs,rt,values))
        f=[float(bool(roots)),np.log1p(len(roots)),np.log1p(n)]
        for subset in subsets:
            cov=[len(s&subset)/max(1,len(subset)) for s in use];f.extend([float(bool(subset)),max(cov,default=0),min(cov,default=0),float(np.mean(cov)) if cov else 0])
        a=np.asarray(stats) if stats else np.zeros((1,7));f.extend(np.log1p(a.mean(0)));f.extend(np.log1p(a.min(0)));features.append(f)
        d=np.asarray(details) if details else np.zeros((1,53));extra.append(np.r_[d.mean(0),d.min(0),d.max(0)])
    return np.asarray(features,np.float32),np.asarray(extra,np.float32)

def build(bank):
    rows=[extract(r) for r in bank];return np.asarray([r[0] for r in rows]),np.asarray([r[1] for r in rows])
