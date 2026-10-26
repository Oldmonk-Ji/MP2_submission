"""Expose independent intermediate decisions before final candidate selection."""
from round159_factors import source_features,constant_features,pair_features
from round147_algebra import normalize,relation_map,ABSENT
from round29_programs import quantities
from fractions import Fraction as F
import numpy as np

def used_sources(p):
    if p[0]=='N':return {int(p[1])}
    if p[0]=='L':return set()
    return set().union(*(used_sources(c) for c in p[1:]))

def packet(row,programs):
    q=row['question'];n=len(quantities(q));groups=[];constants=set()
    for roots in programs:
        unique={}
        for p in roots:
            try:normal=normalize(p);unique[repr(normal)]=normal
            except (ValueError,ZeroDivisionError):continue
        items=[]
        for p in unique.values():
            mapping,cs=relation_map(p);items.append((used_sources(p),cs,mapping));constants.update(cs)
        groups.append(items)
    constants=sorted(constants,key=lambda s:(F(s),s));pairs=[(i,c) for i in range(n) for c in constants];packets=[]
    for items in groups:
        packets.append([dict(source=[int(i in used) for i in range(n)],constant=[int(c in cs) for c in constants],scope=[-1 if m.get(k,ABSENT) is None else m.get(k,ABSENT) for k in pairs]) for used,cs,m in items])
    return [pair_features(q,i,c) for i,c in pairs],packets,[source_features(q,i) for i in range(n)],[constant_features(q,c) for c in constants]

def binary_summary(p,label):
    label=np.asarray(label,dtype=int)
    if not len(label):return np.zeros(4)
    lp=np.log(p[np.arange(len(label)),label].clip(1e-8));groups=[lp[label==k].mean() for k in (0,1) if (label==k).any()]
    return [float(np.mean(groups)),float(lp.min()),float((p.argmax(1)==label).mean()),float(label.mean())]

def summarize(source,constant,scope,uptr,cptr,ptr,packets):
    out=[]
    for i,candidates in enumerate(packets):
        u=source[uptr[i]:uptr[i+1]];c=constant[cptr[i]:cptr[i+1]];s=scope[ptr[i]:ptr[i+1]];rows=[]
        for roots in candidates:
            stats=[]
            for root in roots:
                label=np.asarray(root['scope'],dtype=int);known=(label>=0)&(label<ABSENT)
                if known.any():
                    ll=label[known];ss=s[known];lp=np.log(ss[np.arange(len(ll)),ll].clip(1e-8));expected=sum(root['source'])*sum(root['constant']);scope_stats=[float(lp.mean()),float(lp.min()),float((ss.argmax(1)==ll).mean()),float(known.sum()/max(1,expected))]
                else:scope_stats=[0.]*4
                stats.append(list(binary_summary(u,root['source']))+list(binary_summary(c,root['constant']))+scope_stats)
            a=np.asarray(stats) if stats else np.zeros((1,12));rows.append(np.r_[[float(bool(roots)),np.log1p(len(roots)),np.log1p(len(u)),np.log1p(len(c))],a.mean(0),a.min(0),a.max(0)])
        out.append(rows)
    return np.asarray(out,np.float32)

def checks():
    row=dict(question='Ann has 10 apples and loses 2 apples. How many remain?');programs=[[('N',0)],[('N',1)],[],[('N',0)]];pf,ps,uf,cf=packet(row,programs);assert not cf and len(uf)==2 and not pf
    u=np.array([[.01,.99],[.99,.01]]);out=summarize(u,np.zeros((0,2)),np.zeros((0,4)),[0,2],[0,0],[0,0],[ps])[0];assert out.shape==(4,40);assert out[0,4]>out[1,4];assert out[2,0]==0
    order=[3,1,0,2];pp=packet(row,[programs[k] for k in order]);assert pp[1]==[ps[k] for k in order];assert np.array_equal(summarize(u,np.zeros((0,2)),np.zeros((0,4)),[0,2],[0,0],[0,0],[pp[1]])[0],out[order])
    return dict(source_evidence_preserved_without_literals=True,missing_program_unknown=True,candidate_permutation=True,output_features=40)
