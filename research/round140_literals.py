"""Input-only literal hypotheses and separately balanced present/absent evidence."""
import numpy as np
from fractions import Fraction as F
from round138_literals import normalize,relation_map,ABSENT
from round139_features import pair_features
from round29_programs import quantities

def packet(row,programs):
    groups=[];constants=set()
    for roots in programs:
        unique={}
        for p in roots:
            try:normal=normalize(p);unique[repr(normal)]=normal
            except (ValueError,ZeroDivisionError):continue
        maps=[]
        for p in unique.values():
            mapping,cs=relation_map(p);maps.append(mapping);constants.update(cs)
        groups.append(maps)
    constants=sorted(constants,key=lambda s:(F(s),s));pairs=[(i,c) for i in range(len(quantities(row['question']))) for c in constants]
    features=[pair_features(row['question'],i,c) for i,c in pairs]
    packets=[[[(-1 if mapping.get(pair,ABSENT) is None else mapping.get(pair,ABSENT)) for pair in pairs] for mapping in maps] for maps in groups]
    return features,packets

def summarize(prob,ptr,packets):
    out=[]
    for i,candidates in enumerate(packets):
        p=prob[ptr[i]:ptr[i+1]];rows=[]
        for roots in candidates:
            stats=[]
            for root in roots:
                label=np.asarray(root,dtype=int);known=label>=0
                if not known.any():stats.append(np.zeros(8));continue
                ll=label[known];pp=p[known];assigned=pp[np.arange(len(ll)),ll].clip(1e-8);lp=np.log(assigned);present=ll!=ABSENT;absent=~present
                pl=float(lp[present].mean()) if present.any() else 0.;al=float(lp[absent].mean()) if absent.any() else 0.
                joint=float(np.mean(([pl] if present.any() else [])+([al] if absent.any() else [])))
                miss=float((1-pp[absent,ABSENT]).max()) if absent.any() else 0.;bad=float((1-assigned[present]).max()) if present.any() else 0.
                agree=pp.argmax(1)==ll;agreement=float(np.mean(([agree[present].mean()] if present.any() else [])+([agree[absent].mean()] if absent.any() else [])))
                stats.append([joint,pl,al,miss,bad,agreement,float(present.mean()),float(known.mean())])
            a=np.asarray(stats) if stats else np.zeros((1,8));rows.append(np.r_[[float(bool(roots)),np.log1p(len(roots)),np.log1p(len(p))],a.mean(0),a.min(0),a.max(0)])
        out.append(rows)
    return np.asarray(out,np.float32)

def checks():
    row=dict(question='Ann has 10 dollars and spends 2 dollars. How much remains?');good=('SUB',('N',0),('N',1));bad=('ADD',('N',0),('N',1));programs=[[good],[bad],[],[good]]
    fs,ps=packet(row,programs);p=np.full((len(fs),5),.005);p[np.arange(len(fs)),ps[0][0]]=.98
    f=summarize(p,np.array([0,len(fs)]),[ps])[0];assert f[0,3]>f[1,3] and f[2,0]==0
    order=[3,1,0,2];ff,pp=packet(row,[programs[k] for k in order]);assert ff==fs and pp==[ps[k] for k in order]
    assert np.array_equal(summarize(p,np.array([0,len(fs)]),[pp])[0],f[order])
    assert summarize(np.zeros((0,5)),np.array([0,0]),[[[[]],[],[],[]]]).shape==(1,4,27)
    return dict(missing_required_factor_penalized=True,missing_program_remains_unknown=True,candidate_permutation=True,empty_literal_universe_supported=True)
