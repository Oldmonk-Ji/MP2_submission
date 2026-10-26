"""Soft quantity relevance from TRAIN-aligned reference traces, not hard proof."""
import hashlib,math
from collections import defaultdict,Counter
import numpy as np
from scipy import sparse
from scipy.special import expit
from round46_graph import graph
DIM=4096

def matrix(bank):
    ds=[];ptr=[0]
    for row in bank:
        g=graph(row['question']);qs=g['quantities'];duplicates=Counter(q['value'] for q in qs)
        for i,n in enumerate(g['nodes']):
            f=defaultdict(float)
            def add(s,v=1):f[int.from_bytes(hashlib.blake2b(s.encode(),digest_size=4).digest(),'little')%DIM]+=v
            for key in ('unit','kind','state'):add(key+':'+n[key])
            add('rate:'+str(n['rate']));add('relative_position',i/max(1,len(qs)-1));add('duplicates',math.log1p(duplicates[qs[i]['value']]))
            for side,tokens in [('left',n['left'][-5:]),('right',n['right'][:5])]:
                for j,t in enumerate(tokens):add(side+':'+t);add(side+str(j)+':'+t,.5)
                for a,b in zip(tokens,tokens[1:]):add(side+':'+a+'_'+b)
            for t in g['target'][-12:]:add('target:'+t,.3);add('unit_target:'+n['unit']+':'+t,.3)
            add('target_overlap:'+str(n['unit'] in g['target']))
            ds.append(f)
        ptr.append(len(ds))
    rr=[];cc=[];vv=[]
    for i,f in enumerate(ds):
        for j,v in f.items():rr.append(i);cc.append(j);vv.append(v)
    return sparse.csr_matrix((np.asarray(vv,np.float32),(rr,cc)),shape=(len(ds),DIM)),np.asarray(ptr)

def weighted(usage,prob):
    result=[]
    for candidates,p in zip(usage,prob):
        rows=[];n=len(p)
        for roots in candidates:
            ratios=[];missing=[];extras=[]
            for root in roots:
                mask=np.zeros(n,bool);mask[root]=True
                ratios.append(float(p[mask].sum()/max(float(p.sum()),1e-6)))
                missing.append(float(p[~mask].max()) if (~mask).any() else 0)
                extras.append(float((1-p[mask]).sum()/max(len(root),1)))
            rows.append([float(bool(roots)),max(ratios,default=0),min(ratios,default=0),np.mean(ratios) if ratios else 0,
                min(missing,default=0),np.mean(missing) if missing else 0,min(extras,default=0),np.mean(extras) if extras else 0,
                float(p.mean()) if n else 0,float(p.std()) if n else 0])
        result.append(rows)
    return np.asarray(result,np.float32)

def probabilities(x,ptr,w):
    p=expit(x@w['weight']+w['bias']);return [p[a:b] for a,b in zip(ptr,ptr[1:])]
