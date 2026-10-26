"""Compare candidate relations at quantity pairs that actually disagree.

Alternative evidence is admitted only where another candidate's extracted roots
unanimously assign one operator to that pair. Labels and IDs never enter here.
"""
import numpy as np

def summarize(prob,ptr,packets):
    out=[]
    for i,candidates in enumerate(packets):
        p=prob[ptr[i]:ptr[i+1]];support=[]
        for roots in candidates:
            maps=[dict(r) for r in roots]
            support.append({k:v for k,v in maps[0].items() if all(m.get(k)==v for m in maps[1:])} if maps else {})
        features=[]
        for j,roots in enumerate(candidates):
            scores=[];pairs=set()
            for root in roots:
                logratios=[];chosen=[];agreements=[]
                for k,y in root:
                    alternatives={m[k] for z,m in enumerate(support) if z!=j and k in m and m[k]!=y}
                    if not alternatives:continue
                    a=float(p[k,y]);b=max(float(p[k,v]) for v in alternatives)
                    logratios.append(np.log(max(a,1e-8))-np.log(max(b,1e-8)));chosen.append(np.log(max(a,1e-8)));agreements.append(a>=b);pairs.add(k)
                if logratios:
                    scores.append([np.mean(logratios),min(logratios),np.mean(chosen),np.mean(agreements),len(logratios)/len(root),np.log1p(len(logratios))])
            a=np.asarray(scores) if scores else np.zeros((1,6))
            features.append(np.r_[float(bool(scores)),np.log1p(len(pairs)),a.max(0),a.mean(0)])
        out.append(features)
    return np.asarray(out,np.float32)
