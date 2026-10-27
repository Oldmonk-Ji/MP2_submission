"""Actual final-root quantity coverage, using only question/candidate inputs."""
import numpy as np
from round51_programs import candidate_roots
from round44_programs import locations,canonical
from round46_graph import graph

def extract(row):
    q=row['question'];g=graph(q);nodes=g['nodes'];n=len(nodes);all_ids=set(range(n));target=set(g['target'])
    subsets=[all_ids,{i for i,v in enumerate(nodes) if v['kind']=='quantity'},
        {i for i,v in enumerate(nodes) if v['kind']=='percent'},
        {i for i,v in enumerate(nodes) if v['kind'] in ('fraction','fraction_denominator','tenths_numerator')},
        {i for i,v in enumerate(nodes) if v['kind']=='ordinal'},
        {i for i,v in enumerate(nodes) if v['state']=='before'},
        {i for i,v in enumerate(nodes) if v['state']=='after'},
        {i for i,v in enumerate(nodes) if v['unit'] in target}]
    features=[];usages=[]
    for c in row['candidates']:
        roots={repr(canonical(p)):p for p,k in candidate_roots(q,c['text'],g)}
        use=[];stats=[]
        for p in roots.values():
            ns=[v for _,v in locations(p)];leaves=[v[1] for v in ns if v[0]=='N'];used=set(leaves);use.append(used)
            stats.append([len(ns),len(leaves)-len(used),sum(v[0]=='L' for v in ns)]+[sum(v[0]==op for v in ns) for op in ('ADD','SUB','MUL','DIV')])
        f=[float(bool(roots)),np.log1p(len(roots)),np.log1p(n)]
        for subset in subsets:
            cov=[len(s&subset)/max(1,len(subset)) for s in use]
            f.extend([float(bool(subset)),max(cov,default=0),min(cov,default=0),float(np.mean(cov)) if cov else 0])
        a=np.asarray(stats) if stats else np.zeros((1,7));f.extend(np.log1p(a.mean(0)));f.extend(np.log1p(a.min(0)))
        features.append(f);usages.append([sorted(s) for s in use])
    return np.asarray(features,np.float32),usages

def build(bank):return np.asarray([extract(r)[0] for r in bank],np.float32)
