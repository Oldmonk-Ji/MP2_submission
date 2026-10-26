"""Fuse frozen root/flow evidence with normalized quantity-pair relation packets."""
from round80_features import flow,matrix
from round122_features import row_features, matrix as structure_matrix
from round82_relations import relations,pair_features,matrix as pair_matrix
from round78_features import detail
from round46_graph import graph
from round52_local import roles
from round51_programs import candidate_roots
from round44_programs import locations,canonical
from fractions import Fraction
from itertools import combinations
import numpy as np

def extract(row,arm):
    q=row['question'];g=graph(q);nodes=g['nodes'];n=len(nodes);target=set(g['target']);rs,rt=roles(q);values=[float(Fraction(v['value'])) for v in g['quantities']]
    subsets=[set(range(n)),{i for i,v in enumerate(nodes) if v['kind']=='quantity'},
        {i for i,v in enumerate(nodes) if v['kind']=='percent'},
        {i for i,v in enumerate(nodes) if v['kind'] in ('fraction','fraction_denominator','tenths_numerator')},
        {i for i,v in enumerate(nodes) if v['kind']=='ordinal'},
        {i for i,v in enumerate(nodes) if v['state']=='before'},
        {i for i,v in enumerate(nodes) if v['state']=='after'},
        {i for i,v in enumerate(nodes) if v['unit'] in target}]
    root=[];extra=[];flows=[];packets=[];all_programs=[];pairs=list(combinations(range(n),2));mapping={p:i for i,p in enumerate(pairs)}
    for c in row['candidates']:
        roots={repr(canonical(p)):p for p,k in candidate_roots(q,c['text'],g)};all_programs.append(list(roots.values()));use=[];stats=[];details=[];ff={};packet=[]
        for p in roots.values():
            ns=[v for _,v in locations(p)];leaves=[v[1] for v in ns if v[0]=='N'];used=set(leaves);use.append(used)
            stats.append([len(ns),len(leaves)-len(used),sum(v[0]=='L' for v in ns)]+[sum(v[0]==op for v in ns) for op in ('ADD','SUB','MUL','DIV')]);details.append(detail(p,rs,rt,values))
            for k,v in flow(q,g,rs,p,False).items():ff[k]=ff.get(k,0)+v/max(1,len(roots))
            rel=relations(p)
            if rel:packet.append([[mapping[pair],label] for pair,label in sorted(rel.items())])
        f=[float(bool(roots)),np.log1p(len(roots)),np.log1p(n)]
        for subset in subsets:
            cov=[len(s&subset)/max(1,len(subset)) for s in use];f.extend([float(bool(subset)),max(cov,default=0),min(cov,default=0),float(np.mean(cov)) if cov else 0])
        a=np.asarray(stats) if stats else np.zeros((1,7));f.extend(np.log1p(a.mean(0)));f.extend(np.log1p(a.min(0)));root.append(f)
        d=np.asarray(details) if details else np.zeros((1,53));extra.append(np.r_[d.mean(0),d.min(0),d.max(0)]);ff['root_present']=float(bool(roots));flows.append(ff);packets.append(packet)
    return np.asarray(root,np.float32),np.asarray(extra,np.float32),flows,[pair_features(q,i,j) for i,j in pairs],packets,row_features(row,all_programs,arm)

def build(bank,arm,progress=None):
    rr=[];ee=[];ff=[];pf=[];packets=[];ptr=[0];structure=[]
    for i,row in enumerate(bank):
        r,e,f,p,packet,sf=extract(row,arm);structure.extend(sf);rr.append(r);ee.append(e);ff.extend(f);pf.extend(p);packets.append(packet);ptr.append(len(pf))
        if progress is not None and i%500==0:progress(i)
    return np.asarray(rr),np.asarray(ee),matrix(ff),pair_matrix(pf),np.asarray(ptr),packets,structure_matrix(structure)

def summarize(prob,ptr,packets):
    out=[]
    for i,candidates in enumerate(packets):
        p=prob[ptr[i]:ptr[i+1]];rows=[]
        for roots in candidates:
            rr=[];labels=[]
            for root in roots:
                ix=np.asarray([r[0] for r in root]);y=np.asarray([r[1] for r in root]);pp=p[ix];chosen=pp[np.arange(len(ix)),y];lp=np.log(chosen.clip(1e-8));agreement=(pp.argmax(1)==y);margin=chosen-np.partition(pp,4,axis=1)[:,4]
                rr.append([lp.mean(),lp.min(),agreement.mean(),margin.mean(),np.log1p(len(ix)),-(pp*np.log(pp.clip(1e-8))).sum(1).mean()]);labels.append((y,chosen))
            a=np.asarray(rr) if rr else np.zeros((1,6));f=[float(bool(roots)),np.log1p(len(roots)),a[:,0].max(),a[:,0].mean(),a[:,1].max(),a[:,2].max(),a[:,2].mean(),a[:,3].max(),a[:,4].max(),a[:,5].mean()]
            for k in range(6):
                scores=[float(s[y==k].mean()) for y,s in labels if (y==k).any()];f.extend([float(bool(scores)),max(scores,default=0)])
            rows.append(f)
        out.append(rows)
    return np.asarray(out,np.float32)
