"""Propagate symbolic dimensions through intermediate expressions (unknown stays unknown)."""
import hashlib
from collections import Counter
import round52_local as old
from round52_local import csr,node_scores
BASE=old.node_features

def abstract(p,rs):
    op=p[0]
    if op=='L':return (),frozenset(),False
    if op=='N':
        r=rs[p[1]];u=r['unit'];d=Counter()
        if r['family']=='fraction':return (),frozenset(),False
        if u=='?':return None,frozenset(),False
        d['money' if r['family']=='money' else u]+=1
        if r['den']:d[r['den']]-=1
        return tuple(sorted((k,v) for k,v in d.items() if v)),frozenset([r['owner']]) - {'?'},False
    a,oa,ea=abstract(p[1],rs);b,ob,eb=abstract(p[2],rs)
    error=ea or eb
    if op in ('ADD','SUB'):
        error |= a is not None and b is not None and a!=b
        dim=a if a==b else None
    elif op in ('MUL','DIV'):
        if a is None or b is None:dim=None
        else:
            d=Counter(dict(a))
            for k,v in b:d[k]+=v*(1 if op=='MUL' else -1)
            dim=tuple(sorted((k,v) for k,v in d.items() if v))
    elif op in ('FLOOR','CEIL'):dim=a
    else:dim=None
    return dim,oa|ob,error

def node_features(q,p,parent='ROOT'):
    f=BASE(q,p,parent);rs,target=old.roles(q)
    if p[0] in ('N','L'):return f
    a,oa,ea=abstract(p[1],rs);b,ob,eb=abstract(p[2],rs);d,owners,error=abstract(p,rs)
    def add(s):
        k=1024+int.from_bytes(hashlib.blake2b(('recursive:'+s).encode(),digest_size=4).digest(),'little')%1024
        f[k]=f.get(k,0)+1
    for key,v in dict(known=d is not None,conflict=error,child_conflict=ea or eb,
        compatible=a==b if a is not None and b is not None else 'unknown',
        owner_overlap=bool(oa&ob),owner_count=min(len(owners),3),
        scalar=d==(),target_unit=any(k in target and v==1 for k,v in (d or ())),
        rate=any(v<0 for _,v in (d or ()))).items():add(f'{parent}:{p[0]}:{key}:{v}')
    return f

def packet_matrix(rows,packets):
    before=old.node_features
    try:old.node_features=node_features;return old.packet_matrix(rows,packets)
    finally:old.node_features=before

def focus(pair):
    before=old.node_features
    try:old.node_features=node_features;return old.focus(pair)
    finally:old.node_features=before
