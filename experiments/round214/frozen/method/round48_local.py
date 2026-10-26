"""Local, source-grounded relation decisions with compositional program scoring.

Reuses frozen round46 proposals: no extra programs, rules or external data.
Each internal arithmetic node is scored from operand roles and source evidence;
program score is a fixed soft minimum over those relation decisions.
"""
import hashlib,math
from functools import lru_cache
from collections import defaultdict
import numpy as np
from scipy import sparse
from round46_graph import graph,proposals,LIMIT
from round44_programs import tree,locations

DIM=1024

@lru_cache(maxsize=200000)
def index(s):return int.from_bytes(hashlib.blake2b(s.encode(),digest_size=4).digest(),'little')%DIM

def source_ids(p):return {n[1] for _,n in locations(p) if n[0]=='N'}

def node_features(q,p,parent='ROOT'):
    g=graph(q);f=defaultdict(float)
    def add(s,v=1.):f[index(s)]+=v
    op=p[0];add('op:'+op);add('frame:'+parent+':'+op)
    if op in ('N','L'):return f
    sides=[source_ids(p[1]),source_ids(p[2])]
    add('children:'+op+':'+p[1][0]+':'+p[2][0])
    for side,child,ids in zip(('L','R'),p[1:],sides):
        scale=1/math.sqrt(max(1,len(ids)))
        for i in ids:
            n=g['nodes'][i]
            for kind in ('unit','state','kind'):add(op+side+':'+kind+':'+n[kind],scale)
            for t in n['left'][-4:]:add(op+side+':before:'+t,scale)
            for t in n['right'][:5]:add(op+side+':after:'+t,scale)
        if child[0]=='L':add(op+side+':constant:'+str(child[1]))
        add(op+side+':sources:'+str(min(5,len(ids))))
    for i in sides[0]:
        for j in sides[1]:
            a,b=g['nodes'][i],g['nodes'][j];scale=1/math.sqrt(max(1,len(sides[0])*len(sides[1])))
            add('types:'+a['kind']+':'+op+':'+b['kind'],scale)
            add('units:'+a['unit']+':'+op+':'+b['unit'],scale)
            add('states:'+a['state']+':'+op+':'+b['state'],scale)
            add('source_order:'+op+':'+str(i<j),scale)
            add('same_source:'+op+':'+str(i==j),scale)
    for e in g['edges']:
        if e['source'] in sides[0] and e['target'] in sides[1]:direction='forward'
        elif e['source'] in sides[1] and e['target'] in sides[0]:direction='reverse'
        else:continue
        for cue in e['cues']:add('relation:'+cue+':'+op+':'+direction,.5)
    for t in g['target'][-12:]:add('target:'+op+':'+t,.15)
    return f

def at(p,path):
    for k in path:p=p[k]
    return p

def focus(pair):
    pos=tree(pair['positive']);neg=tree(pair['negative'])
    path=tuple(pair.get('decision',{}).get('path',()))
    if not path and 'path' not in pair.get('decision',{}):
        a,b=pos,neg
        while a[0]==b[0] and a[0] not in ('N','L'):
            diff=[k for k in (1,2) if a[k]!=b[k]]
            if len(diff)!=1:break
            path+=tuple(diff);a=a[diff[0]];b=b[diff[0]]
    try:
        if at(pos,path)[0] in ('N','L') or at(neg,path)[0] in ('N','L'):path=path[:-1]
        a,b=at(pos,path),at(neg,path)
        parent='ROOT' if not path else at(pos,path[:-1])[0]+str(path[-1])
    except (IndexError,TypeError):a,b=pos,neg;parent='ROOT';path=()
    return node_features(pair['question'],a,parent),node_features(pair['question'],b,parent),list(path)

def csr(ds):
    values=[];indices=[];ptr=[0]
    for d in ds:
        for k,v in sorted(d.items()):indices.append(k);values.append(v)
        ptr.append(len(values))
    return sparse.csr_matrix((np.asarray(values,np.float32),np.asarray(indices,np.int32),np.asarray(ptr,np.int32)),shape=(len(ds),DIM))

def program_nodes(q,p):
    for path,n in locations(p):
        if n[0] in ('N','L'):continue
        parent='ROOT' if not path else at(p,path[:-1])[0]+str(path[-1])
        yield node_features(q,n,parent)

def packet_matrix(rows,packets):
    ds=[];ptr=[0]
    for r,p in zip(rows,packets):
        for j in range(LIMIT):
            if j<len(p['hypotheses']):ds.extend(program_nodes(r['question'],p['hypotheses'][j]['program']))
            ptr.append(len(ds))
    return csr(ds),np.asarray(ptr,np.int64)

def node_scores(x,w):return (np.maximum(x@w['w0']+w['b0'],0)@w['w1']+w['b1']).reshape(-1)

def aggregate(scores,ptr):
    counts=np.diff(ptr);groups=np.repeat(np.arange(len(counts)),counts)
    sums=np.bincount(groups,weights=np.exp(-np.clip(scores,-20,20)),minlength=len(counts))
    result=-np.log((sums/np.maximum(counts,1)).clip(1e-20))
    result[counts==0]=0
    return result.reshape(-1,LIMIT).astype(np.float32)

def infer(bank,paths):
    weights=[]
    for path in paths:
        with np.load(path/'model.npz') as w:weights.append({k:w[k].copy() for k in w.files})
    scores=[];masks=[];matched=[]
    for start in range(0,len(bank),32):
        rows=bank[start:start+32];packets=[proposals(r) for r in rows];x,ptr=packet_matrix(rows,packets)
        scores.append(sum(aggregate(node_scores(x,w),ptr) for w in weights)/len(weights))
        mask=np.zeros((len(rows),LIMIT),bool);matches=np.zeros((len(rows),LIMIT,4),bool)
        for i,p in enumerate(packets):
            for j,h in enumerate(p['hypotheses']):mask[i,j]=True;matches[i,j,h['matches']]=True
        masks.append(mask);matched.append(matches)
    return np.concatenate(scores),np.concatenate(masks),np.concatenate(matched)
