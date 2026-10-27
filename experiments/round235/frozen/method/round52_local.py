"""Add identity-independent operand/target relations to lexical node features."""
from collections import defaultdict,Counter
from functools import lru_cache
import re,hashlib,math
import numpy as np
from scipy import sparse
from round48_local import node_features as lexical,source_ids,at,node_scores,aggregate
from round46_graph import graph,LIMIT,words,STOP,singular
from round44_programs import locations,tree

DIM=2048
TIME=set('second minute hour day week month year'.split())
MONEY=set('dollar cent penny nickel dime quarter money cost price salary pay tip'.split())
VERBS={'loss':set('give gives gave returned returns return spent spends spend lost loses lose removes removed remove ate eats eat used uses use sold sells sell'.split()),
       'gain':set('gets got receives received receive earns earned earn bought buys buy checked adds added add grows grew'.split()),
       'comparison':set('more fewer less twice half double triple same'.split())}

@lru_cache(maxsize=12000)
def roles(q):
    g=graph(q);out=[]
    for i,x in enumerate(g['quantities']):
        n=g['nodes'][i];a,b=x['start'],x['end'];right=[singular(t) for t in words(q[b:b+55])[:6]]
        content=[t for t in right if t not in STOP and t not in ('total','additional','identical','other','remaining')]
        unit=content[0] if content else '?'
        context=q[max(n['sentence_start'],a-80):min(len(q),b+60)].lower()
        family='money' if q[max(0,a-1):a]=='$' or unit in MONEY else 'time' if unit in TIME else 'fraction' if n['kind'] in ('percent','fraction','fraction_denominator') else 'count'
        before=q[n['sentence_start']:a];names=re.findall(r'\b[A-Z][a-z]+\b',before)
        owner=names[-1] if names and names[-1] not in ('The','Each','There','If','How','On','In','A') else '?'
        verbs={k:bool(set(words(before[-55:]))&v) for k,v in VERBS.items()}
        den=re.search(r'\b(?:per|each|every)\s+([a-z]+)',q[b:b+65],re.I)
        out.append(dict(unit=unit,family=family,owner=owner,den=singular(den[1].lower()) if den else '',state=n['state'],kind=n['kind'],verbs=verbs,context=context))
    target=[singular(t) for t in g['target']]
    return out,target

def node_features(q,p,parent='ROOT'):
    f=defaultdict(float,lexical(q,p,parent));rs,target=roles(q);op=p[0]
    def add(s,v=1.):
        k=1024+int.from_bytes(hashlib.blake2b(s.encode(),digest_size=4).digest(),'little')%1024;f[k]+=v
    add('op:'+op);add('frame:'+parent+':'+op)
    if op in ('N','L'):return f
    sides=[source_ids(p[1]),source_ids(p[2])]
    for side,child,ids in zip(('L','R'),p[1:],sides):
        scale=1/math.sqrt(max(1,len(ids)))
        add(f'{op}:{side}:child:{child[0]}')
        add(f'{op}:{side}:size:{min(len(ids),5)}')
        for i in ids:
            r=rs[i]
            for k in ('family','kind','state'):add(f'{op}:{side}:{k}:{r[k]}',scale)
            add(f'{op}:{side}:target_unit:{r["unit"] in target}',scale)
            add(f'{op}:{side}:rate:{bool(r["den"])}',scale)
            for k,v in r['verbs'].items():add(f'{op}:{side}:verb:{k}:{v}',scale)
        counts=Counter(n[1] for _,n in locations(child) if n[0]=='N')
        add(f'{op}:{side}:reuse:{min(sum(v-1 for v in counts.values()),4)}')
        add(f'{op}:{side}:remaining:{child[0]=="SUB"}')
    scale=1/math.sqrt(max(1,len(sides[0])*len(sides[1])))
    for i in sides[0]:
        for j in sides[1]:
            a,b=rs[i],rs[j]
            for k in ('unit','owner','den'):
                equal=a[k]==b[k] and a[k] not in ('','?')
                add(f'{op}:same_{k}:{equal}',scale)
            add(f'{op}:dimension:{a["family"]}:{b["family"]}',scale)
            add(f'{op}:cancel_Lrate:{a["den"]==b["unit"] and bool(a["den"])}',scale)
            add(f'{op}:cancel_Rrate:{b["den"]==a["unit"] and bool(b["den"])}',scale)
            add(f'{op}:object_target:{a["unit"] in target}:{b["unit"] in target}',scale)
            add(f'{op}:time_order:{a["state"]}:{b["state"]}',scale)
            add(f'{op}:binding_identity:{i==j}',scale)
    for cue in ('remaining','left','total','more','less','average','each','per','after','before'):
        add(f'{op}:target_intent:{cue}:{cue in target}')
    return f

def focus(pair):
    from round48_local import focus as oldfocus
    _,_,path=oldfocus(pair);pos,neg=tree(pair['positive']),tree(pair['negative'])
    parent='ROOT' if not path else at(pos,path[:-1])[0]+str(path[-1])
    return node_features(pair['question'],at(pos,path),parent),node_features(pair['question'],at(neg,path),parent),path

def csr(ds):
    v=[];idx=[];ptr=[0]
    for d in ds:
        for k,x in sorted(d.items()):idx.append(k);v.append(x)
        ptr.append(len(v))
    return sparse.csr_matrix((np.asarray(v,np.float32),np.asarray(idx,np.int32),np.asarray(ptr,np.int32)),shape=(len(ds),DIM))

def packet_matrix(rows,packets):
    ds=[];ptr=[0]
    for r,packet in zip(rows,packets):
        for j in range(LIMIT):
            if j<len(packet['hypotheses']):
                p=packet['hypotheses'][j]['program']
                for path,n in locations(p):
                    if n[0] not in ('N','L'):
                        parent='ROOT' if not path else at(p,path[:-1])[0]+str(path[-1])
                        ds.append(node_features(r['question'],n,parent))
            ptr.append(len(ds))
    return csr(ds),np.asarray(ptr,np.int64)
