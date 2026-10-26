"""Finite structured relation decisions and deterministic Fraction execution.

Inference proposals depend only on question/candidates. No reference program,
reviewed answer, ID, or reward enters proposal generation or feature hashing.
This is a bounded edit policy, not unrestricted language/program generation.
"""
import ast,re,hashlib,math
from fractions import Fraction
from functools import lru_cache
from collections import Counter,defaultdict
import numpy as np
from scipy import sparse
from round29_programs import quantities,parse,canonical,value,serialize
from features import parse_answer

LIMIT=32;DIM=1024
CONSTANTS={Fraction(x) for x in ('0','1','2','3','4','5','7','10','12','24','60','100','1000','1/2','1/4','1/100')}

def tree(p):return (p[0],p[1]) if p[0] in ('N','L') else (p[0],tree(p[1]),tree(p[2]))
def locations(p,path=()):
    yield path,p
    if p[0] not in ('N','L'):
        yield from locations(p[1],path+(1,));yield from locations(p[2],path+(2,))
def replace(p,path,new):
    if not path:return new
    r=list(p);r[path[0]]=replace(p[path[0]],path[1:],new);return tuple(r)
def valid(p,qs):
    ns=[v for _,v in locations(p) if v[0]=='N'];ls=[v for _,v in locations(p) if v[0]=='L']
    if not ns or p[0] in ('N','L') or len(list(locations(p)))>31:return None
    if any(Fraction(v[1]) not in CONSTANTS for v in ls):return None
    try:
        a=value(p,qs)
        return str(a) if 0<=a<=10**12 else None
    except (ZeroDivisionError,ValueError,IndexError,OverflowError):return None

def edits(p,qs):
    """Each action changes executable structure; metadata alone earns no reward."""
    for path,n in locations(p):
        op=n[0]
        if op in ('N','L'):
            # Object/amount binding points to an occurrence, not numeric identity.
            for j in range(len(qs)):
                if op=='N' and j==n[1]:continue
                if op=='L' and Fraction(n[1])!=Fraction(qs[j]['value']):continue
                yield 'object_binding',replace(p,path,('N',j)),dict(path=path,source=j)
            continue
        a,b=n[1:]
        if op in ('SUB','DIV'):yield 'operation_direction',replace(p,path,(op,b,a)),dict(path=path,operation=op,orientation='reversed')
        for new in {'ADD':['SUB'],'SUB':['ADD'],'MUL':['DIV'],'DIV':['MUL']}[op]:
            yield 'operation_direction',replace(p,path,(new,a,b)),dict(path=path,operation=new)
        if op=='MUL':
            for f,base,side in [(a,b,2),(b,a,1)]:
                try:fraction=value(f,qs)
                except (ValueError,ZeroDivisionError):continue
                if not 0<fraction<1:continue
                if base[0]=='SUB':
                    yield 'fraction_scope',replace(p,path+(side,),base[1]),dict(path=path,scope='original_instead_of_remaining')
                if base[0]=='N':
                    for j in range(len(qs)):
                        if j!=base[1] and Fraction(qs[j]['value'])>1:
                            yield 'fraction_scope',replace(p,path+(side,),('SUB',base,('N',j))),dict(path=path,scope='remaining_instead_of_original',deduction_source=j)
                if base[0]=='SUB' and base[2][0]=='MUL':
                    # Scope alternatives keep a whole prior subtraction attached.
                    yield 'fraction_scope',replace(p,path+(side,),base[2]),dict(path=path,scope='transferred_component')
            for state,factor in [(a,b),(b,a)]:
                if state[0] in ('ADD','SUB'):
                    yield 'time_state',replace(p,path,(state[0],('MUL',state[1],factor),state[2])),dict(path=path,state='change_after_rate')
        if op in ('ADD','SUB') and a[0]=='MUL':
            yield 'time_state',replace(p,path,('MUL',(op,a[1],b),a[2])),dict(path=path,state='change_before_rate')

def proposals(row):
    q=row['question'];qs=quantities(q);answers=[parse_answer(c['text']) for c in row['candidates']]
    originals={};groups=defaultdict(dict)
    for c in row['candidates']:
        r=parse(q,c['text']);p=tree(r['program'])
        if not r['root_found']:continue
        key=repr(canonical(p));ans=valid(p,qs)
        if ans is not None:originals[key]=dict(program=canonical(p),action='candidate_program',decision={},answer=ans)
        for action,pp,decision in edits(p,qs):
            pp=canonical(pp);a=valid(pp,qs)
            if a is not None:
                key=repr(pp);groups[action].setdefault(key,dict(program=pp,action=action,decision=decision,answer=a))
    chosen=[];seen=set()
    # No candidate index enters ordering; permutation invariance by content.
    for key in sorted(originals):chosen.append(originals[key]);seen.add(key)
    buckets={k:sorted(v,key=lambda s:hashlib.sha256(s.encode()).digest()) for k,v in groups.items()}
    while len(chosen)<LIMIT:
        added=False
        for k in ('object_binding','fraction_scope','time_state','operation_direction'):
            while buckets.get(k) and buckets[k][0] in seen:buckets[k].pop(0)
            if buckets.get(k):
                key=buckets[k].pop(0);chosen.append(groups[k][key]);seen.add(key);added=True
                if len(chosen)==LIMIT:break
        if not added:break
    for h in chosen:h['matches']=[j for j,a in enumerate(answers) if a is not None and a==h['answer']]
    return dict(quantities=qs,hypotheses=chosen)

@lru_cache(maxsize=300000)
def hash_index(s):return int.from_bytes(hashlib.blake2b(s.encode(),digest_size=4).digest(),'little')%DIM
def terms(s):return re.findall(r'[a-z]+',s.lower())
def features(q,qs,p):
    f=defaultdict(float)
    def add(s,v=1.):f[hash_index(s)]+=v
    tail=re.split(r'[.?]',q)[-2] if q.endswith('?') else q[-100:];target=terms(tail)[-14:]
    for t in terms(q):add('question:'+t,.05)
    for t in target:add('target:'+t,.2)
    contexts=[]
    for x in qs:
        left=terms(q[max(0,x['start']-35):x['start']])[-3:];right=terms(q[x['end']:x['end']+45])[:5]
        contexts.append(['before_'+t for t in left]+['after_'+t for t in right])
    nodes=list(locations(p));scale=1/max(1,math.sqrt(len(nodes)))
    def walk(n,path):
        op=n[0]
        if op=='N':
            ix=n[1];add('source_position:'+str(ix),scale)
            for t in contexts[ix]:add('used:'+t,scale);add('role:'+path[-2:]+':'+t,scale)
            return {ix}
        if op=='L':add('constant:'+str(n[1]),scale);return set()
        a=walk(n[1],path+op+'L');b=walk(n[2],path+op+'R');add('operation:'+op,scale)
        for side,ixs in [('L',a),('R',b)]:
            for ix in ixs:
                for t in contexts[ix]:add(op+side+':'+t,scale/max(1,len(ixs)))
        for t in target:add('target_op:'+op+':'+t,scale)
        for i in a:
            for j in b:
                add('ordered_sources:'+op+':'+str(i)+':'+str(j),scale)
        return a|b
    used=walk(p,'ROOT');add('root:'+p[0]);add('coverage:'+str(min(5,len(used))));add('unused:'+str(min(5,len(qs)-len(used))))
    # Neither final value nor correctness labels enter features.
    return f

def matrix(rows,packets):
    data=[];indices=[];indptr=[0]
    for r,packet in zip(rows,packets):
        for j in range(LIMIT):
            f=features(r['question'],packet['quantities'],packet['hypotheses'][j]['program']) if j<len(packet['hypotheses']) else {}
            for k,v in sorted(f.items()):indices.append(k);data.append(v)
            indptr.append(len(data))
    return sparse.csr_matrix((np.asarray(data,np.float32),np.asarray(indices,np.int32),np.asarray(indptr,np.int32)),shape=(len(rows)*LIMIT,DIM))

def expression(text):
    def visit(n):
        if isinstance(n,ast.Name) and re.fullmatch(r'n\d+',n.id):return ('N',int(n.id[1:]))
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return ('L',str(Fraction(str(n.value))))
        if isinstance(n,ast.BinOp):return ({ast.Add:'ADD',ast.Sub:'SUB',ast.Mult:'MUL',ast.Div:'DIV'}[type(n.op)],visit(n.left),visit(n.right))
        raise ValueError('unsupported expression')
    return canonical(visit(ast.parse(text,mode='eval').body))
