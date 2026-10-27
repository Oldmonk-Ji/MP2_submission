"""Input-only quantity/relation graph, bounded arithmetic DAGs and exact execution.

No reviewed question/program lookup and no reference access. Graph slots are
lexical hypotheses, not certified general semantic parsing. Unknown stays unknown.
"""
import ast, hashlib, re, math
from collections import defaultdict
from functools import lru_cache
from fractions import Fraction
from itertools import product
import numpy as np
from scipy import sparse
from round44_programs import proposals as old_proposals, locations, tree, canonical, value, CONSTANTS
from round29_programs import quantities
from round4_quantities import arithmetic_steps
from round42_relations import solve as chain_solve
from features import parse_answer

LIMIT=64
DIM=2048
OPS={ast.Add:'ADD',ast.Sub:'SUB',ast.Mult:'MUL',ast.Div:'DIV'}
STOP=set('the a an of to in on at for and has have had is are was were with it he she his her their they that this each per'.split())

def words(s):return re.findall(r'[a-z]+',s.lower())
def singular(w):return w[:-1] if w.endswith('s') and not w.endswith('ss') else w

@lru_cache(maxsize=12000)
def graph(q):
    qs=quantities(q);nodes=[]
    for i,x in enumerate(qs):
        a,b=x['start'],x['end'];start=max(q.rfind('.',0,a),q.rfind(';',0,a),q.rfind('?',0,a))+1
        end=q.find('.',b);end=len(q) if end<0 else end
        left=words(q[max(start,a-60):a])[-6:];right=words(q[b:min(end,b+60)])[:7]
        content=[singular(t) for t in right if t not in STOP]
        unit=content[0] if content else 'unknown'
        context=' '.join(left+right)
        kind='quantity'
        if re.match(r'\s*(?:%|percent\b)',q[b:],re.I):kind='percent'
        elif x['raw'].lower() in ('half','quarter') or '/' in x['raw']:kind='fraction'
        elif re.search(r'\ba\s*$',q[max(0,a-5):a]) and x['raw'].lower() in ('third','fourth','fifth'):kind='fraction_denominator'
        elif x['raw'].lower() in ('first','second','third','fourth','fifth'):kind='ordinal'
        elif re.match(r'-tenths\b',q[b:],re.I):kind='tenths_numerator'
        if q[max(0,a-1):a]=='$':unit='money'
        state='after' if re.search(r'\b(after|remaining|left|now|then)\b',context) else 'before' if re.search(r'\b(before|initially|first|original)\b',context) else 'unspecified'
        rate=bool(re.search(r'\b(each|per|every)\b',context))
        nodes.append(dict(source=i,value=x['value'],span=[a,b],sentence_start=start,
                          kind=kind,unit=unit,state=state,rate=rate,left=left,right=right))
    edges=[]
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):
            between=q[qs[i]['end']:qs[j]['start']].lower()
            if len(between)>150:continue
            cues=[w for w in ('each','per','more','less','fewer','same','remaining','after','before','twice','than','of') if re.search(r'\b'+w+r'\b',between)]
            if cues:edges.append(dict(source=i,target=j,cues=cues,evidence=[qs[i]['end'],qs[j]['start']],status='lexical_hypothesis'))
    target=re.split(r'[.?!]',q.rstrip('? .'))[-1]
    return dict(quantities=qs,nodes=nodes,edges=edges,target=words(target)[-24:])

def valid(p,qs):
    ns=list(locations(p))
    if p[0] in ('N','L') or len(ns)>31 or not any(v[0]=='N' for _,v in ns):return None
    if any(v[0]=='L' and Fraction(v[1]) not in CONSTANTS for _,v in ns):return None
    try:
        a=value(p,qs)
        return str(a) if 0<=a<=10**12 else None
    except (ZeroDivisionError,ValueError,IndexError,OverflowError):return None

def dag_roots(q,text,g):
    """Bind candidate arithmetic to source occurrences and earlier intermediates.

    Multiple equal-valued mentions remain alternatives. Final-answer equality
    only identifies candidate-claimed roots, never a reference-correct label.
    """
    qs=g['quantities'];direct=defaultdict(list);cache=defaultdict(dict)
    for i,x in enumerate(qs):direct[Fraction(x['value'])].append(('N',i))
    def unique(ps,limit=6):
        d={repr(canonical(p)):canonical(p) for p in ps if len(list(locations(p)))<=31}
        return [d[k] for k in sorted(d,key=lambda k:(len(k),k))[:limit]]
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            v=Fraction(str(n.value));ps=list(direct.get(v,[]))+list(cache.get(v,{}).values())
            for i,x in enumerate(qs):
                kind=g['nodes'][i]['kind'];nv=Fraction(x['value'])
                if kind=='percent' and nv/100==v:ps.append(('DIV',('N',i),('L','100')))
                if kind=='fraction_denominator' and nv and 1/nv==v:ps.append(('DIV',('L','1'),('N',i)))
                if kind=='tenths_numerator' and nv/10==v:ps.append(('DIV',('N',i),('L','10')))
            if not ps and v in CONSTANTS:ps=[('L',str(v))]
            return unique(ps)
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            return [('SUB',('L','0'),p) for p in visit(n.operand)] if isinstance(n.op,ast.USub) else visit(n.operand)
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:
            return unique([(OPS[type(n.op)],a,b) for a,b in product(visit(n.left),visit(n.right))],8)
        return []
    roots=[];answer=parse_answer(text)
    for left,right,computed,claimed in arithmetic_steps(text)[:48]:
        try:
            ps=visit(ast.parse(left,mode='eval').body);rv=Fraction(str(claimed))
            for p in ps:
                if valid(p,qs) is None:continue
                # Only a locally correct arithmetic step can define a reusable value.
                if abs(float(value(p,qs)-rv))<=1e-7*max(1,abs(float(rv))):cache[rv][repr(p)]=p
                if answer is not None and abs(float(rv-Fraction(answer)))<=1e-7*max(1,abs(float(rv))):roots.append(p)
            cache[rv]={repr(p):p for p in unique(list(cache[rv].values()),8)}
        except (ValueError,ZeroDivisionError,SyntaxError,OverflowError,RecursionError):continue
    return unique(roots,16)

def chain_program(q,g):
    proof=chain_solve(q)
    if proof['answer'] is None:return []
    qs=g['quantities'];edges=[];bases={}
    def fact(f):
        if f['kind']=='target_period':return ('L','1')
        e=f.get('evidence')
        if not e:return ('L',f['value'])
        inside=[i for i,x in enumerate(qs) if e['start']<=x['start']<e['end']]
        if f.get('quantity_multiplier',1)!=1:
            raw=[i for i in inside if qs[i]['raw'].lower()!='dozen']
            if len(raw)==1:return ('MUL',('N',raw[0]),('L',str(f['quantity_multiplier'])))
        exact=[i for i in inside if qs[i]['value']==f['value']]
        if len(exact)==1:return ('N',exact[0])
        if f['kind']=='per_relation' and len(inside)==2:return ('DIV',('N',inside[0]),('N',inside[1]))
        raise ValueError('unknown binding')
    try:
        for f in proof['facts']:
            if f['kind'] in ('per_relation','unit_conversion'):
                if f.get('rule')=='explicit_activity_period':p=fact(next(x for x in proof['facts'] if x['kind']=='activity_count_per_period' and x['unit']==f['child']))
                else:p=fact(f)
                edges.append(p)
            elif f['kind'] in ('total_count','target_period'):bases[f['unit']]=fact(f)
        result=('L','0')
        for eq in proof['equations']:
            p=bases[eq['source']]
            for j in eq['rate_edges']:p=('MUL',p,edges[j])
            result=('ADD',result,p)
        return [canonical(result)]
    except (ValueError,KeyError,StopIteration):return []

def scope_programs(q,g):
    """Alternative scopes are hypotheses to learn, never proof-based overrides."""
    qs=g['quantities'];nodes=g['nodes'];fractions=[]
    for i,n in enumerate(nodes):
        if n['kind']=='percent':fractions.append(('DIV',('N',i),('L','100')))
        elif n['kind']=='fraction' and 0<Fraction(qs[i]['value'])<1:fractions.append(('N',i))
    if not fractions or len(fractions)>3:return []
    bases=[i for i,n in enumerate(nodes) if n['kind']=='quantity' and Fraction(qs[i]['value'])>1][:4]
    out=[]
    for i in bases:
        base=('N',i);total=('L','0');remaining=base;increasing=base;part=base
        for f in fractions:
            total=('ADD',total,f)
            remaining=('MUL',remaining,('SUB',('L','1'),f))
            increasing=('MUL',increasing,('ADD',('L','1'),f))
            part=('MUL',part,f)
        out.extend([remaining,increasing,part,('MUL',base,('SUB',('L','1'),total))])
        if len(fractions)==3:
            rest=('MUL',base,('SUB',('SUB',('L','1'),fractions[0]),fractions[1]))
            out.append(('MUL',rest,('SUB',('L','1'),fractions[2])))
    return [canonical(p) for p in out]

def proposals(row):
    q=row['question'];g=graph(q);qs=g['quantities'];old=old_proposals(row)
    chosen=list(old['hypotheses']);seen={repr(canonical(tree(h['program']))) for h in chosen}
    groups=defaultdict(dict)
    for c in row['candidates']:
        for p in dag_roots(q,c['text'],g):groups['arithmetic_graph'][repr(p)]=p
    for p in chain_program(q,g):groups['source_rate_graph'][repr(p)]=p
    for p in scope_programs(q,g):groups['scope_graph'][repr(p)]=p
    answers=[parse_answer(c['text']) for c in row['candidates']]
    queues={k:sorted(v,key=lambda key:hashlib.sha256(key.encode()).digest()) for k,v in groups.items()}
    while len(chosen)<LIMIT:
        added=False
        for kind in ('arithmetic_graph','source_rate_graph','scope_graph'):
            while queues.get(kind):
                key=queues[kind].pop(0)
                if key in seen:continue
                p=groups[kind][key];a=valid(p,qs)
                if a is None:continue
                chosen.append(dict(program=p,action=kind,decision={},answer=a,matches=[j for j,x in enumerate(answers) if x is not None and x==a]))
                seen.add(key);added=True;break
            if len(chosen)==LIMIT:break
        if not added:break
    return dict(quantities=qs,hypotheses=chosen,graph=g)

@lru_cache(maxsize=300000)
def index(s):return int.from_bytes(hashlib.blake2b(s.encode(),digest_size=4).digest(),'little')%DIM

def features(q,qs,p):
    g=graph(q);f=defaultdict(float);scale=1/math.sqrt(max(1,len(list(locations(p)))))
    def add(s,v=1.):f[index(s)]+=v
    for t in g['target']:add('target:'+t,.15)
    def walk(n,path):
        op=n[0]
        if op=='L':add('constant:'+str(n[1]),scale);return set()
        if op=='N':
            i=n[1];node=g['nodes'][i]
            for role in ('kind','unit','state'):
                add('leaf_'+role+':'+node[role],scale);add('path:'+path+':'+role+':'+node[role],scale)
            for t in node['left'][-4:]:add('left:'+t,scale);add('role:'+path[-7:]+':left:'+t,scale)
            for t in node['right'][:5]:add('right:'+t,scale);add('role:'+path[-7:]+':right:'+t,scale)
            add('source:'+str(i),scale);return {i}
        a=walk(n[1],path+'/'+op+'L');b=walk(n[2],path+'/'+op+'R')
        add('op:'+op,scale)
        for i in a:
            for j in b:
                left,right=g['nodes'][i],g['nodes'][j]
                add('typed_edge:'+left['unit']+':'+op+':'+right['unit'],scale)
                add('state_edge:'+left['state']+':'+op+':'+right['state'],scale)
                add('kind_edge:'+left['kind']+':'+op+':'+right['kind'],scale)
                add('ordered:'+op+':'+str(i)+':'+str(j),scale)
                add('same_sentence:'+op+':'+str(left['sentence_start']==right['sentence_start']),scale)
        for edge in g['edges']:
            if (edge['source'] in a and edge['target'] in b) or (edge['source'] in b and edge['target'] in a):
                orientation='forward' if edge['source'] in a else 'reverse'
                for cue in edge['cues']:add('relation:'+cue+':'+op+':'+orientation,scale)
        for t in g['target'][-10:]:add('target_op:'+op+':'+t,scale)
        return a|b
    used=walk(p,'root');add('root:'+p[0]);add('coverage:'+str(min(10,len(used))))
    for i,n in enumerate(g['nodes']):
        if i not in used:add('unused:'+n['kind']+':'+n['unit'],.1)
    return f

def csr(ds):
    data=[];indices=[];indptr=[0]
    for d in ds:
        for j,v in sorted(d.items()):indices.append(j);data.append(v)
        indptr.append(len(data))
    return sparse.csr_matrix((np.asarray(data,np.float32),np.asarray(indices,np.int32),np.asarray(indptr,np.int32)),shape=(len(ds),DIM))

def matrix(rows,packets):
    return csr([features(r['question'],p['quantities'],p['hypotheses'][j]['program']) if j<len(p['hypotheses']) else {}
                for r,p in zip(rows,packets) for j in range(LIMIT)])
