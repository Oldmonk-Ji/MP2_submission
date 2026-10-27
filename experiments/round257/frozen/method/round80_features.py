"""Quantity-influence/local-language interactions, with a sign-removal control."""
import re,math
from collections import defaultdict
from fractions import Fraction
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from round46_graph import graph,STOP
from round52_local import roles
from round51_programs import candidate_roots
from round44_programs import locations,canonical
from round78_features import detail,dual
DIM=16384
INTENTS=set('remain remaining left spend spent spending pay paid cost earn earned profit total altogether together difference more less fewer initial initially original start change gained lost needs need buy bought'.split())

def flow(q,g,rs,p,signed):
    values=[float(Fraction(v['value'])) for v in g['quantities']];out=defaultdict(float)
    try:
        value,d=dual(p,values);e=np.clip(np.asarray(values)*d/max(abs(value),1),-10,10)
        if not np.isfinite(e).all():raise ValueError('nonfinite')
    except (ValueError,ZeroDivisionError,OverflowError):return {'unsupported':1.}
    names={s.lower() for s in re.findall(r'\b[A-Z][a-z]+\b',q)}-{'the','how','what','each','there','if','a'}
    target=set(g['target']);intent=sorted(target&INTENTS)[:3] or ['unspecified'];scale=1/math.sqrt(max(1,len(values)))
    for i,(r,n,v) in enumerate(zip(rs,g['nodes'],e)):
        sign='unused' if abs(v)<1e-8 else ('positive' if v>0 else 'negative') if signed else 'active'
        weight=scale*(.25+math.log1p(abs(v)))
        words=[('L',t) for t in n['left'][-5:]]+[('R',t) for t in n['right'][:5]]
        words=[(side,'entity' if t in names else t) for side,t in words if t not in STOP]
        attrs=[f'{k}={r[k]}' for k in ('family','kind','state')]
        attrs+=['unit_target='+str(r['unit'] in target),'owner_target='+str(r['owner'].lower() in target and r['owner']!='?'),'rate='+str(bool(r['den']))]
        attrs += [k+'='+str(vv) for k,vv in r['verbs'].items()]
        for attr in attrs:out[sign+':'+attr]+=weight
        for cue in intent:
            for attr in attrs:out[sign+':asked='+cue+':'+attr]+=weight
            for side,t in words:out[sign+':asked='+cue+':'+side+':'+t]+=weight*.5
    return out

def extract(row):
    q=row['question'];g=graph(q);nodes=g['nodes'];n=len(nodes);target=set(g['target']);rs,rt=roles(q);values=[float(Fraction(v['value'])) for v in g['quantities']]
    subsets=[set(range(n)),{i for i,v in enumerate(nodes) if v['kind']=='quantity'},
        {i for i,v in enumerate(nodes) if v['kind']=='percent'},
        {i for i,v in enumerate(nodes) if v['kind'] in ('fraction','fraction_denominator','tenths_numerator')},
        {i for i,v in enumerate(nodes) if v['kind']=='ordinal'},
        {i for i,v in enumerate(nodes) if v['state']=='before'},
        {i for i,v in enumerate(nodes) if v['state']=='after'},
        {i for i,v in enumerate(nodes) if v['unit'] in target}]
    root=[];extra=[];flows={k:[] for k in ('unsigned','signed')}
    for c in row['candidates']:
        roots={repr(canonical(p)):p for p,k in candidate_roots(q,c['text'],g)};use=[];stats=[];details=[];ff={k:defaultdict(float) for k in flows}
        for p in roots.values():
            ns=[v for _,v in locations(p)];leaves=[v[1] for v in ns if v[0]=='N'];used=set(leaves);use.append(used)
            stats.append([len(ns),len(leaves)-len(used),sum(v[0]=='L' for v in ns)]+[sum(v[0]==op for v in ns) for op in ('ADD','SUB','MUL','DIV')]);details.append(detail(p,rs,rt,values))
            for arm in flows:
                for k,v in flow(q,g,rs,p,arm=='signed').items():ff[arm][k]+=v/max(1,len(roots))
        f=[float(bool(roots)),np.log1p(len(roots)),np.log1p(n)]
        for subset in subsets:
            cov=[len(s&subset)/max(1,len(subset)) for s in use];f.extend([float(bool(subset)),max(cov,default=0),min(cov,default=0),float(np.mean(cov)) if cov else 0])
        a=np.asarray(stats) if stats else np.zeros((1,7));f.extend(np.log1p(a.mean(0)));f.extend(np.log1p(a.min(0)));root.append(f)
        d=np.asarray(details) if details else np.zeros((1,53));extra.append(np.r_[d.mean(0),d.min(0),d.max(0)])
        for arm in flows:ff[arm]['root_present']=float(bool(roots));flows[arm].append(dict(ff[arm]))
    return np.asarray(root,np.float32),np.asarray(extra,np.float32),flows

def matrix(rows):return FeatureHasher(n_features=DIM,input_type='dict',alternate_sign=False,dtype=np.float32).transform(rows)

def build(bank,arm):
    root=[];extra=[];fs=[]
    for row in bank:
        r,e,f=extract(row);root.append(r);extra.append(e);fs.extend(f[arm])
    return np.asarray(root),np.asarray(extra),matrix(fs)
