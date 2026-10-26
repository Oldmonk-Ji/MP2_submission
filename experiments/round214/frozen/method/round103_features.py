"""Question/program compatibility with local-structure and whole-template arms."""
import re,math
from collections import Counter,defaultdict
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from round101_templates import canonical
from round29_programs import quantities
DIM=65536
STOP=set('a an the and is are was were has have had to in on at it he she his her their they how what does did do'.split())

def question_tokens(q):
    qs=quantities(q);names={m.group().lower() for m in re.finditer(r'\b[A-Z][a-z]+\b',q)}-{'a','an','the','if','when','after','before','each','every','how','what','there','in','on','at'}
    def words(s):return ['person' if t in names else t for t in re.findall(r'[a-z]+',s.lower())]
    tokens=Counter('q:'+w for w in words(q) if w not in STOP)
    target=re.split(r'[.?!]',q.rstrip('? .'))[-1];tail=words(target)
    tokens.update('t:'+w for w in tail if w not in STOP)
    tokens.update('t2:'+a+'_'+b for a,b in zip(tail,tail[1:]))
    for i,x in enumerate(qs):
        for side,ws in [('L',words(q[:x['start']])[-3:]),('R',words(q[x['end']:])[:3])]:
            tokens.update(f'N{i}:{side}:{w}' for w in ws if w not in STOP)
    return {k:1.+math.log(v) for k,v in tokens.items()}

def local_tags(p):
    tags=Counter()
    def shallow(n):return repr(n) if n[0] in ('N','L') else n[0]
    def walk(n):
        if n[0] in ('N','L'):tags['leaf:'+repr(n)]+=1;return
        tags['branch:'+n[0]+':'+','.join(shallow(c) for c in n[1:])]+=1
        for c in n[1:]:walk(c)
    walk(p);tags['root:'+p[0]]+=1
    return {k:(1.+math.log(v))/math.sqrt(max(1,len(tags))) for k,v in tags.items()}

def row_features(row,packet,arm):
    words=question_tokens(row['question']);result=[]
    for roots in packet:
        programs={}
        for p in roots:
            try:normal=canonical(p);programs[repr(normal)]=normal
            except (ValueError,ZeroDivisionError):continue
        f=defaultdict(float)
        for key,p in programs.items():
            tags=local_tags(p)
            if arm=='whole':tags['whole:'+key]=1.
            for t,v in tags.items():
                f[t]+=v/max(1,len(programs))
                for word,w in words.items():f[t+'|'+word]+=v*w*.2/max(1,len(programs))
        f['supported']=float(bool(programs));result.append(dict(f))
    return result

def matrix(rows):
    return FeatureHasher(n_features=DIM,input_type='dict',alternate_sign=False,dtype=np.float32).transform(rows) if rows else sparse.csr_matrix((0,DIM),dtype=np.float32)

def build(bank,packets,arm):return matrix([d for row,packet in zip(bank,packets) for d in row_features(row,packet,arm)])

def checks():
    p=('SUB',('N',0),('MUL',('N',1),('N',2)));q=('MUL',('SUB',('N',0),('N',1)),('N',2))
    assert canonical(p)!=canonical(q)
    assert canonical(('MUL',('N',0),('MUL',('N',1),('N',2))))==canonical(('MUL',('MUL',('N',2),('N',0)),('N',1)))
    row=dict(question='Anna has 10 dollars and spends 2 dollars on each of 3 items. How much remains?');packet=[[p],[q],[],[p]]
    x=build([row],[packet],'whole').toarray();perm=[3,0,2,1]
    assert np.array_equal(build([row],[[packet[k] for k in perm]],'whole').toarray(),x[perm])
    return dict(scope_distinguished=True,associative_commutative_equivalence=True,candidate_permutation=True)
