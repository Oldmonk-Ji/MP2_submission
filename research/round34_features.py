"""Explicit question/operand/target alignments, hashed learned soft evidence."""
import re
from collections import defaultdict
from fractions import Fraction
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize
import numpy as np
from round33_text import names
from round4_quantities import STOP

DIM=2**18


def features(question,item):
    if not item.get('informative',False):return {'unknown':1.}
    entities=names(question)
    def words(s):
        result=[]
        for w in re.findall(r"[a-z]+(?:['’]s)?",s.lower()):
            stem=w.replace('’',"'").removesuffix("'s")
            w=entities.get(stem,w)
            if w not in STOP:result.append(w)
        return result
    qs=item['quantities'];local=[];owners=[]
    for q in qs:
        left=words(question[:q['start']]);right=words(question[q['end']:])
        local.append(left[-4:]+right[:4])
        owners.append(next((w for w in reversed(left) if w.startswith('entity')),'unknown'))
    target=words(re.split(r'[.!]',question)[-1])[-24:]
    global_words=words(question)
    cues=sorted(set(global_words)|{a+'_'+b for a,b in zip(global_words,global_words[1:])})[:100]
    f={};relations=defaultdict(set);paths=defaultdict(set)
    def add(s):f[s]=1.
    def walk(p,path=()):
        if p[0]=='N':paths[p[1]].add('_'.join(path[-4:]) or 'root');return {p[1]}
        if p[0]=='L':
            add('literal:'+str(p[1]));return set()
        a=walk(p[1],path+(p[0]+'L',));b=walk(p[2],path+(p[0]+'R',))
        for i in a:
            for j in b:
                if i==j:continue
                lo,hi=sorted((i,j));op=p[0]
                if i>j and op in ('SUB','DIV'):op='R'+op
                relations[(lo,hi)].add(op)
        add('op:'+p[0]);return a|b
    used=walk(item['program']);root=item['program'][0]
    for w in cues:add('root:'+root+':'+w)
    for w in target:add('target-root:'+root+':'+w)
    for i,pp in paths.items():
        for path in pp:
            add('path:'+path)
            add('target-owner:'+path+':'+str(owners[i] in target))
            for w in local[i]:add('local:'+path+':'+w)
            for w in target:add('target:'+path+':'+w)
            for a in local[i]:
                for b in target[:12]:add('align:'+path+':'+a+':'+b)
    for (i,j),ops in relations.items():
        if len(ops)!=1:continue
        op=next(iter(ops));between=words(question[qs[i]['end']:qs[j]['start']])[:25]
        add('pair:'+op+':same-owner:'+str(owners[i]==owners[j]))
        add('pair:'+op+':left-target:'+str(owners[i] in target)+':right-target:'+str(owners[j] in target))
        for w in between:add('between:'+op+':'+w)
        for a in local[i]:
            for b in local[j]:add('pair-align:'+op+':'+a+':'+b)
        for w in target:add('pair-target:'+op+':'+w)
    if item['clean']:
        for i,q in enumerate(qs):
            if i not in used and Fraction(q['value']) not in (0,1,Fraction(1,2),Fraction(1,4),100):
                for a in local[i]:
                    for b in target[:12]:add('unused:'+a+':'+b)
    return f


def transform(examples):
    hasher=FeatureHasher(n_features=DIM,input_type='dict',alternate_sign=True,dtype=np.float32)
    x=hasher.transform(features(q,p) for q,p in examples)
    return normalize(x,copy=False).tocsr()
