"""Position-linked question/program tokens; no reference or ID access at inference."""
import re
from fractions import Fraction
from collections import Counter
import numpy as np
from round46_graph import graph
from round51_programs import candidate_roots
from round44_programs import locations,canonical
MAXLEN=240
EXCLUDE=set('the a an how what when who which if after before then each every there this that he she his her they it in on at for from one two three four five six seven eight nine ten monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november december'.split())

def question_tokens(q):
    names={}
    for m in re.finditer(r'\b[A-Z][a-z]+\b',q):
        if m[0].lower() not in EXCLUDE:names.setdefault(m[0].lower(),'name_'+str(len(names)))
    def words(s):return [names.get(w,w) for w in re.findall(r'[a-z]+|[?.,;:%$]',s.lower())]
    g=graph(q);tokens=[];start=0
    for i,v in enumerate(g['quantities']):
        tokens.extend(words(q[start:v['start']]));tokens.extend(['qnum_'+str(i),'kind_'+g['nodes'][i]['kind']])
        val=Fraction(v['value']);tokens.append('value_fraction' if val.denominator!=1 else 'value_small' if abs(val)<=10 else 'value_large')
        if v['raw'].isalpha():tokens.append(v['raw'].lower())
        start=v['end']
    tokens.extend(words(q[start:]));return tokens if len(tokens)<=128 else tokens[:88]+['truncated_question']+tokens[-39:]

def program_tokens(p):
    if p is None:return ['unknown_program']
    if p[0]=='N':return ['qnum_'+str(p[1])]
    if p[0]=='L':return ['constant_'+str(p[1])]
    return ['op_'+p[0].lower()]+program_tokens(p[1])+program_tokens(p[2])

def sentence(q,p):return ['cls']+question_tokens(q)+['sep']+program_tokens(p)[:96]+['end']

def candidate_sentences(row):
    g=graph(row['question']);result=[]
    for c in row['candidates']:
        roots={repr(canonical(p)):p for p,k in candidate_roots(row['question'],c['text'],g) if len(list(locations(p)))<=63}
        # Input-only tie-breaking: prefer source coverage, then a shorter executable tree.
        ps=sorted(roots.values(),key=lambda p:(-len({n[1] for _,n in locations(p) if n[0]=='N'}),len(list(locations(p))),repr(p)))
        result.append(sentence(row['question'],ps[0] if ps else None))
    return result

def vocab(sequences):
    count=Counter(t for seq in sequences for t in seq);return {'pad':0,'unk':1,**{t:i+2 for i,(t,n) in enumerate(sorted(count.items(),key=lambda x:(-x[1],x[0]))[:4094])}}

def encode(sequences,vocabulary):
    ids=np.zeros((len(sequences),MAXLEN),np.int64);segment=np.zeros_like(ids)
    for i,seq in enumerate(sequences):
        seq=seq[:MAXLEN];ids[i,:len(seq)]=[vocabulary.get(t,1) for t in seq]
        pos=seq.index('sep')+1;segment[i,pos:len(seq)]=1
    return ids,segment
