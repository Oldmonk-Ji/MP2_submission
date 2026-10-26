"""Small explicit linear constraint graph: age(person, time) = age(person,0)+time.

Only named people, explicit age facts, ratios and differences are supported.
Every numeric mention must be consumed; unresolved references cause abstention.
This is a restricted grammar, not a general semantic parser.
"""
import re
import numpy as np
from round4_quantities import WORDS

NUM=r'\d+(?:\.\d+)?'
NAME=r'[A-Z][a-z]+'
def normalize(s):
    s=s.replace('\u2019',"'")
    for word,value in sorted(WORDS.items(),key=lambda a:-len(a[0])):
        if word not in ('first','second','third','fourth','fifth','once','quarter','dozen'):
            s=re.sub(r'\b'+word+r'\b',str(value),s,flags=re.I)
    s=re.sub(r'\ba year ago\b','1 year ago',s,flags=re.I)
    s=re.sub(r'\bnext year\b','in 1 year',s,flags=re.I)
    return s
def time_of(s):
    hits=[]
    for pat,sign in [(rf'\bin ({NUM}) years?\b',1),(rf'\b({NUM}) years? (?:from now|later)\b',1),(rf'\b({NUM}) years? ago\b',-1)]:
        hits.extend((m.start(),m.end(),sign*float(m.group(1))) for m in re.finditer(pat,s,re.I))
    if len(hits)>1:return None,[]
    return (hits[0][2] if hits else 0),[(a,b) for a,b,_ in hits]
def solve(question):
    if not re.search(r'\b(?:age|old|older|younger)\b',question,re.I):return None
    s=normalize(question);equations=[];used=set();people=set();target=None;target_time=0;target_op='single'
    def consume(text,spans,offset):
        for a,b in spans:
            for m in re.finditer(NUM,text[a:b]):used.update(range(offset+a+m.start(),offset+a+m.end()))
    for sm in re.finditer(r'(?:[^.!?]|\.(?=\d))+',s):
        clause=sm.group();offset=sm.start();dt,tsp=time_of(clause)
        if dt is None:return None
        consume(clause,tsp,offset)
        query=re.search(r'(?:how old (?:is|will)|how many years old is)\s+([A-Za-z]+)',clause,re.I)
        if query:
            target=query.group(1).lower();target_time=dt
            clause_for_facts=clause[:query.start()]
        elif re.search(r'\b(?:sum|total|average)\b.*\bages\b',clause,re.I):
            target='all';target_op='mean' if re.search(r'\baverage\b',clause,re.I) else 'sum';target_time=dt;clause_for_facts=''
        else:clause_for_facts=clause
        dt,_=time_of(clause_for_facts)
        if dt is None:return None
        # Ratios at shared time; explicit "today/now" on the RHS overrides it.
        pat=rf"\b({NAME}) (?:is|will be|was) ({NUM})(?: times)? (?:as old as |the age of |)(?:({NAME})(?:'s|')?(?: age)?)"
        for m in re.finditer(pat,clause_for_facts):
            a,k,b=m.group(1).lower(),float(m.group(2)),m.group(3).lower()
            if k<=0:return None
            suffix=clause_for_facts[m.end():]
            rt=0 if re.match(r'\s*(?:is )?(?:today|now)\b',suffix,re.I) else dt
            equations.append(dict(coefficients={a:1,b:-k},rhs=k*rt-dt,kind='ratio',left_time=dt,right_time=rt,source=m.group()))
            people.update((a,b));consume(clause,[(m.start(),m.end())],offset)
        pat=rf'\b({NAME}) (?:is|was|will be) ({NUM}) years? (older|younger) than ({NAME})\b'
        for m in re.finditer(pat,clause_for_facts):
            a,b=m.group(1).lower(),m.group(4).lower();d=float(m.group(2))*(1 if m.group(3)=='older' else -1)
            equations.append(dict(coefficients={a:1,b:-1},rhs=d,kind='difference',left_time=dt,right_time=dt,source=m.group()));people.update((a,b));consume(clause,[(m.start(),m.end())],offset)
        pat=rf'\b({NAME}) (?:is|was|turned|will(?: be)?) (?:currently )?({NUM})(?=\s*(?:years? (?:old|now)|now\b|[,;]|$))'
        for m in re.finditer(pat,clause_for_facts):
            a=m.group(1).lower();v=float(m.group(2));at=0 if 'currently' in m.group() or re.match(r'\s*now\b',clause_for_facts[m.end():]) else dt
            equations.append(dict(coefficients={a:1},rhs=v-at,kind='fact',left_time=at,source=m.group()));people.add(a);consume(clause,[(m.start(),m.end())],offset)
    if target is None or not equations:return None
    if target in ('he','she') and len(people)==1:target=next(iter(people))
    if target!='all' and target not in people:return None
    if any(any(k not in used for k in range(m.start(),m.end())) for m in re.finditer(NUM,s)):return None
    names=sorted(people);a=np.array([[e['coefficients'].get(n,0) for n in names] for e in equations]);b=np.array([e['rhs'] for e in equations])
    if np.linalg.matrix_rank(a)<len(names):return None
    values=np.linalg.lstsq(a,b,rcond=None)[0]
    if np.max(np.abs(a@values-b))>1e-7 or np.any(values<0):return None
    future=values+target_time
    value=float(future.mean() if target_op=='mean' else future.sum()) if target=='all' else float(future[names.index(target)])
    if value<0:return None
    return dict(value=value,unit='year',entities=names,current_ages=values.tolist(),equations=equations,target=dict(entity=target,time=target_time,operation=target_op),all_numeric_mentions_consumed=True)
