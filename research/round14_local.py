"""Candidate-conditioned local age constraints, exact arithmetic, original spans.

This is a bounded diagnostic prototype, NOT a general semantic parser. Grammar
outputs are weak labels. It deliberately does not resolve multi-person pronouns.
No references, IDs, fitted statistics, or data files are used by this module.
"""
from fractions import Fraction as F
from itertools import combinations
import re

WORDS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety'.split(), list(range(21)) + list(range(30, 100, 10))))
NUMBER = r'(?:\d+(?:\.\d+)?(?:/\d+)?|' + '|'.join(WORDS) + r')'
NAME = r"[A-Z][a-z]+(?:['’]s (?:father|mother|sister|brother|grandfather|grandmother|dad|mom))?"
VERB = r'(?:is|was|will be|will)'
FACT = re.compile(rf"\b(?P<a>{NAME}) (?P<verb>{VERB}) (?:(?:currently|now) )?(?P<v>{NUMBER}) years? old\b")
RATIO = re.compile(rf"\b(?P<a>{NAME}) (?P<verb>{VERB}) (?P<v>twice|half|{NUMBER} times) as old as (?P<b>{NAME})(?![a-z])")
DIFFERENCE = re.compile(rf"\b(?P<a>{NAME}) (?P<verb>{VERB}) (?P<v>{NUMBER}) years? (?P<direction>older|younger) than (?P<b>{NAME})(?![a-z])")
EXCLUDE = {'he','she','it','they','if','when','the','in','how','what','his','her','their','then','today','now'}


def number(s):
    s=s.lower().strip()
    return F(WORDS[s]) if s in WORDS else F(s)


def entity(s):
    return s.replace('’', "'").lower()


def spans(text):
    # Decimal dots are not sentence boundaries; offsets always refer to ORIGINAL.
    for m in re.finditer(r'(?:[^.!?\n]|\.(?=\d))+[.!?]?', text):
        yield m.start(), m.end(), m.group()


def evidence(text, start, end):
    return dict(start=start, end=end, text=text[start:end])


def time_hits(text):
    out=[]
    for pat, sign in [(rf'\bin ({NUMBER}) years?\b',1), (rf'\b({NUMBER}) years? (?:from now|later)\b',1), (rf'\b({NUMBER}) years? ago\b',-1)]:
        for m in re.finditer(pat,text,re.I): out.append((m.start(), m.end(), sign*number(m.group(1))))
    for m in re.finditer(r'\bnext year\b', text, re.I): out.append((m.start(),m.end(),F(1)))
    return sorted(out)


def times(clause, match):
    # Relative event times cannot silently become the current time.
    if re.search(r'\b(?:when|once|until|by the time|born|birthday)\b',clause,re.I): return None
    if re.search(r'\b(?:not|never|assume|suppose|hypothetically)\b', clause,re.I): return None
    hits=time_hits(clause)
    if len(hits)>1: return None
    t=hits[0][2] if hits else F(0)
    verb=match.group('verb')
    if not hits and verb in ('was','will','will be'): return None
    left=t
    right=t
    if 'b' in match.groupdict():
        suffix=clause[match.end():]
        if re.match(r'\s*(?:is\s+)?(?:now|today|currently)\b',suffix,re.I): right=F(0)
        # A time phrase wholly after the RHS, without sentence-level scope, is
        # ambiguous: it could modify RHS only. Do not force a shared time.
        if hits and hits[0][0]>=match.end() and not re.match(r'\s*(?:is\s+)?(?:now|today|currently)\b',suffix,re.I): return None
    return left,right


def equation(kind, a, b, v, lt=F(0), rt=F(0), span=None):
    coeff={a:F(1)}
    if kind=='age_fact': rhs=v-lt
    elif kind=='age_ratio':
        coeff[b]=coeff.get(b,F(0))-v
        rhs=v*rt-lt
    else:
        coeff[b]=coeff.get(b,F(0))-1
        rhs=v+rt-lt
    return dict(kind=kind,left=a,right=b,value=str(v),left_time=str(lt),right_time=str(rt),
                coefficients={k:str(x) for k,x in coeff.items() if x},rhs=str(rhs),
                unit='year',evidence=span)


def extract(text, known_entities=None):
    out=[]; unknown=[]
    for start,end,clause in spans(text):
        for regex,kind in [(FACT,'age_fact'),(RATIO,'age_ratio'),(DIFFERENCE,'age_difference')]:
            for m in regex.finditer(clause):
                a=entity(m.group('a')); b=entity(m.group('b')) if 'b' in m.groupdict() else None
                if a in EXCLUDE or b in EXCLUDE: continue
                # Do not extract the tail of a multiword proper name as a person.
                prefix=clause[:m.start()].rstrip()
                if re.search(r'\b[A-Z][a-z]+$',prefix): continue
                es=[a]+([b] if b else [])
                if known_entities is not None and not set(es)<=set(known_entities): continue
                ts=times(clause,m)
                if ts is None:
                    unknown.append(dict(reason='unresolved_time_or_negation',evidence=evidence(text,start,end)));continue
                raw=m.group('v').lower()
                v=F(2) if raw=='twice' else F(1,2) if raw=='half' else number(raw.removesuffix(' times'))
                if kind=='age_difference' and m.group('direction')=='younger':v=-v
                e=equation(kind,a,b,v,*ts,span=evidence(text,start,end))
                e['claim_span']=evidence(text,start+m.start(),start+m.end())
                out.append(e)
    # Duplicate evidence does not increase its weight.
    unique={tuple((k,str(e[k])) for k in ('kind','left','right','value','left_time','right_time')):e for e in out}
    return list(unique.values())[:16],unknown


def rref(equations, names):
    rows=[[F(e['coefficients'].get(n,0)) for n in names]+[F(e['rhs'])] for e in equations]
    pivot=0
    for j in range(len(names)):
        k=next((k for k in range(pivot,len(rows)) if rows[k][j]),None)
        if k is None:continue
        rows[pivot],rows[k]=rows[k],rows[pivot]
        scale=rows[pivot][j];rows[pivot]=[v/scale for v in rows[pivot]]
        for k in range(len(rows)):
            if k!=pivot and rows[k][j]:
                scale=rows[k][j];rows[k]=[a-scale*b for a,b in zip(rows[k],rows[pivot])]
        pivot+=1
        if pivot==len(rows):break
    inconsistent=any(not any(row[:-1]) and row[-1] for row in rows)
    return rows,bool(inconsistent)


def status(equations, claim):
    names=sorted({k for e in equations+[claim] for k in e['coefficients']})
    if len(names)>12:return 'insufficient'
    rows,inconsistent=rref(equations,names)
    if inconsistent:return 'inconsistent_source'
    vec=[F(claim['coefficients'].get(n,0)) for n in names]+[F(claim['rhs'])]
    for row in rows:
        j=next((j for j,x in enumerate(row[:-1]) if x),None)
        if j is not None and vec[j]:
            scale=vec[j]/row[j];vec=[a-scale*b for a,b in zip(vec,row)]
    if any(vec[:-1]):return 'insufficient'
    return 'entailed' if vec[-1]==0 else 'contradicted'


def minimal_support(equations, claim):
    target=status(equations,claim)
    if target not in ('entailed','contradicted'):return []
    keep=list(equations)
    for eq in list(keep):
        fewer=[e for e in keep if e is not eq]
        if status(fewer,claim)==target:keep=fewer
    return keep


def target(question, names):
    for start,end,clause in spans(question):
        m=re.search(rf'\b[Hh]ow old (?:is|will) (?P<a>{NAME}|he|she)\b',clause)
        if not m:continue
        if re.search(r'\b(?:when|until|before|after|was|would)\b',clause,re.I):return None
        a=entity(m.group('a'))
        if a in ('he','she'):
            if len(names)!=1:return None
            a=names[0]
        if a not in names:return None
        hits=time_hits(clause)
        if len(hits)>1:return None
        if 'will' in m.group() and not hits:return None
        return dict(entity=a,time=str(hits[0][2] if hits else F(0)),unit='year',evidence=evidence(question,start,end))
    return None


def inspect(question, candidate_texts, parsed_answers):
    source,unresolved=extract(question)
    names=sorted({k for e in source for k in e['coefficients']})
    goal=target(question,names)
    ambiguity=bool(re.search(r'\btimes (?:older|younger)\b',question,re.I))
    claims=[]
    for j,text in enumerate(candidate_texts):
        relations,_=extract(text,names)
        if goal is not None and parsed_answers[j] is not None:
            eq=equation('age_fact',goal['entity'],None,F(parsed_answers[j]),F(goal['time']))
            eq['evidence']=dict(source='official_parsed_final_answer',value=parsed_answers[j])
            eq['final_target_claim']=True;relations.append(eq)
        for e in relations:
            state='ambiguous' if ambiguity else status(source,e)
            support=minimal_support(source,e) if state in ('entailed','contradicted') else []
            claims.append(dict(candidate=j,claim=e,status=state,question_evidence=[x['evidence'] for x in support],
                               support_relations=support,semantic_gold=False))
    contrasts=[]
    for a,b in combinations(range(len(candidate_texts)),2):
        left=[c for c in claims if c['candidate']==a];right=[c for c in claims if c['candidate']==b]
        # At most one record per candidate pair: final target preferred, else a
        # relation with the same ordered endpoints. Never invert an age ratio.
        pairs=[]
        for x in left:
            for y in right:
                kx=x['claim'];ky=y['claim']
                if tuple(kx.get(k) for k in ('kind','left','right','left_time','right_time'))!=tuple(ky.get(k) for k in ('kind','left','right','left_time','right_time')):continue
                if kx['coefficients']==ky['coefficients'] and kx['rhs']==ky['rhs']:continue
                pairs.append((not(kx.get('final_target_claim') and ky.get('final_target_claim')),x,y))
        if pairs:
            _,x,y=sorted(pairs,key=lambda p:(p[0],str(p[1]['claim']),str(p[2]['claim'])))[0]
            contrasts.append(dict(candidate_pair=[a,b],difference_type=x['claim']['kind'],left=x,right=y,
                                  provenance='original question span + bounded exact arithmetic; weak semantic parse'))
    return dict(source_relations=source,unresolved=unresolved,target=goal,ambiguous=ambiguity,claims=claims,contrasts=contrasts)
