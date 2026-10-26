"""Input-only typed bindings and restricted entity-ledger/time solvers.

No training references, question IDs, annotations or stored predictions are read.
Unresolved owners/targets abstain; the deployment gate lives in the evaluator.
"""
import re
import numpy as np
from round5_age_graph import normalize,time_of

NUMBER=r'\d[\d,]*(?:\.\d+)?(?:/\d+)?'
PERSON=r"[A-Z][a-z]+(?:'s (?:father|mother|sister|brother|grandfather|grandmother))?"
ACTION=r'(?:is buying|is saving|buys?|bought|has|have|had|owns?|ordered|receives?|received|earns?|spends?|spent|saves?|pays?|paid|keeps?|kept|lost|loses?)'
ADJECTIVES={'small','large','new','old','more','additional','another','extra','some','the','a','an'}
def value(s):
    s=s.replace(',','');p=s.split('/');return float(p[0])/float(p[1]) if len(p)==2 else float(s)
def item(s):
    s=re.split(r'\b(?:for|in|from|at|on|and|of)\b',s.strip(),1)[0]
    ws=[w for w in re.findall(r'[a-z]+',s.lower()) if w not in ADJECTIVES]
    if not ws:return None
    if ws[-1].endswith('ies'):ws[-1]=ws[-1][:-3]+'y'
    elif ws[-1].endswith('s') and not ws[-1].endswith('ss'):ws[-1]=ws[-1][:-1]
    return '_'.join(ws)
def fact(kind,owner,obj,v,unit='count',time='now',related=None):
    return dict(kind=kind,owner=owner,item=obj,value=v,unit=unit,time=time,related=related)
def target(kind,owner,obj,unit='count',time='final'):
    return dict(kind=kind,owner=owner,item=obj,unit=unit,time=time)

def extract(question):
    s=normalize(question);facts=[];provenance=[];warnings=[];goal=None
    # An owner requires a named subject governing a relevant verb. A location's
    # capitalization alone is insufficient. Single-subject pronouns are allowed.
    owners=[m.group(1).lower() for m in re.finditer(rf'\b({PERSON})\s+{ACTION}\b',s) if m.group(1).lower() not in ('he','she','it','they','if','the','a','an')]
    owner=owners[0] if owners and len(set(owners))==1 else None
    if owner is None and len(set(owners))>1:warnings.append('multiple named owners: no global pronoun binding')
    def add(f,span):
        if f['owner'] is None or f['item'] is None:return
        facts.append(f);provenance.append(dict(fact_index=len(facts)-1,normalized_span=list(span),text=s[span[0]:span[1]]))
    # Purchase quantity and price must share the same local noun phrase.
    pat=rf'\b({NUMBER})\s+([A-Za-z][A-Za-z\s-]{{0,55}}?)\s+(?:for|at)\s+\$({NUMBER})(\s+(?:each|apiece))?'
    for m in re.finditer(pat,s):
        obj=item(m.group(2))
        if not owner or re.search(r'\b(?:has|buys|bought|is|was|will|paid)\b',m.group(2),re.I):continue
        add(fact('count',owner,obj,value(m.group(1))),m.span())
        # A per-unit tail must never silently become a total price.
        if re.match(r'\s+per\b',s[m.end():],re.I):
            facts.pop();provenance.pop();continue
        unit=bool(m.group(4));stage='before_discount' if not unit and re.search(r'\b(?:discount|off sale)\b',s,re.I) else 'now'
        add(fact('unit_price' if unit else 'total_price',owner,obj,value(m.group(3)),f'dollar/{obj}' if unit else 'dollar',stage),m.span())
    if owner:
        for m in re.finditer(rf'\bpaid with (?:a )?\$({NUMBER})',s,re.I):add(fact('payment',owner,'money',value(m.group(1)),'dollar'),m.span())
        for m in re.finditer(rf'\b(?:budget of|already has|currently has)\s+\$({NUMBER})',s,re.I):add(fact('balance',owner,'money',value(m.group(1)),'dollar','initial'),m.span())
        for m in re.finditer(rf'\bhas\s+\$({NUMBER})\s+(?:remaining|left)',s,re.I):add(fact('balance',owner,'money',value(m.group(1)),'dollar','final'),m.span())
        for m in re.finditer(rf'\bearns?\s+\$({NUMBER})\s+(?:each|per|a)\s+(day|hour|week)',s,re.I):add(fact('earn_rate',owner,'work',value(m.group(1)),f'dollar/{m.group(2).lower()}'),m.span())
        for m in re.finditer(rf'\b({NUMBER})\s+days?\s+(?:a|per|each)\s+week',s,re.I):add(fact('work_rate',owner,'work',value(m.group(1)),'day/week'),m.span())
        # A percentage discount is tied only to one already identified purchase.
        objects={f['item'] for f in facts if f['kind']=='count'}
        for m in re.finditer(rf'\b({NUMBER})%\s+off',s,re.I):
            if len(objects)==1:add(fact('discount',owner,next(iter(objects)),value(m.group(1))/100,'ratio'),m.span())
        # Inventory balance and changes are distinct facts, never all summed.
        stock=None
        for m in re.finditer(rf'\b(?:has|had|owns)\s+({NUMBER})\s+([a-z]+(?:\s+[a-z]+)?)',s):
            obj=item(re.split(r'\b(?:left|remaining|now)\b',m.group(2))[0])
            if obj in ('dime','nickel','dollar','cent'):continue
            if obj:
                stock=obj;add(fact('balance',owner,obj,value(m.group(1)),'count','initial'),m.span())
        for m in re.finditer(rf'\b(ordered|received|lost|loses)\s+({NUMBER})(?:\s+(more|additional))?(?:\s+([a-z]+))?',s,re.I):
            raw=m.group(4);obj=stock if raw is None or raw.lower() in ('and','then','of','them') else item(raw)
            if stock and obj==stock:add(fact('change',owner,obj,value(m.group(2))*(-1 if m.group(1).lower() in ('lost','loses') else 1)),m.span())
        # Target resolution is conservative; no guessed answer from a partial ledger.
        if re.search(r'how much change',s,re.I):goal=target('change',owner,'money','dollar')
        elif re.search(r'how many weeks',s,re.I):goal=target('duration',owner,'work','week')
        elif stock and re.search(r'how many .*\b(?:now|left)\b',s,re.I):goal=target('balance',owner,stock)
        pm=re.search(r'pay per ([a-z]+).*discount',s,re.I)
        if pm:goal=target('unit_price',owner,item(pm.group(1)),f'dollar/{item(pm.group(1))}','after_discount')
    # Explicit named-person age facts; relation endpoints and stages stay visible.
    if re.search(r'\b(?:age|old|older|younger)\b',s,re.I):
        first=re.match(rf'\s*({PERSON})\b',s);first_person=first.group(1).lower() if first else None
        for sm in re.finditer(r'(?:[^.!?]|\.(?=\d))+',s):
            clause=sm.group();dt,_=time_of(clause)
            if dt is None:warnings.append('multiple time scopes');continue
            qm=re.search(r'(?:how old (?:is|will)|how many years old is)\s+(.+?)(?:\?|$)',clause,re.I)
            if qm:
                raw=re.sub(r'\s+(?:be|now|today|in)\b.*','',qm.group(1),flags=re.I).strip().lower()
                if raw.startswith('his ') and first_person:raw=first_person+"'s "+raw[4:]
                goal=target('age',raw,'age','year',dt)
                clause=clause[:qm.start()];dt,_=time_of(clause)
                if dt is None:continue
            pat=rf'\b({PERSON})(?:,\s*[^,]+,)?\s+(?:who )?(?:is|was|turned|will be)\s+(?:currently )?({NUMBER})\s+years?\s+(?:old|now)\b'
            local_subject=None
            for m in re.finditer(pat,clause):
                who=m.group(1).lower();local_subject=who;first_person=first_person or who
                add(fact('age',who,'age',value(m.group(2)),'year',dt),(sm.start()+m.start(),sm.start()+m.end()))
            pat=rf"\b({PERSON})\s+(?:is|will be|was)\s+({NUMBER})(?: times)?\s+(?:as old as |the age of |)(?:({PERSON})(?:'s|')?(?: age)?)"
            for m in re.finditer(pat,clause):
                who=m.group(1).lower();other=m.group(3).lower();first_person=first_person or who
                suffix=clause[m.end():];rt=0 if re.match(r'\s*(?:is )?(?:today|now)\b',suffix,re.I) else dt
                add(fact('age_ratio',who,'age',value(m.group(2)),'ratio',dt,other),(sm.start()+m.start(),sm.start()+m.end()))
                # Preserve endpoint times separately; evaluator checks this too.
                provenance[-1]['right_time']=rt
            pat=rf'(?:(\b{PERSON})\s+)?(?:is|was)\s+(?:now )?({NUMBER})\s+years?\s+(older|younger)\s+than\s+({PERSON})'
            for m in re.finditer(pat,clause):
                who=m.group(1).lower() if m.group(1) else local_subject
                if who:add(fact('age_difference',who,'age',value(m.group(2))*(1 if m.group(3)=='older' else -1),'year',dt,m.group(4).lower()),(sm.start()+m.start(),sm.start()+m.end()))
    bound=set()
    for p in provenance:
        a,b=p['normalized_span'];bound.update(range(a,b))
    # Time phrases are explicit constraints even when not separate amount facts.
    for m in re.finditer(rf'\b(?:in {NUMBER} years?|{NUMBER} years? (?:ago|from now))\b',s,re.I):bound.update(range(m.start(),m.end()))
    unbound=[dict(text=m.group(),span=list(m.span())) for m in re.finditer(NUMBER,s) if any(i not in bound for i in range(m.start(),m.end()))]
    return dict(facts=facts,target=goal,provenance=provenance,warnings=warnings,unbound_numeric_spans=unbound)

def solve_ledger(graph):
    """Only complete supported inventory/change/discount ledgers return a value."""
    facts=graph['facts'];t=graph['target']
    if not t or graph['unbound_numeric_spans'] or graph['warnings']:return None
    owner=t['owner'];fs=[f for f in facts if f['owner']==owner]
    if t['kind']=='balance':
        fs=[f for f in fs if f['item']==t['item']];initial=[f for f in fs if f['kind']=='balance' and f['time']=='initial']
        if len(initial)!=1 or any(f['kind'] not in ('balance','change') for f in fs):return None
        return dict(value=initial[0]['value']+sum(f['value'] for f in fs if f['kind']=='change'),unit=t['unit'],rule='initial plus signed changes',target=t)
    if t['kind']=='change':
        paid=[f['value'] for f in fs if f['kind']=='payment'];counts={f['item']:f['value'] for f in fs if f['kind']=='count'};prices={f['item']:f['value'] for f in fs if f['kind']=='unit_price'}
        if len(paid)!=1 or not counts or set(counts)!=set(prices):return None
        return dict(value=paid[0]-sum(counts[k]*prices[k] for k in counts),unit='dollar',rule='payment minus sum(quantity * unit price)',target=t)
    if t['kind']=='unit_price':
        fs=[f for f in fs if f['item']==t['item']];vs={k:[f['value'] for f in fs if f['kind']==k] for k in ('count','total_price','discount')}
        if any(len(v)!=1 for v in vs.values()) or vs['count'][0]<=0:return None
        return dict(value=vs['total_price'][0]*(1-vs['discount'][0])/vs['count'][0],unit=t['unit'],rule='discounted total / quantity',target=t)
    return None

def solve_time(graph):
    t=graph['target'];fs=[f for f in graph['facts'] if f['kind'] in ('age','age_ratio','age_difference')]
    if not t or t['kind']!='age' or not fs or graph['unbound_numeric_spans'] or graph['warnings']:return None
    names=sorted({f['owner'] for f in fs}|{f['related'] for f in fs if f['related']});rows=[];rhs=[]
    for f in fs:
        if not isinstance(f['time'],(int,float)):return None
        row=np.zeros(len(names));row[names.index(f['owner'])]+=1
        if f['kind']=='age':b=f['value']-f['time']
        elif f['kind']=='age_difference':row[names.index(f['related'])]-=1;b=f['value']
        else:
            row[names.index(f['related'])]-=f['value'];idx=graph['facts'].index(f);p=next(p for p in graph['provenance'] if p['fact_index']==idx);b=f['value']*p.get('right_time',f['time'])-f['time']
        rows.append(row);rhs.append(b)
    a=np.array(rows);b=np.array(rhs)
    if t['owner'] not in names or np.linalg.matrix_rank(a)<len(names):return None
    ages=np.linalg.lstsq(a,b,rcond=None)[0]
    if np.max(np.abs(a@ages-b))>1e-7 or np.any(ages<0):return None
    return dict(value=float(ages[names.index(t['owner'])]+t['time']),unit='year',rule='joint person/time linear equations',target=t,entities=names,matrix=a.tolist(),rhs=b.tolist())
