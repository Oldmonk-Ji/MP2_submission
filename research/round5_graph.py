"""Occurrence-preserving, partial entity/unit/stage constraint graphs.

Unknown bindings stay unknown. A unit mismatch is evidence, never a proof that
the official final numeric answer is wrong. No labels or dataset IDs are read.
"""
import re
from collections import Counter
from fractions import Fraction
import numpy as np
from round4_quantities import TOKEN, WORDS, trace

SKIP=set('a an the of for in on at to and or is are was were be has have had he she it his her they them their with how many much what does do did will would can each per than as by from that this more less total remaining additional other new old different same full whole long tall small large red blue green gold silver next last first second third'.split())
ALIASES={'people':'person','children':'child','men':'person','women':'person','boys':'person','girls':'person','students':'person','classmates':'person','parents':'person','siblings':'person','hours':'hour','minutes':'minute','seconds':'second','feet':'foot','inches':'inch','dollars':'dollar','cents':'cent','pounds':'pound','ounces':'ounce'}
PHYSICAL={'hour':'time','minute':'time','second':'time','day':'time','week':'time','month':'time','year':'time','foot':'length','inch':'length','meter':'length','mile':'length','kilometer':'length','pound':'mass','ounce':'mass','gram':'mass','kilogram':'mass','dollar':'money','cent':'money','gallon':'volume','pint':'volume','liter':'volume','quart':'volume'}
VERBS=set('buy buys bought sell sells sold spend spends spent earn earns earned receive receives received give gives gave get gets got lose loses lost add adds added make makes made need needs take takes took leave leaves left cost costs costing pay pays paid have has had will would can run runs walk walks go goes went want wants work works worked save saves saved read reads put puts'.split())

def words(s):return re.findall(r"[a-z]+",s.lower())
def canon(w):
    if w in ALIASES:return ALIASES[w]
    if w.endswith('ies') and len(w)>4:return w[:-3]+'y'
    if w.endswith('s') and not w.endswith('ss') and len(w)>3:return w[:-1]
    return w
def noun(s):
    for w in words(s)[:4]:
        if w in VERBS or w in ('percent','times','timesmore'):return None
        if w not in SKIP and w not in WORDS:return canon(w)
    return None
def stage(s):
    if re.search(r'\b(after|then|later|next|remaining|left|finally|now)\b',s,re.I):return 'after'
    if re.search(r'\b(original|initial|originally|initially|before|already|started|start)\b',s,re.I):return 'before'
    return 'unspecified'
def flow(s):
    if re.search(r'\b(buy|buys|bought|spend|spent|paid|pay|cost)\b',s,re.I):return 'purchase'
    if re.search(r'\b(sell|sells|sold|earn|earned|receive|received)\b',s,re.I):return 'sale'
    return 'unspecified'
def goal(s):
    if re.search(r'\b(profit|profitability)\b',s,re.I):return 'profit'
    if re.search(r'\b(revenue|sales)\b',s,re.I):return 'revenue'
    if re.search(r'\b(difference|more than|less than)\b',s,re.I):return 'difference'
    if re.search(r'\b(left|remaining|remain)\b',s,re.I):return 'remaining'
    if re.search(r'\b(total|altogether|in all)\b',s,re.I):return 'total'
    return 'unspecified'

def extract(question):
    nodes=[];edges=[]
    for m in TOKEN.finditer(question):
        raw=m.group().lower()
        try:value=float(WORDS[raw]) if raw in WORDS else float(sum(Fraction(p) for p in raw.replace(',','').split()))
        except (ValueError,ZeroDivisionError):continue
        # Ordinals identify phases rather than object counts.
        ordinal=raw in ('first','second','third','fourth','fifth')
        left=question[max(0,m.start()-90):m.start()];right=question[m.end():m.end()+70]
        clause=re.split(r'[.!?;,]',left)[-1]
        unit='dollar' if left.rstrip().endswith('$') else ('ratio' if re.match(r'\s*(%|percent\b|times\b)',right) or raw in ('half','twice','double','doubled','triple','tripled') else noun(right))
        # Unsupported written fractions are not interpreted as whole counts.
        fraction_part=(m.end()<len(question) and question[m.end()]=='-') or (m.start()>0 and question[m.start()-1]=='-')
        if ordinal or fraction_part:unit=None
        denominator=None
        rate=re.match(r'\s*(?:([a-z]+)\s+)?(?:per|a|an|each|every)\s+([a-z]+)',right,re.I)
        if rate and (not rate.group(1) or canon(rate.group(1))==unit):denominator=canon(rate.group(2))
        # Explicit repeated containers: "each tray has 5 rows".
        container=re.search(r'\b(?:each|every)\s+(\w+)\s+(?:has|holds|contains|with)\s*$',clause,re.I)
        if container:denominator=canon(container.group(1))
        owners=re.findall(r"\b([A-Z][a-z]+)(?:'s)?\b",clause)
        owner=next((o for o in reversed(owners) if o.lower() not in SKIP|set(WORDS)|{'if','there','when','calculate','before','after'}),None)
        node=dict(id=len(nodes),span=[m.start(),m.end()],raw=m.group(),value=value,entity=unit,unit=unit,per=denominator,owner=owner,stage=stage(clause),flow=flow(clause),ordinal=ordinal)
        nodes.append(node)
        if denominator:edges.append(dict(type='rate',source=node['id'],numerator=unit,denominator=denominator))
    target_match=list(re.finditer(r'\b(?:how many|how much|what)\b',question,re.I))
    tail=question[target_match[-1].start():] if target_match else re.split(r'[.!?]',question.rstrip('.?!'))[-1]
    u=re.search(r'how many\s+([a-z]+)',tail,re.I)
    unit=canon(u.group(1)) if u else None
    if re.search(r'how much\s+(?:money|profit|revenue)|\bdollars?\b|\bprice\b|how much.*\b(?:pay|cost|earn|spend)\b',tail,re.I):unit='dollar'
    if re.search(r'\bage\b',tail,re.I):unit='year'
    if re.search(r'\bhow long\b',tail,re.I):
        length_units=[n['unit'] for n in nodes if PHYSICAL.get(n['unit'])=='length']
        if length_units:unit=length_units[-1]
    if unit in SKIP or unit in VERBS:unit=None
    per=re.search(r'\b(?:per|each|every)\s+([a-z]+)',tail,re.I)
    target=dict(text=tail,unit=unit,per=canon(per.group(1)) if per else None,stage=stage(tail),goal=goal(tail))
    for i,a in enumerate(nodes):
        for b in nodes[i+1:]:
            if a['value']==b['value']:edges.append(dict(type='equal_value_distinct_mentions',source=i,target=b['id']))
            if a['entity'] and a['entity']==b['entity']:edges.append(dict(type='same_entity',source=i,target=b['id'],same_owner=a['owner']==b['owner'],same_stage=a['stage']==b['stage']))
    return dict(nodes=nodes,edges=edges,target=target)

def dimension(unit,per=None):
    if unit is None:return None
    d=Counter()
    if unit!='ratio':d[PHYSICAL.get(unit,unit)]+=1
    if per:d[PHYSICAL.get(per,per)]-=1
    return {k:v for k,v in d.items() if v}

NAMES=['root','ambiguous_bindings','unknown_bindings','dimension_checked','dimension_conflicts','target_dimension_known','target_dimension_match','target_dimension_mismatch','unused_bound_mentions','reused_source_mentions','used_mentions','rate_mentions_used','mixed_owners','mixed_stages','both_purchase_sale','target_profit_missing_flow','target_per','target_goal_agreement','target_goal_conflict','target_unit_in_final','target_unit_other_final','stage_after_initial_only','all_sources_bound','binding_coverage']

def candidate(graph,question,text,reference=False):
    t=trace(question,text,reference=reference,repair_subexpressions=True)
    bindings={};ambiguous=0;unknown=0
    for i,q in enumerate(t['quantities']):
        matches=[n for n in graph['nodes'] if n['value']==q['value'] and not n['ordinal']]
        if len(matches)==1:bindings[i]=matches[0]
        elif len(matches)>1:ambiguous+=1
        else:unknown+=1
    counts=Counter();checked=0;conflicts=0;depth=0
    def walk(p):
        nonlocal checked,conflicts,depth
        depth+=1
        if depth>500:return None
        if not p:return None
        if p[0]=='q':
            counts[p[1]]+=1;n=bindings.get(p[1]);return dimension(n['unit'],n['per']) if n else None
        if p[0]=='c':return {} if p[1] in (0,1,2,3,4,5,10,12,24,60,100,1000) else None
        a=walk(p[1]);b=walk(p[2])
        if a is None or b is None:return None
        if p[0] in ('+','-'):
            checked+=1
            if a!=b:conflicts+=1;return None
            return a
        if p[0] not in ('*','/'):return None
        d=Counter(a)
        for k,v in b.items():d[k]+=v*(1 if p[0]=='*' else -1)
        return {k:v for k,v in d.items() if v}
    result=walk(t['program']) if t['root_found'] else None
    used=[bindings[i] for i in counts if i in bindings];target=graph['target'];td=dimension(target['unit'],target['per'])
    known=result is not None and td is not None
    final=' '.join([l for l in text.splitlines() if l.strip() and not l.strip().startswith('####')][-3:])
    fg=goal(final);owners={n['owner'] for n in used if n['owner']};stages={n['stage'] for n in used if n['stage']!='unspecified'};flows={n['flow'] for n in used}
    finalunits={canon(w) for w in words(final)}
    vals=[t['root_found'],ambiguous,unknown,checked,conflicts,known,known and result==td,known and result!=td,
          sum(i not in counts for i in bindings) if t['root_found'] else 0,sum(max(0,v-1) for i,v in counts.items() if i in bindings),len(used),sum(n['per'] is not None for n in used),
          len(owners)>1,len(stages)>1,'purchase' in flows and 'sale' in flows,target['goal']=='profit' and not {'purchase','sale'}<=flows,
          target['per'] is not None,fg==target['goal'] and fg!='unspecified',fg!=target['goal'] and fg!='unspecified' and target['goal']!='unspecified',
          target['unit'] in finalunits if target['unit'] else False,bool(target['unit']) and target['unit'] not in finalunits,
          target['stage']=='after' and stages=={'before'},bool(counts) and all(i in bindings for i in counts),len(bindings)/max(1,len(t['quantities']))]
    return np.asarray(vals,dtype=np.float32),dict(program=t['program'],bindings={str(i):n['id'] for i,n in bindings.items()},uses=dict(counts),result_dimension=result,target_dimension=td,dimension_conflicts=conflicts)

def features(bank):
    rows=[]
    for row in bank:
        g=extract(row['question']);rows.append([candidate(g,row['question'],c['text'])[0] for c in row['candidates']])
    return np.asarray(rows,np.float32)
