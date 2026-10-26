"""Explicit container/rate relation graph, with source spans and target type.

Unlike a dimension-product heuristic, multiplication requires a text-supported
directed edge. Unknown/unconsumed quantities never become arbitrary factors.
"""
import re
from fractions import Fraction
from round4_quantities import TOKEN,WORDS
from round40_budget import singular

N=r'(?:\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|hundred)'
NP=r'[A-Za-z][A-Za-z -]{0,65}?'
END=r'(?=\s*(?:[,.;?!]|$)|\s+\b(?:and|but|with|each|per|in|on|for|to|at|does|do|did|has|have|are|is|need|needs|will|can|went|that|which|every|during|daily)\b)'
STOP=set('a an the total all number of their his her its our there in it these those them flower'.split())
BAD=set('how what bought buy buying went go get make made makes making more less twice half third first second fourth five other another remaining remainder left leftover before after now tomorrow yesterday days day weeks week months month years year hours hour minutes minute seconds second daily every time times enough maximum minimum at least only'.split())

def value(s):return Fraction(WORDS[s.lower()]) if s.lower() in WORDS else Fraction(s.replace(',',''))

def unit(s):
    s=re.split(r'\bof\b',s,1,flags=re.I)[0]
    ws=[('box' if w=='boxes' else singular(w)) for w in re.findall(r'[a-z]+',s.lower()) if w not in STOP]
    if not ws or len(ws)>4 or any(w in BAD for w in ws):return None
    return ' '.join(ws)

def solve(q):
    result=dict(status='unknown',answer=None,facts=[],target=None,issues=[],equations=[])
    def fail(s):result['issues'].append(s);return result
    # State changes, heterogeneous time bases, comparisons and fractions of a
    # state are separate semantic relations and not silently multiplied here.
    if re.search(r'[$%]|\b(?:remaining|left|less|more|fewer|twice|half|third|quarter|percent|average|ratio|before|after|then|now|first|second|third|fourth|another|additional|extra|sold|sell|burst|drops?|lost|loses?|per day|per week|per hour|per minute|per year|daily|weekly|week|weeks|hours|days)\b',q,re.I):
        # "ready to sell" contains no sale event; keep this explicit modal
        # infinitive from the discovery example, without inferring a sale.
        residual=re.sub(r'ready to sell','ready',q,flags=re.I)
        if re.search(r'[$%]|\b(?:remaining|left|less|more|fewer|twice|half|third|quarter|percent|average|ratio|before|after|then|now|first|second|third|fourth|another|additional|extra|sold|sell|burst|drops?|lost|loses?|per day|per week|per hour|per minute|per year|daily|weekly|week|weeks|hours|days)\b',residual,re.I):return fail('unsupported_state_or_target_operation')
    targets=list(re.finditer(r'\bhow many\s+('+NP+r')'+END,q,re.I))
    if len(targets)!=1:return fail('unresolved_target')
    tm=targets[0];target=unit(tm.group(1))
    if not target:return fail('unknown_target_unit')
    if re.search(r'\b(?:each|per)\b',q[tm.start():]):return fail('target_is_a_rate_not_total')
    result['target']=dict(kind='total_count',unit=target,evidence=dict(start=tm.start(),end=tm.end(),text=q[tm.start():tm.end()]))
    covered=[];edges=[];bases=[]
    def span(a,b):return dict(start=a,end=b,text=q[a:b])
    def add_edge(parent,child,factor,a,b,rule):
        if not parent or not child or parent==child:return
        edges.append((parent,child,factor));covered.append((a,b))
        result['facts'].append(dict(kind='per_relation',parent=parent,child=child,value=str(factor),evidence=span(a,b),rule=rule))
    p=re.compile(r'\beach\s+('+NP+r')\s+(?:has|have|contains|holds|needs|requires|gives|is)\s+('+N+r')\s+('+NP+r')'+END,re.I)
    for m in p.finditer(q):add_edge(unit(m.group(1)),unit(m.group(3)),value(m.group(2)),m.start(),m.end(),'explicit_each_subject')
    p=re.compile(r'\b('+N+r')\s+('+NP+r')\s+in\s+each\s+('+NP+r')'+END,re.I)
    for m in p.finditer(q):add_edge(unit(m.group(3)),unit(m.group(2)),value(m.group(1)),m.start(),m.end(),'explicit_in_each')
    p=re.compile(r'\b('+N+r')\s+('+NP+r')\s+per\s+(?:('+N+r')\s+)?('+NP+r')'+END,re.I)
    for m in p.finditer(q):add_edge(unit(m.group(4)),unit(m.group(2)),value(m.group(1))/(value(m.group(3)) if m.group(3) else 1),m.start(),m.end(),'explicit_per')
    # A container count followed by "with N objects each" gives a directed
    # child-per-container edge, not a second independent total.
    p=re.compile(r'\b('+N+r')\s+('+NP+r')\s+with\s+('+N+r')\s+('+NP+r')\s+each\b',re.I)
    for m in p.finditer(q):
        parent=unit(m.group(2));child=unit(m.group(4))
        if parent and child:
            add_edge(parent,child,value(m.group(3)),m.start(3),m.end(),'container_with_each')
    # Every remaining source quantity must be a grounded object count.
    for m in TOKEN.finditer(q):
        if any(a<=m.start()<b for a,b in covered):continue
        tail=q[m.end():];mm=re.match(r'\s+('+NP+r')'+END,tail,re.I)
        u=unit(mm.group(1)) if mm else None
        if not u:return fail('unbound_source_quantity')
        try:v=value(m.group())
        except (ValueError,ZeroDivisionError):return fail('unsupported_number')
        if v<0:return fail('negative_count')
        bases.append((u,v));end=m.end()+mm.end();covered.append((m.start(),end))
        result['facts'].append(dict(kind='total_count',unit=u,value=str(v),evidence=span(m.start(),end),rule='explicit_count'))
    if not edges or not bases:return fail('no_connected_relation_problem')
    # Resolve a short unit only to one explicitly supplied more-specific unit.
    units={target}|{u for u,v in bases}|{u for a,b,v in edges for u in (a,b)}
    def canonical(u):
        matches=[s for s in units if u.split()[-1]==s.split()[-1] and set(u.split())<=set(s.split())]
        maximal=[s for s in matches if not any(set(s.split())<set(t.split()) for t in matches)]
        return maximal[0] if len(maximal)==1 else None
    mapping={u:canonical(u) for u in units}
    if any(v is None for v in mapping.values()):return fail('ambiguous_unit_scope')
    target=mapping[target];edges=[(mapping[a],mapping[b],v) for a,b,v in edges];bases=[(mapping[u],v) for u,v in bases]
    if len({u for u,v in bases})!=len(bases):return fail('repeated_unit_requires_owner_or_event_binding')
    if len({(a,b) for a,b,v in edges})!=len(edges):return fail('repeated_rate_requires_scope_binding')
    outgoing={}
    for j,(a,b,v) in enumerate(edges):outgoing.setdefault(a,[]).append((b,v,j))
    values=[];used=set()
    for u,v in bases:
        paths=[]
        def walk(node,amount,path,seen):
            if node==target:paths.append((amount,path));return
            for child,rate,index in outgoing.get(node,[]):
                if child in seen:continue
                walk(child,amount*rate,path+[index],seen|{child})
        walk(u,v,[],{u})
        if len(paths)!=1:return fail('no_unique_source_to_target_path')
        amount,path=paths[0];values.append(amount);used.update(path)
        result['equations'].append(dict(source=u,source_count=str(v),rate_edges=path,target=target,contribution=str(amount)))
    if used!=set(range(len(edges))):return fail('unconnected_or_unused_relation')
    if len(values)>1 and any(u==target for u,v in bases):return fail('possible_part_whole_overlap')
    answer=sum(values,Fraction(0))
    if answer.denominator!=1:return fail('nonintegral_object_count')
    result.update(status='proved_within_grammar',answer=str(answer))
    return result
