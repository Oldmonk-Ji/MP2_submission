"""Explicit lexical agreement only; unknown context retains alternatives."""
import re
from fractions import Fraction as F
from round46_graph import singular
STOP=set('a an the of to as and or is are was were has have had more less fewer times total each per remaining left now then plus minus divided multiplied'.split())
STATES={'morning','afternoon','evening','yesterday','today','tomorrow','initially','finally'}
def plain(s):
    s=re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}',r' \1 ',s)
    return s.replace('**','').replace('$',' dollars ').replace('\\,',' ')
def unit(s):
    m=re.match(r'\s*[-–]?\s*([A-Za-z]+)',s)
    if not m or m[1].lower() in STOP:return ''
    return singular(m[1].lower())
def context(s,names):
    ws=set(re.findall(r'[a-z]+',s.lower()));owners=ws&names;states=ws&STATES
    return dict(owner=next(iter(owners)) if len(owners)==1 else '',state=next(iter(states)) if len(states)==1 else '')
def source_context(q,qs,i,names):
    n=qs[i];start=max(q.rfind('.',0,n['start']),q.rfind(';',0,n['start']),q.rfind(',',0,n['start']))+1
    result=context(q[start:n['end']],names);result['unit']='dollar' if q[max(0,n['start']-1):n['start']]=='$' else unit(q[n['end']:]);return result
def hint(line,v,names):
    s=plain(line);us=set()
    for m in re.finditer(r'(?<![\w.])\d+(?:\.\d+)?(?:/\d+)?(?!\w)',s):
        try:
            if F(m[0])==v:
                u=unit(s[m.end():])
                if u:us.add(u)
        except (ValueError,ZeroDivisionError):pass
    result=context(s,names);result['unit']=next(iter(us)) if len(us)==1 else '';return result
def names_in(q):
    excluded=set('A An The If How What When Once After Before There Each Every In On Half One Two Three Four Five Six Seven Eight Nine Ten'.lower().split())
    return {n.lower() for n in re.findall(r'\b[A-Z][a-z]+\b',q)}-excluded
def choose(q,qs,ps,v,line,metadata):
    names=names_in(q);wanted=hint(line,v,names)
    records=[source_context(q,qs,p[1],names) if p[0]=='N' else metadata.get(repr(p),{}) for p in ps]
    def conflicts(r):return any(wanted.get(k) and r.get(k) and wanted[k]!=r[k] for k in ('unit','owner','state'))
    def matches(r):return any(wanted.get(k) and wanted[k]==r.get(k) for k in ('unit','owner','state')) and not conflicts(r)
    if not any(matches(r) for r in records):return ps
    return [p for p,r in zip(ps,records) if not conflicts(r)]
def checks():
    from round29_programs import quantities
    q='Ann has 5 apples and Bob has 5 pears.';qs=quantities(q);p=[('N',0),('N',1)];assert choose(q,qs,p,F(5),'5 apples * 2 = 10 apples',{})==[p[0]];assert choose(q,qs,p,F(5),'5 * 2 = 10',{})==p
    q='In the morning Ann has 5 apples. In the afternoon Ann has 5 apples.';qs=quantities(q);assert choose(q,qs,p,F(5),'In the afternoon 5 apples + 2 = 7 apples',{})==[p[1]]
    assert choose(q,qs,p,F(5),'5 apples + 2 = 7 apples',{})==p
    return dict(explicit_object=True,explicit_time=True,unknown_preserved=True)
