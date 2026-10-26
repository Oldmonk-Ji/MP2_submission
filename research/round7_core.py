"""Input-only features for learned quantity roles and relation linking."""
import re
from fractions import Fraction
from round4_quantities import WORDS

W={k:v for k,v in WORDS.items() if k not in ('first','second','third','fourth','fifth')}
RX=re.compile(r'(?<![\w.])(?:\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?|\.\d+)(?!\w)|\b(?:'+'|'.join(W)+r')\b',re.I)
EXCLUDE=set('a an the if in on at to how what when there while due every each three two four five ten twenty hundred mr mrs their his her he she they it this that last next after before'.split())|set(W)
ROLES=('father','mother','sister','brother','dad','mom','puppy','wife','husband','friend')
def quantities(q):
    out=[]
    for m in RX.finditer(q):
        raw=m.group().lower()
        try:v=float(W[raw]) if raw in W else float(Fraction(raw.replace(',','')))
        except (ValueError,ZeroDivisionError):continue
        out.append(dict(raw=m.group(),value=v,start=m.start(),end=m.end()))
    return out
def entities(q):
    found={}
    for m in re.finditer(r"\b[A-Z][a-z]+\b",q):
        name=m.group().lower()
        if name not in EXCLUDE:found.setdefault(name,[]).append(m.start())
    names=list(found)
    # Candidate generation only: competing possessive owners remain candidates.
    for role in ROLES:
        hits=[m.start() for m in re.finditer(r'\b'+role+r'\b',q,re.I)]
        if hits:
            for n in names:found.setdefault(n+"'s "+role,[]).extend(hits)
    return dict(NIL=[-1],**found)
def lex(s):return re.findall(r'[a-z]+|[%$]',s.lower())
def qfeatures(q,qs,i,context=True):
    x=qs[i];left=q[:x['start']];right=q[x['end']:];e=entities(q);names={n for n in e if "'s" not in n}
    def masked(ws):return ['ENTITY' if w in names else w for w in ws]
    l=masked(lex(left)[-7:]);r=masked(lex(right)[:7]);f={'bias':1.,'index':str(min(i,5)),'n':str(min(len(qs),8)),'dollar_before':left.rstrip().endswith('$'),'dollar_after':right.lstrip().startswith('$'),'percent':right.lstrip().startswith('%'),'fraction':'/' in x['raw'] or x['raw'].lower() in ('half','quarter'),'written':x['raw'].isalpha()}
    for k,w in enumerate(reversed(l)):f['l'+str(k)]=w
    for k,w in enumerate(r):f['r'+str(k)]=w
    if context:
        for w in set(l+r):f['ctx:'+w]=True
        f['l2']=' '.join(l[-2:]);f['r2']=' '.join(r[:2]);f['question_tail']=' '.join(masked(lex(q)[-5:]))
    return f
def owner_features(q,qs,i,entity,context=True):
    x=qs[i];es=entities(q);positions=es[entity];nonnegative=[p for p in positions if p>=0];nearest=min(nonnegative,key=lambda p:abs(p-x['start'])) if nonnegative else -1
    f=qfeatures(q,qs,i,context);f.update(nil=entity=='NIL',relative="'s" in entity)
    if nearest>=0:
        gap=x['start']-nearest;f.update(before=gap>=0,distance=min(abs(gap),300)/100.,near_bucket=str(min(abs(gap)//20,10)),same_sentence=not bool(re.search(r'[.!?]',q[min(nearest,x['start']):max(nearest,x['start'])])))
        preceding=[(max(p for p in ps if p<x['start']),n) for n,ps in es.items() if n!='NIL' and any(0<=p<x['start'] for p in ps)]
        f['nearest_preceding']=bool(preceding) and max(preceding)[1]==entity
        if context:
            between=q[min(nearest,x['start']):max(nearest,x['start'])]
            for w in set(lex(between))&{'he','she','his','her','they','their','than','as','is','was','will','buys','bought','cost','costs','each','per','and','but'}:f['between:'+w]=True
            f['role']=entity.split("'s ")[-1] if "'s " in entity else 'name'
    return f
def scope_features(q,qs,i,j,context=True):
    f=qfeatures(q,qs,i,context);f['nil']=j==-1
    if j>=0:
        a,b=qs[i],qs[j];f.update(offset=str(max(-5,min(5,i-j))),before=j<i,distance=min(abs(a['start']-b['start']),300)/100.,candidate_dollar=q[:b['start']].rstrip().endswith('$'))
        if context:
            for k,v in qfeatures(q,qs,j,True).items():f['candidate:'+k]=v
            for w in set(lex(q[min(a['end'],b['end']):max(a['start'],b['start'])])):f['between:'+w]=True
    return f
