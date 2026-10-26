"""Bounded exact Laurent expansion for equivalent scalar and percentage forms.

No source amounts are substituted. A sum in a denominator or excess term growth
abstains from expansion and retains the previous exact tree normalization.
"""
from fractions import Fraction as F
from round101_templates import fold,execute
from round138_literals import normalize as previous,relation_map,ABSENT,LABELS
MAX_TERMS=32

def trim(p):
    p={m:c for m,c in p.items() if c}
    if len(p)>MAX_TERMS or any(sum(abs(e) for _,e in m)>16 for m in p):raise ValueError('bounded expansion exceeded')
    return p

def monomial(a,b,sign=1):
    d=dict(a)
    for i,e in b:d[i]=d.get(i,0)+sign*e
    return tuple(sorted((i,e) for i,e in d.items() if e))

def polynomial(p):
    op=p[0]
    if op=='N':return {((int(p[1]),1),):F(1)}
    if op=='L':return trim({():F(p[1])})
    a,b=polynomial(p[1]),polynomial(p[2]);out={}
    if op in ('ADD','SUB'):
        out=a.copy()
        for m,c in b.items():out[m]=out.get(m,F(0))+(c if op=='ADD' else -c)
    elif op=='MUL':
        for ma,ca in a.items():
            for mb,cb in b.items():
                m=monomial(ma,mb);out[m]=out.get(m,F(0))+ca*cb
    elif op=='DIV':
        if not b:raise ZeroDivisionError
        if len(b)!=1:raise ValueError('nonmonomial denominator')
        mb,cb=next(iter(b.items()))
        for ma,ca in a.items():out[monomial(ma,mb,-1)]=ca/cb
    else:raise ValueError('unsupported operator')
    return trim(out)

def from_polynomial(p):
    terms=[]
    for m,c in sorted(p.items()):
        factors=[('L',str(c))]
        for i,e in m:
            atom=('N',i) if e>0 else ('DIV',('L','1'),('N',i))
            factors.extend([atom]*abs(e))
        terms.append(fold('MUL',factors))
    return fold('ADD',terms)

def normalize(p):
    try:return from_polynomial(polynomial(p))
    except ValueError:return previous(p)

def candidate_constants(packet):
    result=set()
    for roots in packet:
        for p in roots:
            try:_,cs=relation_map(normalize(p));result.update(cs)
            except (ValueError,ZeroDivisionError):pass
    return sorted(result,key=lambda s:(F(s),s))

def checks():
    n0,n1,n2=[('N',i) for i in range(3)];h=('L','100')
    pairs=[(('DIV',n1,('DIV',n0,h)),('MUL',('DIV',h,n0),n1)),
           (('MUL',n0,('SUB',('L','1'),('DIV',n1,h))),('SUB',n0,('DIV',('MUL',n0,n1),h))),
           (('SUB',n0,('ADD',n1,n2)),('SUB',('SUB',n0,n1),n2)),
           (('DIV',n0,('DIV',('L','1'),n1)),('MUL',n0,n1))]
    for a,b in pairs:assert normalize(a)==normalize(b)
    assert normalize(('MUL',n0,('L','2')))!=normalize(('MUL',n1,('L','2')))
    q=('DIV',n0,('ADD',n1,n2));assert normalize(q)==previous(q)
    assert execute(normalize(('SUB',n0,n0)),[F(3)])==0
    return dict(nested_division=True,distributed_percentage=True,subtraction_chain=True,reciprocal=True,source_identity_preserved=True,nonmonomial_denominator_falls_back=True)
