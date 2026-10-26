"""Exact generic arithmetic templates, preserving source slots and scope."""
from fractions import Fraction as F

def fold(op,children):
    flat=[]
    for c in children:flat.extend(c[1:] if c[0]==op else (c,))
    literal=F(0 if op=='ADD' else 1);terms=[]
    for c in flat:
        if c[0]=='L':literal=literal+F(c[1]) if op=='ADD' else literal*F(c[1])
        else:terms.append(c)
    if op=='MUL' and not literal:return ('L','0')
    if literal!=F(0 if op=='ADD' else 1) or not terms:terms.append(('L',str(literal)))
    terms=sorted(terms,key=repr)
    return terms[0] if len(terms)==1 else (op,*terms)

def canonical(p):
    if p[0]=='N':return ('N',int(p[1]))
    if p[0]=='L':return ('L',str(F(p[1])))
    a,b=canonical(p[1]),canonical(p[2])
    if p[0] in ('ADD','MUL'):return fold(p[0],(a,b))
    if p[0]=='SUB':return fold('ADD',(a,fold('MUL',(('L','-1'),b))))
    if p[0]=='DIV':
        if b==('L','1'):return a
        if a[0]==b[0]=='L':return ('L',str(F(a[1])/F(b[1])))
        return ('DIV',a,b)
    raise ValueError('Unsupported template operation')

def execute(p,values):
    if p[0]=='N':return values[p[1]]
    if p[0]=='L':return F(p[1])
    if p[0]=='ADD':return sum((execute(c,values) for c in p[1:]),F(0))
    if p[0]=='MUL':
        v=F(1)
        for c in p[1:]:v*=execute(c,values)
        return v
    if p[0]=='DIV':return execute(p[1],values)/execute(p[2],values)
    raise ValueError('Unsupported template operation')
