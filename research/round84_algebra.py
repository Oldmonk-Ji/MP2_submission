"""Exact bounded factoring for repeated-source programs; constants stay explicit.

Read-once trees retain the established monotone representation. Repeated-source
trees may be factored without interpreting prose or rebinding source quantities.
"""
from functools import lru_cache
from collections import Counter
from round44_programs import locations,tree
from round82_relations import normalize,relations as old_relations

def combine(op,xs):
    xs=sorted(xs,key=repr)
    if not xs:return ('L','0' if op=='ADD' else '1')
    p=xs[0]
    for x in xs[1:]:p=(op,p,x)
    return p

def expression(p,sp):
    op=p[0]
    if op=='N':return sp.Symbol('q'+str(p[1]))
    if op=='L':return sp.Rational(p[1])
    a,b=expression(p[1],sp),expression(p[2],sp)
    if op=='ADD':return a+b
    if op=='SUB':return a-b
    if op=='MUL':return a*b
    if op=='DIV':return a/b
    raise ValueError('unsupported operator')

def program(e,sp):
    if e.is_Symbol:return ('N',int(str(e)[1:]))
    if e.is_Rational:return ('L',str(e))
    if e.is_Add:
        pos=[];neg=[]
        for t in e.as_ordered_terms():
            if t.could_extract_minus_sign():neg.append(program(-t,sp))
            else:pos.append(program(t,sp))
        return ('SUB',combine('ADD',pos),combine('ADD',neg)) if neg else combine('ADD',pos)
    if e.could_extract_minus_sign():return ('SUB',('L','0'),program(-e,sp))
    if e.is_Mul:
        top=[];bottom=[]
        for t in e.as_ordered_factors():
            if t.is_Pow and t.exp.is_Integer and t.exp<0:bottom.append(program(t.base**(-t.exp),sp))
            else:top.append(program(t,sp))
        return ('DIV',combine('MUL',top),combine('MUL',bottom)) if bottom else combine('MUL',top)
    if e.is_Pow and e.exp.is_Integer and abs(e.exp)<=6:
        t=program(e.base,sp);power=combine('MUL',[t]*abs(int(e.exp)))
        return ('DIV',('L','1'),power) if e.exp<0 else power
    raise ValueError('unsupported symbolic result')

@lru_cache(maxsize=8192)
def _recover(p):
    original=old_relations(p)
    if original is not None:return normalize(p)
    nodes=list(locations(p));source=[v[1] for _,v in nodes if v[0]=='N']
    if len(nodes)>63 or len(set(source))>8 or any(v[0] not in ('N','L','ADD','SUB','MUL','DIV') for _,v in nodes):return None
    import sympy as sp
    try:
        raw=expression(p,sp);expanded=sp.expand(sp.cancel(raw));symbols=sorted(expanded.free_symbols,key=str)
        choices=[sp.factor_terms(expanded),sp.factor(expanded)]
        for order in (symbols,list(reversed(symbols))):
            choices.append(sp.collect(expanded,order))
            n,d=sp.fraction(sp.cancel(raw))
            try:choices.append(sp.horner(n,*order)/sp.horner(d,*order))
            except (sp.PolynomialError,ValueError):pass
        candidates=[]
        for e in choices:
            try:
                q=normalize(program(e,sp));ns=list(locations(q));leaves=[v[1] for _,v in ns if v[0]=='N']
                if len(leaves)!=len(set(leaves)) or len(ns)>63 or not leaves:continue
                # Exact symbolic identity, not equality at the original numbers.
                if sp.cancel(raw-expression(q,sp))!=0:continue
                candidates.append(q)
            except (ValueError,TypeError,ZeroDivisionError):continue
        return min(candidates,key=lambda q:(len(list(locations(q))),repr(q))) if candidates else None
    except (ValueError,TypeError,ZeroDivisionError,sp.PolynomialError):return None

def recover(p):return _recover(tree(p))

def relations(p):
    q=recover(p);return old_relations(q) if q is not None else None
