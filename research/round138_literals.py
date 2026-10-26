"""Exact literal normalization and source/literal scope, including absence."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations,product
from round101_templates import fold,execute
LABELS=['ADD','MUL','DIV','DIV_REVERSE','ABSENT']
ABSENT=4

def normalize(p):
    if p[0]=='N':return ('N',int(p[1]))
    if p[0]=='L':return ('L',str(F(p[1])))
    a,b=normalize(p[1]),normalize(p[2])
    if p[0] in ('ADD','MUL'):return fold(p[0],(a,b))
    if p[0]=='SUB':return fold('ADD',(a,fold('MUL',(('L','-1'),b))))
    if p[0]=='DIV':
        if b[0]=='L':return fold('MUL',(a,('L',str(1/F(b[1])))))
        return ('DIV',a,b)
    raise ValueError('unsupported literal relation operator')

def relation_map(normal):
    evidence=defaultdict(set);constants=set()
    def cross(left,right,op):
        for a,b in product(left,right):
            if a[0]==b[0]:continue
            source,literal=(a,b) if a[0]=='N' else (b,a)
            label=0 if op=='ADD' else 1 if op=='MUL' else 2 if a[0]=='N' else 3
            evidence[(source[1],literal[1])].add(label)
    def walk(p):
        if p[0] in ('N','L'):
            if p[0]=='L':constants.add(p[1])
            return [p]
        groups=[walk(c) for c in p[1:]]
        for a,b in combinations(groups,2):cross(a,b,p[0])
        return [x for g in groups for x in g]
    walk(normal)
    # Multiple occurrences can imply conflicting paths. Keep those unknown.
    return {k:next(iter(v)) if len(v)==1 else None for k,v in evidence.items()},constants

def candidate_constants(packet):
    result=set()
    for roots in packet:
        for p in roots:
            try:_,cs=relation_map(normalize(p));result.update(cs)
            except (ValueError,ZeroDivisionError):pass
    return sorted(result,key=lambda s:(F(s),s))

def checks():
    a=('MUL',('N',0),('DIV',('N',1),('L','100')));b=('MUL',('N',0),('MUL',('N',1),('L','1/100')))
    assert normalize(a)==normalize(b)
    assert normalize(('ADD',('N',0),('L','0')))==('N',0)
    assert normalize(('MUL',('N',0),('L','1')))==('N',0)
    x=normalize(('SUB',('N',0),('N',1)));r,c=relation_map(x)
    assert r[(0,'-1')]==0 and r[(1,'-1')]==1
    assert execute(normalize(a),[F(20),F(30)])==F(6)
    return dict(percent_denominator_and_decimal_factor_equivalent=True,neutral_literals_removed=True,subtraction_scope_preserved=True,exact_fraction_execution=True)
