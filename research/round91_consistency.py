"""Exact affine constraints in candidate traces; no claims of semantic validity.

Only single-letter, single-variable expressions are accepted. Different values
for one symbol are soft conflict evidence, since a writer may reuse a symbol.
"""
import ast,re
from fractions import Fraction as F
from collections import defaultdict
import numpy as np
from round51_programs import normalized

def affine(n):
    if isinstance(n,ast.Constant) and type(n.value) in (int,float):return F(0),F(str(n.value))
    if isinstance(n,ast.Name) and len(n.id)==1:return F(1),F(0)
    if isinstance(n,ast.UnaryOp):
        a,b=affine(n.operand)
        if isinstance(n.op,ast.USub):return -a,-b
        if isinstance(n.op,ast.UAdd):return a,b
    if isinstance(n,ast.BinOp):
        a,b=affine(n.left);c,d=affine(n.right)
        if isinstance(n.op,ast.Add):return a+c,b+d
        if isinstance(n.op,ast.Sub):return a-c,b-d
        if isinstance(n.op,ast.Mult) and not (a and c):return a*d+c*b,b*d
        if isinstance(n.op,ast.Div) and not c and d:return a/d,b/d
    raise ValueError('Not an affine expression')

def constraints(text):
    # Reuse fixed arithmetic normalization, but accept x = 19 as a binding too.
    result=[]
    for line in normalized(text).splitlines():
        line=re.sub(r'^\s*[-*+]\s+','',line)
        for segment in re.split(r'[;:]',line):
            for m in re.finditer('=',segment):
                a=re.search(r'([\dA-Za-z().+*/ \t\-]+)$',segment[:m.start()]);b=re.match(r'([\dA-Za-z().+*/ \t\-]+)',segment[m.end():])
                if not a or not b:continue
                left=re.sub(r'^.*\b[A-Za-z]{2,}\b\s*','',a[0].strip())
                right=re.split(r'\s+[A-Za-z]{2,}\b',b[0].strip(),maxsplit=1)[0].strip().rstrip('.')
                try:
                    aa=ast.parse(left,mode='eval').body;bb=ast.parse(right,mode='eval').body;nodes=list(ast.walk(aa))+list(ast.walk(bb))
                    names={n.id for n in nodes if isinstance(n,ast.Name)}
                    if len(nodes)>80 or len(names)!=1 or any(len(s)!=1 for s in names):continue
                    ac,av=affine(aa);bc,bv=affine(bb);coef=ac-bc;constant=bv-av
                    if max(abs(coef),abs(constant))>10**12:continue
                    result.append(dict(variable=next(iter(names)),coefficient=str(coef),constant=str(constant),root=str(constant/coef) if coef else None,identity=not coef and not constant,impossible=not coef and bool(constant),left=left,right=right))
                except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):continue
    return result[:64]

def features(text):
    rows=constraints(text);groups=defaultdict(list);impossible=sum(r['impossible'] for r in rows);identity=sum(r['identity'] for r in rows)
    for r in rows:
        if r['root'] is not None:groups[r['variable']].append(F(r['root']))
    pairs=bad=adj=badadj=0;res=[];conflicting_groups=0;worst=0
    for values in groups.values():
        disagree=False
        for i,a in enumerate(values):
            for b in values[:i]:
                rel=float(abs(a-b)/max(F(1),abs(a),abs(b)));wrong=rel>.001 and abs(a-b)>F('0.02')
                pairs+=1;bad+=wrong;res.append(rel);disagree|=wrong
            if i:
                b=values[i-1];adj+=1;badadj+=float(abs(a-b)/max(F(1),abs(a),abs(b)))>.001 and abs(a-b)>F('0.02')
        conflicting_groups+=disagree;worst=max(worst,len(set(values)))
    return np.asarray([bool(rows),np.log1p(len(rows)),np.log1p(len(groups)),np.log1p(pairs),bad/max(1,pairs),float(bad>0),badadj/max(1,adj),np.log1p(impossible),np.log1p(identity),max(res,default=0),float(np.mean(res)) if res else 0,conflicting_groups/max(1,len(groups)),np.log1p(worst),np.log1p(sum(len(v) for v in groups.values()))],np.float32)

def build(bank):return np.asarray([[features(c['text']) for c in r['candidates']] for r in bank],np.float32)

def checks():
    assert affine(ast.parse('x+3*x+15*x',mode='eval').body)==(19,0)
    assert features('x+3x+15x=380\n20x=380\nx=19')[5]==1
    assert features('x+3x+15x=380\n19x=380\nx=20')[5]==0
    assert features('x+2=x+3')[7]>0
    assert not constraints('x+y=12')
    assert not constraints('x*x=12')
    assert not constraints('12 apples = 3 boxes')
    assert features('2*x=1\nx=0.5')[5]==0
    return dict(exact_coefficients=True,contradiction=True,consistent_trace=True,impossible_identity=True,multivariable_rejected=True,nonlinear_rejected=True,units_not_variables=True)
