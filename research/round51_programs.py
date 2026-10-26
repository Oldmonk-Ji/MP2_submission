"""Bounded source arithmetic DAG and single-variable linear equation execution.

Rules designed on fit folds only. Candidate equations are hypotheses, not proofs.
No final-answer insertion, reference access or question-ID features.
"""
import ast,re,hashlib,math
from fractions import Fraction as F
from collections import defaultdict
from itertools import product
from round49_programs import proposals as previous,arithmetic_evidence,UNIT_FRACTION
from round46_graph import graph,LIMIT,OPS
from round44_programs import tree,canonical,locations,CONSTANTS
from round4_quantities import math_text
from features import parse_answer

ZERO=('L','0');ONE=('L','1')

def value(p,qs):
    if p[0]=='N':return F(qs[p[1]]['value'])
    if p[0]=='L':return F(p[1])
    a,b=value(p[1],qs),value(p[2],qs)
    if p[0]=='ADD':v=a+b
    elif p[0]=='SUB':v=a-b
    elif p[0]=='MUL':v=a*b
    elif p[0]=='DIV':v=a/b
    elif p[0]=='FLOOR':v=F(math.floor(a/b))
    elif p[0]=='CEIL':v=F(math.ceil(a/b))
    else:raise ValueError('unknown operation')
    if abs(v)>10**15:raise ValueError('unbounded')
    return v

def simplify(op,a,b):
    if op=='ADD' and a==ZERO:return b
    if op in ('ADD','SUB') and b==ZERO:return a
    if op=='SUB' and a==b:return ZERO
    if op=='MUL' and (a==ZERO or b==ZERO):return ZERO
    if op in ('MUL','DIV') and b==ONE:return a
    if op=='MUL' and a==ONE:return b
    return canonical((op,a,b))

def constants(q):
    cs=set(CONSTANTS)
    if re.search(r'\byears?\b',q,re.I):cs|={F(52),F(365)}
    if re.search(r'\b(quarters?|dimes?|nickels?|pennies|cents?)\b',q,re.I):cs|={F(25),F('1/20'),F('1/10')}
    return cs

def valid(p,qs,cs):
    ns=list(locations(p))
    if len(ns)>63 or not any(v[0]=='N' for _,v in ns):return None
    if any(v[0]=='L' and F(v[1]) not in cs for _,v in ns):return None
    try:
        v=value(p,qs)
        return str(v) if 0<=v<=10**12 else None
    except (ValueError,ZeroDivisionError,OverflowError,IndexError):return None

def unique(ps,limit=4):
    d={repr(canonical(p)):canonical(p) for p in ps if len(list(locations(p)))<=63}
    return [d[k] for k in sorted(d,key=lambda k:(len(k),k))[:limit]]

def normalized(text):
    s=math_text(UNIT_FRACTION.sub(' ',text))
    s=re.sub(r'\bhalf\s+of\s+', '(1/2) * ',s,flags=re.I)
    s=re.sub(r'(?<=[\d)])\s+of\s+(?=\d)', ' * ',s,flags=re.I)
    s=re.sub(r'\\(?:Rightarrow|rightarrow|implies|approx)', ';',s)
    s=s.replace('→',';').replace('≈','=').replace('^','**')
    # Only unit/annotation spans between numeric terms and arithmetic syntax.
    s=re.sub(r'(?<=[\d)])\s+[A-Za-z][A-Za-z /-]{1,45}?(?=\s*[+*/=])',' ',s)
    s=re.sub(r'(?<=\d)\s*\([A-Za-z][^()\d]*\)','',s)
    s=re.sub(r'(?<=\d)([a-zA-Z])\b',r'*\1',s)
    s=re.sub(r'/\s*(?:years?|weeks?|days?|hours?|minutes?|months?)\b',' ',s,flags=re.I)
    # An explicit inequality contributes its numeric boundary, not its truth.
    s=re.sub(r'([\d().+*/ \t\-]+)\s*[<>](?:=)?[^\n]*',r'\1 = 0',s)
    return s

def equations(text):
    for line in normalized(text).splitlines():
        line=re.sub(r'^\s*[-*+]\s+','',line)
        for segment in re.split(r'[;:]',line):
            for m in re.finditer('=',segment):
                a=re.search(r'([\dA-Za-z().+*/ \t\-]+)$',segment[:m.start()]);b=re.match(r'([\dA-Za-z().+*/ \t\-]+)',segment[m.end():])
                if not a or not b:continue
                left=a[0].strip();right=b[0].strip().rstrip('.')
                # Strip prose boundaries while preserving standalone variables.
                left=re.sub(r'^.*\b[A-Za-z]{2,}\b\s*','',left)
                right=re.split(r'\s+[A-Za-z]{2,}\b',right,maxsplit=1)[0].strip().rstrip('.')
                try:
                    aa=ast.parse(left,mode='eval').body;bb=ast.parse(right,mode='eval').body
                    nodes=list(ast.walk(aa))+list(ast.walk(bb))
                    if len(nodes)>100:continue
                    names={n.id for n in nodes if isinstance(n,ast.Name)}
                    if len(names)>1 or any(len(x)!=1 for x in names):continue
                    if not any(isinstance(n,ast.BinOp) for n in nodes):continue
                    yield aa,bb,names
                except (SyntaxError,ValueError):continue

def candidate_roots(q,text,g):
    qs=g['quantities'];cs=constants(q);cache=defaultdict(list);direct=defaultdict(list);roots=[]
    for i,x in enumerate(qs):direct[F(x['value'])].append(('N',i))
    def bind(v):
        ps=list(direct[v])+list(cache[v])
        for i,x in enumerate(qs):
            n=F(x['value']);kind=g['nodes'][i]['kind'];p=('N',i)
            if kind=='percent':
                f=('DIV',p,('L','100'))
                if n/100==v:ps.append(f)
                if 1-n/100==v:ps.append(('SUB',ONE,f))
                if 1+n/100==v:ps.append(('ADD',ONE,f))
            if kind=='fraction_denominator' and n and 1/n==v:ps.append(('DIV',ONE,p))
            for factor,cue in ((60,r'hour|minute'),(7,r'week|day'),(12,r'dozen|year|month'),(100,r'cent|dollar'),(52,r'year|week'),(365,r'year|day')):
                if F(factor) not in cs or not re.search(cue,q,re.I):continue
                if n*factor==v:ps.append(('MUL',p,('L',str(factor))))
                if n/factor==v:ps.append(('DIV',p,('L',str(factor))))
        if v in cs:ps.append(('L',str(v)))
        if not ps:
            # Fill one omitted arithmetic intermediate using source operands.
            # These remain competing hypotheses and are never process gold.
            terms=[('N',i) for i in range(len(qs))]+[('L',str(k)) for k in (1,2,7,60) if F(k) in cs]
            for a,b in product(terms,terms):
                if a[0]==b[0]=='L':continue
                av,bv=value(a,qs),value(b,qs)
                for op,res in (('ADD',av+bv),('SUB',av-bv),('MUL',av*bv),('DIV',av/bv if bv else None)):
                    if res==v:ps.append(simplify(op,a,b))
        return unique(ps)
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return bind(F(str(n.value)))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return [simplify('SUB',ZERO,p) for p in visit(n.operand)]
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:return unique([simplify(OPS[type(n.op)],a,b) for a,b in product(visit(n.left),visit(n.right))])
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Pow) and isinstance(n.right,ast.Constant) and type(n.right.value)==int and 1<=n.right.value<=4:
            ps=[]
            for p in visit(n.left):
                out=p
                for _ in range(n.right.value-1):out=simplify('MUL',out,p)
                ps.append(out)
            return unique(ps)
        return []
    def linear(n):
        if isinstance(n,ast.Name):return [(ONE,ZERO)]
        if not any(isinstance(t,ast.Name) for t in ast.walk(n)):return [(ZERO,p) for p in visit(n)]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return [(simplify('SUB',ZERO,a),simplify('SUB',ZERO,b)) for a,b in linear(n.operand)]
        if not isinstance(n,ast.BinOp) or type(n.op) not in OPS:return []
        result=[];op=OPS[type(n.op)]
        for (a,b),(c,d) in product(linear(n.left),linear(n.right)):
            if op in ('ADD','SUB'):result.append((simplify(op,a,c),simplify(op,b,d)))
            elif op=='MUL' and (a==ZERO or c==ZERO):result.append((simplify('ADD',simplify('MUL',a,d),simplify('MUL',c,b)),simplify('MUL',b,d)))
            elif op=='DIV' and c==ZERO:result.append((simplify('DIV',a,d),simplify('DIV',b,d)))
        return result[:8]
    answer=parse_answer(text)
    if answer is None:return []
    # Process equalities in source order; numeric intermediates must execute.
    for a,b,names in list(equations(text))[:64]:
        try:
            if names:
                ps=[]
                for (aa,ab),(ba,bb) in product(linear(a),linear(b)):
                    ps.append(simplify('DIV',simplify('SUB',bb,ab),simplify('SUB',aa,ba)))
                kind='linear_equation'
            else:ps=visit(a)+visit(b);kind='typed_arithmetic_dag'
            for p in unique(ps,8):
                v=valid(p,qs,cs)
                if v is None:continue
                # A side expression is executable evidence, even if the stated equality is false.
                cache[F(v)]=unique(cache[F(v)]+[p])
                if v==answer:roots.append((p,kind))
                if re.search(r'round|whole|minimum|at least|strictly|full|complete',q+' '+text,re.I):
                    for op in ('FLOOR','CEIL'):
                        pp=(op,p,ONE)
                        if valid(pp,qs,cs)==answer:roots.append((pp,'integer_boundary'))
                    pp=('ADD',('FLOOR',p,ONE),ONE)
                    if re.search(r'strictly|more than',q,re.I) and valid(pp,qs,cs)==answer:roots.append((pp,'strict_integer_boundary'))
        except (ValueError,ZeroDivisionError,OverflowError,RecursionError):continue
    return roots

def proposals(row):
    old=previous(row);q=row['question'];g=old['graph'];cs=constants(q);answers=[parse_answer(c['text']) for c in row['candidates']]
    chosen=list(old['hypotheses']);seen={repr(canonical(tree(h['program']))) for h in chosen};extra={}
    for c in row['candidates']:
        for p,kind in candidate_roots(q,c['text'],g):
            p=canonical(p);key=repr(p);v=valid(p,g['quantities'],cs)
            if key in seen or v is None:continue
            extra[key]=dict(program=p,action=kind,decision={},answer=v,matches=[j for j,a in enumerate(answers) if a==v])
    for key in sorted(extra,key=lambda k:hashlib.sha256(k.encode()).digest()):
        if len(chosen)>=LIMIT:break
        chosen.append(extra[key]);seen.add(key)
    return dict(old,hypotheses=chosen)

def checks():
    q='Lucy gives Linda 5 dollars and originally has 20 dollars. How much did Linda have?'
    g=graph(q);r=candidate_roots(q,'20 - 5 = 15\n15 = x + 5\n#### 10',g)
    assert any(value(p,g['quantities'])==10 and k=='linear_equation' for p,k in r)
    q='A ticket costs 2 dollars, a pass 50 dollars. Minimum rides so the pass is strictly cheaper?'
    g=graph(q);r=candidate_roots(q,'50 / 2 = 25\n#### 26',g)
    assert any(value(p,g['quantities'])==26 for p,k in r)
    assert not list(equations('x + y = 12'))
    assert value(('CEIL',('L','5'),('L','2')),[])==3
    assert value(('FLOOR',('L','5'),('L','2')),[])==2
    return dict(linear=True,strict_integer=True,exact_rounding=True,multiple_variables_rejected=True)
