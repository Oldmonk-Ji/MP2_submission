"""Strict optional numeric trace evidence; never infer correctness from absence.

Only explicit, numerically correct equalities create intermediates. A claimed
number alone is not a root. Unsupported dependencies and overflow remain unknown.
"""
import ast,re
from fractions import Fraction as F
from itertools import product
from round29_programs import quantities
from round44_programs import canonical,tree,locations
from round51_programs import equations,value
from round152_markup import clean
from round175_roles import kind
from features import parse_answer
from round192_context import choose,hint,names_in
LIMIT=64
OPS={ast.Add:'ADD',ast.Sub:'SUB',ast.Mult:'MUL',ast.Div:'DIV'}
class Unknown(ValueError):pass

def numeric(n):
    if isinstance(n,ast.Constant) and type(n.value) in (int,float):return F(str(n.value))
    if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):return numeric(n.operand)*(1 if isinstance(n.op,ast.UAdd) else -1)
    if isinstance(n,ast.BinOp):
        a,b=numeric(n.left),numeric(n.right)
        if isinstance(n.op,ast.Add):out=a+b
        elif isinstance(n.op,ast.Sub):out=a-b
        elif isinstance(n.op,ast.Mult):out=a*b
        elif isinstance(n.op,ast.Div):out=a/b
        elif isinstance(n.op,ast.Pow) and b.denominator==1 and 1<=b<=4:out=a**int(b)
        else:raise Unknown('unsupported_numeric_operator')
        if abs(out)>10**15:raise Unknown('numeric_bound')
        return out
    raise Unknown('not_numeric')

def unique(ps):
    result={}
    for p in ps:
        p=canonical(tree(p))
        if len(list(locations(p)))>63:raise Unknown('program_size')
        result[repr(p)]=p
        if len(result)>LIMIT:raise Unknown('ambiguity_overflow')
    return list(result.values())

def extract(q,text):
    answer=parse_answer(text)
    if answer is None:return [],dict(reason='no_parsed_final')
    qs=quantities(q);kinds=[kind(q,i) for i in range(len(qs))];direct={};cache={};tainted=set();events=[];accepted=0;metadata={};line_context=""
    for i,n in enumerate(qs):
        if kinds[i]!='ordinal':direct.setdefault(F(n['value']),[]).append(('N',i))
    constants={F(0),F(1)}
    for v,cue in [(100,r'%|percent|cent|dollar'),(60,r'\bhours?\b|\bminutes?\b|\bseconds\b'),(7,r'\bweeks?\b|\bdays?\b'),(12,r'\bdozen\b|\byears?\b|\bmonths?\b'),(24,r'\bdays?\b|\bhours?\b'),(1000,r'kilomet|kilogram|millil')]:
        if re.search(cue,q,re.I):constants.add(F(v))
    def bind(v):
        if v in tainted:raise Unknown('unresolved_prior_result')
        ps=list(direct.get(v,[]))+list(cache.get(v,[]))
        for i,n in enumerate(qs):
            nv=F(n['value']);p=('N',i);fraction=None
            if kinds[i]=='percent':fraction=('DIV',p,('L','100'));fv=nv/100
            elif kinds[i]=='fraction_denominator' and nv:fraction=('DIV',('L','1'),p);fv=1/nv
            elif kinds[i]=='fraction' and 0<nv<1:fraction=p;fv=nv
            if fraction is not None:
                if fv==v:ps.append(fraction)
                if 1-fv==v:ps.append(('SUB',('L','1'),fraction))
                if 1+fv==v:ps.append(('ADD',('L','1'),fraction))
                if fv and 1/fv==v:ps.append(('DIV',('L','1'),fraction))
        if v in constants:ps.append(('L',str(v)))
        if not ps:raise Unknown('ungrounded_operand')
        return unique(choose(q,qs,ps,v,line_context,metadata))
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return bind(F(str(n.value)))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            ps=visit(n.operand);return unique([('SUB',('L','0'),p) for p in ps]) if isinstance(n.op,ast.USub) else ps
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:
            a,b=visit(n.left),visit(n.right)
            if len(a)*len(b)>LIMIT:raise Unknown('ambiguity_overflow')
            return unique([(OPS[type(n.op)],x,y) for x,y in product(a,b)])
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Pow) and isinstance(n.right,ast.Constant) and type(n.right.value)==int and 1<=n.right.value<=4:
            out=[]
            for p in visit(n.left):
                t=p
                for _ in range(n.right.value-1):t=('MUL',t,p)
                out.append(t)
            return unique(out)
        raise Unknown('unsupported_expression')
    steps=[(line,eq) for line in clean(text).splitlines() for eq in equations(line)]
    for line_context,(aa,bb,names) in steps[:64]:
        if names:events.append('symbolic_equation_unknown');continue
        target=None
        try:
            left,right=numeric(aa),numeric(bb);target=right
            if left!=right:raise Unknown('incorrect_numeric_equality')
            expr=aa if any(isinstance(n,ast.BinOp) for n in ast.walk(aa)) else bb
            ps=visit(expr)
            if any(value(p,qs)!=target for p in ps):raise Unknown('lineage_value_mismatch')
            merged=unique(cache.get(target,[])+ps);cache[target]=merged;accepted+=1
            result_context=hint(line_context,target,names_in(q))
            for p in ps:
                if p[0] not in ("N","L"):metadata[repr(p)]=result_context
        except (ValueError,ZeroDivisionError,OverflowError,IndexError) as e:
            if target is not None:tainted.add(target)
            events.append(str(e))
    final=F(answer);roots=[] if final in tainted else cache.get(final,[]);roots=[p for p in roots if any(n[0]=='N' for _,n in locations(p))]
    assert all(value(p,qs)==final for p in roots)
    return roots,dict(reason='verified_trace' if roots else 'unknown_trace',accepted_steps=accepted,events=events)

def checks():
    q='Ann has 4 boxes with 3 pencils in each box and loses 2 pencils.';p,r=extract(q,'4 * 3 = 12\n12 - 2 = 10\n#### 10');assert p and all(value(t,quantities(q))==10 for t in p)
    assert not extract('Ann has 10 pencils.','7 + 3 = 10\n#### 10')[0]
    assert not extract('Ann has 4 boxes of 3 pencils, 11 shelves, and loses 2 pencils.','4 * 3 = 11\n11 - 2 = 9\n#### 9')[0]
    q='One fourth of 80 engines are defective.';p,r=extract(q,'**80 * 0.75 = 60 engines**\n#### 60');assert p and all(value(t,quantities(q))==60 for t in p)
    assert not extract('The second box has 5 pencils.','2 * 5 = 10\n#### 10')[0]
    assert extract('Ann has 4 boxes with 3 pencils each.','12 = 4 * 3\n#### 12')[0]
    return dict(complete_intermediates=True,claimed_number_not_root=True,failed_intermediate_not_rebound=True,fraction_and_markup=True,ordinal_binding_rejected=True,reversed_equality=True)
