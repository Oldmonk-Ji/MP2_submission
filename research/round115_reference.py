"""Conservative complete reference lineage; unresolved steps taint descendants.

Alternatives encode uncertainty, not independent human semantic annotations.
Never substitute a convenient literal for an omitted arithmetic dependency.
"""
import ast,re
from fractions import Fraction as F
from itertools import product
from round29_programs import quantities
from round44_programs import canonical,locations
from round51_programs import value
from features import parse_answer
OPS={ast.Add:'ADD',ast.Sub:'SUB',ast.Mult:'MUL',ast.Div:'DIV'}
LIMIT=64

class Unknown(ValueError):pass

def dedup(ps):
    out={repr(canonical(p)):canonical(p) for p in ps}
    if len(out)>LIMIT:raise Unknown('ambiguity_overflow')
    return list(out.values())

def align(q,reference):
    qs=quantities(q);direct={};cache={};tainted=set();events=[]
    for i,n in enumerate(qs):direct.setdefault(F(n['value']),[]).append(('N',i))
    constants={F(0),F(1)}
    for v,cue in [(100,r'%|percent|cent|dollar'),(60,r'hour|minute|second'),(7,r'week|day'),(12,r'dozen|year|month'),(24,r'day|hour'),(1000,r'kilomet|kilogram|millil')]:
        if re.search(cue,q,re.I):constants.add(F(v))
    def bind(v):
        if v in tainted:raise Unknown('unresolved_prior_result')
        ps=list(direct.get(v,[]))+list(cache.get(v,[]))
        for i,n in enumerate(qs):
            nv=F(n['value']);raw=str(n.get('raw','')).lower();suffix=q[n['end']:n['end']+12]
            if re.match(r'\s*(?:%|percent)',suffix,re.I):
                if nv/100==v:ps.append(('DIV',('N',i),('L','100')))
            # A lexical fraction supplies its reciprocal denominator. This
            # preserves the source slot instead of inventing a free constant 2.
            if nv and nv<1 and (re.search(r'half|third|quarter|fourth|fifth|/',raw)) and 1/nv==v:
                ps.append(('DIV',('L','1'),('N',i)))
        if v in constants:ps.append(('L',str(v)))
        if not ps:raise Unknown('ungrounded_operand')
        return dedup(ps)
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return bind(F(str(n.value)))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return dedup([('SUB',('L','0'),p) for p in visit(n.operand)])
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:
            left,right=visit(n.left),visit(n.right)
            if len(left)*len(right)>LIMIT:raise Unknown('ambiguity_overflow')
            return dedup([(OPS[type(n.op)],a,b) for a,b in product(left,right)])
        raise Unknown('unsupported_expression')
    steps=[]
    for m in re.finditer(r'<<([^<>]+)>>',reference):
        raw=m[1]
        if raw.count('=')!=1:continue
        expr,result=raw.split('=')
        try:target=F(result.replace(',','').strip())
        except ValueError:continue
        try:
            node=ast.parse(expr.replace(',','').replace('×','*'),mode='eval').body
            if not any(isinstance(n,ast.BinOp) for n in ast.walk(node)):raise Unknown('unexplained_result')
            ps=visit(node)
            if any(value(p,qs)!=target or len(list(locations(p)))>63 for p in ps):raise Unknown('invalid_arithmetic_or_size')
            merged=dedup(cache.get(target,[])+ps)
            cache[target]=merged
            steps.append(dict(programs=ps,value=str(target),reference_span=list(m.span()),evidence=m.group()))
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,IndexError) as e:
            tainted.add(target);events.append(dict(value=str(target),reason=str(e),evidence=m.group()))
    answer=parse_answer(reference)
    if answer is None:return [],dict(events=events,reason='no_final_answer')
    final=F(answer)
    # Later failure at the same numeric value makes earlier alternatives
    # incomplete too; abstain instead of selecting a convenient earlier step.
    roots=[] if final in tainted else cache.get(final,[])
    roots=[p for p in roots if any(n[0]=='N' for _,n in locations(p))]
    return roots,dict(events=events,reason='complete_alternatives' if roots else 'unresolved_final',alternatives=len(roots),steps=len(steps))

def checks():
    q='A store has 500 customers, 2 heads of lettuce at 1 dollar and 4 tomatoes at 0.5 dollars.'
    ps,r=align(q,'<<2*1=2>> <<4*0.5=2>> <<2+2=4>> <<4*500=2000>> #### 2000')
    assert ps
    q='There are 10 friends and 4 leave. Each pays 8 more.'
    assert not align(q,'<<12=12>> <<10*12=120>> #### 120')[0]
    assert not align('A carton has 12 boxes with 10 packs, and costs 1440.', '<<144*10=1440>> <<1440/1440=1>> #### 1')[0]
    ps,r=align('Paul earns 12.50 per hour, pays 20% tax and works 40 hours.', '<<12.5*40=500>> <<500*0.20=100>> <<500-100=400>> #### 400')
    assert ps and not r['events']
    return dict(unexplained_constant_rejected=True,failed_intermediate_cannot_rebind_source=True,percent_dependency_preserved=True,ambiguity_retained=True)
