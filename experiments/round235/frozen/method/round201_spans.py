"""Unique monotone operand occurrence alignment; ambiguous alignment is unknown."""
import ast,re
from fractions import Fraction as F
from round192_context import unit,context,names_in,source_context,choose as line_choose,hint
NUM=re.compile(r'(?<![\w.])\d+(?:\.\d+)?(?!\w)')
def readable(line):
    s=re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}',r' \1 ',line)
    s=re.sub(r'(?<=\d),(?=\d{3}\b)','',s)
    for a,b in [('\\times','*'),('\\cdot','*'),('\\div','/'),('×','*'),('÷','/'),('**',''),('$',''),('\\,',' ')]:s=s.replace(a,b)
    return s
def align(line,aa,bb,q):
    s=readable(line);tokens=list(NUM.finditer(s));values=[F(m[0]) for m in tokens]
    nodes=[]
    for expr in (aa,bb):nodes.extend(sorted([n for n in ast.walk(expr) if isinstance(n,ast.Constant) and type(n.value) in (int,float)],key=lambda n:n.col_offset))
    wanted=[F(str(n.value)) for n in nodes];starts=[i for i in range(len(values)-len(wanted)+1) if values[i:i+len(wanted)]==wanted]
    if len(starts)!=1 or not wanted:return {}
    result={};names=names_in(q)
    for node,m in zip(nodes,tokens[starts[0]:starts[0]+len(nodes)]):
        lo=max([s.rfind(c,0,m.start()) for c in '+*/=;'])+1
        ends=[x for c in '+*/=;' if (x:=s.find(c,m.end()))>=0];hi=min(ends) if ends else len(s)
        local=s[lo:hi];r=context(local,names);tail=s[m.end():hi].lstrip(' }\\')
        r['unit']=unit(tail);r['span']=[m.start(),m.end()];result[id(node)]=r
    return result
def choose(q,qs,ps,v,line,metadata,wanted=None):
    if wanted is None:return line_choose(q,qs,ps,v,line,metadata)
    names=names_in(q);records=[source_context(q,qs,p[1],names) if p[0]=='N' else metadata.get(repr(p),{}) for p in ps]
    def conflict(r):return any(wanted.get(k) and r.get(k) and wanted[k]!=r[k] for k in ('unit','owner','state'))
    def match(r):return any(wanted.get(k) and wanted[k]==r.get(k) for k in ('unit','owner','state')) and not conflict(r)
    return [p for p,r in zip(ps,records) if not conflict(r)] if any(match(r) for r in records) else ps
def checks():
    from round29_programs import quantities
    q='Ann has 20 balls and 20 minutes.';a=ast.parse('20 * 20',mode='eval').body;b=ast.parse('400',mode='eval').body;d=align('20 balls * 20 minutes = 400',a,b,q);assert d[id(a.left)]['unit']=='ball' and d[id(a.right)]['unit']=='minute'
    ps=[('N',0),('N',1)];qs=quantities(q);assert choose(q,qs,ps,F(20),'',{},d[id(a.left)])==[ps[0]];assert choose(q,qs,ps,F(20),'',{},d[id(a.right)])==[ps[1]]
    assert align('20 * 20 = 400; 20 * 20 = 400',a,b,q)=={}
    d=align('20 * 20 = 400',a,b,q);assert choose(q,qs,ps,F(20),'',{},d[id(a.left)])==ps
    q='In the morning Ann has 5 apples. In the afternoon Ann has 5 apples.';a=ast.parse('5 + 5',mode='eval').body;b=ast.parse('10',mode='eval').body;d=align('morning 5 apples + afternoon 5 apples = 10 apples',a,b,q);assert d[id(a.left)]['state']=='morning' and d[id(a.right)]['state']=='afternoon'
    return dict(distinct_operand_units=True,distinct_operand_states=True,ambiguous_alignment_unknown=True,missing_context_keeps_competitors=True)
