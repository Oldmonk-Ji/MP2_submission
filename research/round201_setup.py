from round22_common import ROOT,dump,sha
from round201_spans import checks
if __name__=='__main__':
    r=ROOT/'research';out=ROOT/'experiments/round201';out.mkdir();s=(r/'round192_trace.py').read_text(encoding='utf-8')
    s=s.replace('from round192_context import choose,hint,names_in','from round201_spans import choose,hint,names_in,align')
    s=s.replace('def bind(v):','def bind(v,node=None):').replace('choose(q,qs,ps,v,line_context,metadata)','choose(q,qs,ps,v,line_context,metadata,operand_context.get(id(node)))')
    s=s.replace('return bind(F(str(n.value)))','return bind(F(str(n.value)),n)')
    s=s.replace('ps=visit(expr)','operand_context=align(line_context,aa,bb,q)\n            ps=visit(expr)')
    s=s.replace('result_context=hint(line_context,target,names_in(q))','result_context=operand_context.get(id(bb),hint(line_context,target,names_in(q)))')
    (r/'round201_trace.py').write_text(s,encoding='utf-8')
    s=(r/'round179_audit.py').read_text(encoding='utf-8').replace('round179_trace','round201_trace').replace("experiments/round179'","experiments/round202'").replace("glob('round179*.py')","glob('round201*.py')")
    (r/'round202_audit.py').write_text(s,encoding='utf-8')
    dump(out/'plan.json',dict(checks=checks(),fit_evidence='round200 fit-only1916 annotated repeated-number lines; reviewed12 include20 houses versus20 items and multi-equation repeated operands. Exact contiguous numeric-token sequence must uniquely align AST leaves; otherwise previous uncertainty-preserving line context.',method='Per-operand explicit unit/object/owner/time spans bounded by arithmetic operators; computed result metadata uses its own result occurrence. Unknown alternatives preserved. No numeric-value-only occurrence matching.',test_opened=False,source_hash=sha(r/'round201_trace.py')))
