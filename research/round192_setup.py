from round22_common import ROOT,dump,sha
from round192_context import checks
if __name__=='__main__':
    r=ROOT/'research';out=ROOT/'experiments/round192';out.mkdir();s=(r/'round179_trace.py').read_text(encoding='utf-8')
    s=s.replace('LIMIT=64','from round192_context import choose,hint,names_in\nLIMIT=64')
    s=s.replace('accepted=0','accepted=0;metadata={};line_context=""')
    s=s.replace('return unique(ps)\n    def visit','return unique(choose(q,qs,ps,v,line_context,metadata))\n    def visit')
    s=s.replace('for aa,bb,names in list(equations(clean(text)))[:64]:','steps=[(line,eq) for line in clean(text).splitlines() for eq in equations(line)]\n    for line_context,(aa,bb,names) in steps[:64]:')
    s=s.replace('cache[target]=merged;accepted+=1','cache[target]=merged;accepted+=1\n            result_context=hint(line_context,target,names_in(q))\n            for p in ps:\n                if p[0] not in ("N","L"):metadata[repr(p)]=result_context')
    (r/'round192_trace.py').write_text(s,encoding='utf-8')
    s=(r/'round179_audit.py').read_text(encoding='utf-8').replace('round179_trace','round192_trace').replace("experiments/round179'","experiments/round193'").replace("glob('round179*.py')","glob('round192*.py')")
    (r/'round193_audit.py').write_text(s,encoding='utf-8')
    dump(out/'plan.json',dict(checks=checks(),fit_evidence='round191 fit944 same-valued parents; reviewed first8 including minutes/balls, sheep/bushels, first/second day and repeated half. Unknown annotations must not eliminate alternatives.',method='Only reject explicit object/unit/owner/time disagreement when an explicitly compatible option exists. Unknown alternatives survive. No universal intermediate precedence. Source and intermediate contexts use explicit lexical mentions only; no pronoun inference.',test_opened=False,source_hash=sha(r/'round192_trace.py')))
