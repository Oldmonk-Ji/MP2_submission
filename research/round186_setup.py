"""Create isolated optional binding ablation; frozen originals unchanged."""
from round22_common import ROOT,dump,sha
if __name__=='__main__':
    research=ROOT/'research';out=ROOT/'experiments/round186';out.mkdir()
    src=(research/'round179_trace.py').read_text(encoding='utf-8')
    src=src.replace("if kinds[i]!='ordinal':direct", "if kinds[i]!='ordinal' and not (n['raw'].lower()=='once' and re.match(r'\\s+(?:the|a|an|he|she|it|they|water)\\b',q[n['end']:],re.I)):direct")
    src=src.replace("ps=list(direct.get(v,[]))+list(cache.get(v,[]))", "# Optional causal interpretation: prefer explicit verified intermediates.\n        prior=list(cache.get(v,[]))\n        if prior:return unique(prior)\n        ps=list(direct.get(v,[]))\n        # A directly stated quantity is stronger than an accidental inverse fraction.\n        if ps:\n            if v in constants:ps.append(('L',str(v)))\n            return unique(ps)")
    (research/'round186_trace.py').write_text(src,encoding='utf-8')
    audit=(research/'round179_audit.py').read_text(encoding='utf-8').replace('round179_trace','round186_trace').replace("experiments/round179'","experiments/round187'").replace("glob('round179*.py')","glob('round186*.py')")
    (research/'round187_audit.py').write_text(audit,encoding='utf-8')
    dump(out/'plan.json',dict(fit_evidence='TRAIN-fit review183:188 inverse20percent misbound to5shirts;570 intermediate4 eggs versus source4 people/milk;625 temporal Once parsed1. Candidate expressions remain hypotheses.',method='Optional causal binding interpretation: verified intermediate before direct source, direct source before inferred fraction alias; contextual temporal Once excluded. Preserve original179 route in incumbent. This precedence is a heuristic, not semantic truth; measure losses as well as recovery.',test_opened=False,source_sha256=sha(research/'round186_trace.py')))
