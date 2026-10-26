from round22_common import ROOT,load,read,dump,sha
from round101_templates import canonical,execute
from round98_influence import dual
from round29_programs import quantities
from fractions import Fraction as F
from collections import Counter
import numpy as np

if __name__=='__main__':
    out=ROOT/'experiments/round101';out.mkdir();b,a,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)}
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:folds=z['folds']
    count=Counter();checks=skips=0;rng=np.random.default_rng(101101)
    for r in read(ROOT/'experiments/round63/accepted_reference_programs.json'):
        i=index[r['train_index']]
        if folds[i]==2:continue
        p=r['program'];normal=canonical(p);count[repr(normal)]+=1;n=len(quantities(b[i]['question']))
        for _ in range(4):
            values=[F(int(v)) for v in rng.integers(1,30,n)]
            try:assert dual(p,values)[0]==execute(normal,values);checks+=1
            except ZeroDivisionError:skips+=1
    report=dict(fit_only=True,test_opened=False,reference_programs=sum(count.values()),unique_templates=len(count),parents_in_repeated_templates=sum(n for n in count.values() if n>=2),templates_at_least_two=sum(n>=2 for n in count.values()),templates_at_least_five=sum(n>=5 for n in count.values()),exact_random_arithmetic_checks=checks,zero_denominator_skips=skips,top_templates=count.most_common(12))
    dump(out/'fit_audit.json',report);dump(out/'normalizer_freeze.json',dict(source_sha256=sha(ROOT/'research/round101_templates.py'),audit_sha256=sha(out/'fit_audit.json'),test_opened=False));print(report,flush=True)
