"""TRAIN-fit-only coverage/identity audit before any held-out factoring use."""
from round22_common import ROOT,load,read,dump,sha
from round84_algebra import recover,relations
from round82_relations import relations as before
from round29_programs import value
from round44_programs import locations
from collections import Counter
import numpy as np,time
OUT=ROOT/'experiments/round84'

if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'fit_plan.json',dict(stage='TRAIN fit folds 0/1 only',no_new_quantity_bindings=True,constants_preserved=True,max_nodes=63,max_sources=8,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round84*.py')}))
    b,a,y,ids=load('train');counts=Counter();rows=[];rng=np.random.default_rng(84084)
    for r in read(ROOT/'experiments/round63/accepted_reference_programs.json'):
        if r['fold']==2:continue
        counts['fit_reference_parents']+=1;p=r['program'];old=before(p);q=recover(p);new=relations(p)
        counts['old_with_pairs']+=int(bool(old));counts['new_with_pairs']+=int(bool(new));counts['recovered_programs']+=int(old is None and q is not None)
        if q is not None:
            n=max(v[1] for _,v in locations(p) if v[0]=='N')+1
            for _ in range(5):
                qs=[dict(value=str(v)) for v in rng.integers(1,31,size=n)]
                try:assert value(p,qs)==value(q,qs);counts['exact_numeric_checks']+=1
                except ZeroDivisionError:counts['zero_denominator_skips']+=1
        if old is None:rows.append(dict(train_index=r['train_index'],old=p,new=q,pair_count=len(new or {})))
    dump(OUT/'fit_recovery_audit.json',dict(counts=dict(counts),seconds=time.perf_counter()-tick,examples=rows,held_opened=False,test_opened=False));print(dict(counts),'seconds',time.perf_counter()-tick,flush=True)
