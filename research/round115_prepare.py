from round22_common import ROOT,CODE,load,read,dump,sha
from round115_reference import align,checks
from common import read_jsonl
from collections import Counter
import numpy as np,sys,time
OUT=ROOT/'experiments/round115'
if __name__=='__main__':
    mode=sys.argv[1];assert mode in ('fit','full');tick=time.perf_counter()
    if mode=='fit':
        OUT.mkdir();dump(OUT/'plan.json',dict(checks=checks(),hypothesis='Old strict labels silently accepted untracked computed constants and rebound failed intermediate values to equal-valued source quantities. Preserve complete bounded alternatives and reject unresolved dependencies.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round115*.py')},test_opened=False))
    else:
        for src,h in read(OUT/'plan.json')['source_hashes'].items():assert sha(ROOT/src)==h,src
        assert read(OUT/'fit_stats.json')['fit_only']
    b,a,y,ids=load('train');refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    excluded={r['train_index'] for r in read(ROOT/'experiments/round56/held_source_review_before_reference.json')}
    sources=[];reports=[];stats=Counter();prior={r['train_index'] for r in read(ROOT/'experiments/round63/accepted_reference_programs.json')}
    for i,idx in enumerate(ids):
        idx=int(idx)
        if idx in excluded or (mode=='fit' and f[i]==2):continue
        assert b[i]['question']==refs[idx]['question'];ps,r=align(b[i]['question'],refs[idx]['answer'])
        stats['questions']+=1;stats['parents_with_complete_alternatives']+=bool(ps);stats['programs']+=len(ps)
        if idx in prior:stats['previous_reference_parents']+=1;stats['previous_now_abstained']+=not bool(ps)
        for e in r['events']:stats['event_'+e['reason']]+=1
        reports.append(dict(train_index=idx,fold=int(f[i]),**r))
        sources.extend(dict(train_index=idx,fold=int(f[i]),program=p,reference_evidence='complete arithmetic lineage alternatives; unresolved values never replaced by constants') for p in ps)
    dump(OUT/f'{mode}_sources.json',sources);dump(OUT/f'{mode}_reports.json',reports);dump(OUT/f'{mode}_stats.json',dict(stats=stats,fit_only=mode=='fit',seconds=time.perf_counter()-tick,test_opened=False));print(stats,flush=True)
