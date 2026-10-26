"""Fixed hard-relation ablation: TRAIN relations and full official VAL."""
from data_access import ROOT
from round22_common import read,dump,sha
from round56_selector import infer,select
from round56_local import focus,csr,node_scores
from round44_programs import edits,valid
from collections import Counter,defaultdict
import round46_evaluate
import numpy as np

OUT=ROOT/'experiments/round63'

if __name__=='__main__':
    pairs=[]
    for r in read(OUT/'held_source_review_before_reference.json'):
        counts=Counter();seen=set()
        for kind,p,d in edits(r['program'],r['quantities']):
            a=valid(p,r['quantities']);key=repr(p)
            if a is None or a==r['source_answer'] or key in seen or counts[kind]>=4:continue
            counts[kind]+=1;seen.add(key);pairs.append(dict(train_index=r['train_index'],question=r['question'],positive=r['program'],negative=p,decision=d,kind=kind))
    ff=[focus(p) for p in pairs];xp=csr([p[0] for p in ff]);xn=csr([p[1] for p in ff]);m=[]
    for seed in (48048,48049):
        with np.load(OUT/'diagnostic'/f'sft_s{seed}'/'model.npz') as w:m.append(node_scores(xp,w)-node_scores(xn,w))
    mm=np.mean(m,axis=0);groups=defaultdict(list);parents=defaultdict(list)
    for p,v in zip(pairs,mm):groups[p['kind']].append(bool(v>0));parents[p['train_index']].append(bool(v>0))
    relation=dict(n=len(pairs),correct=int((mm>0).sum()),parent_mean_accuracy=float(np.mean([np.mean(v) for v in parents.values()])),
        by_kind={k:dict(n=len(v),correct=sum(v)) for k,v in groups.items()},test_opened=False)
    dump(OUT/'held_relation_results.json',relation);print('HELD_RELATION',relation,flush=True)
    round46_evaluate.OUT=OUT;round46_evaluate.infer=infer;round46_evaluate.select=select;round46_evaluate.main()
    best=read(OUT/'validation_result.json')[0];best['bundle']='experiments/round63/bundle_sft'
    dump(OUT/'final_feature_selection.json',dict(records=[best],best_new=best,selected='round25' if best['correct']<=752 else 'round63',test_target_fixed_before_test=True,test_used_for_selection=False))
