from round22_common import ROOT,load,read,dump,sha
from round109_alignment import annotate,checks
from scipy import sparse
import numpy as np,sys,time
OUT=ROOT/'experiments/round116'

if __name__=='__main__':
    mode=sys.argv[1];assert mode in ('fit','full');tick=time.perf_counter()
    if mode=='fit':OUT.mkdir();dump(OUT/'plan.json',dict(method='Use complete ambiguity-preserving round115 reference lineages, then exact symbolic rational equality to label compatible alternatives inside existing candidate roots. Supervise only candidates containing both equivalent and non-equivalent programs. No new inference hypotheses.',checks=checks(),loss='For aligned TRAIN parents, half candidate ranking and half within-candidate alignment; otherwise unchanged candidate ranking. Parent mass stays fixed.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round116*.py')},test_opened=False))
    else:
        for src,h in read(OUT/'plan.json')['source_hashes'].items():assert sha(ROOT/src)==h
        assert read(OUT/'fit_stats.json')['fit_only']
    b,a,y,ids=load('train');packets=read(ROOT/'experiments/round102/train_programs.json');ptr=np.load(ROOT/'experiments/round107/train_root_ptr.npy')
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    sources=read(ROOT/f'experiments/round115/{mode}_sources.json');excluded={r['train_index'] for r in read(ROOT/'experiments/round56/held_source_review_before_reference.json')};keep=f!=2 if mode=='fit' else np.ones(len(b),bool)
    gold,active,report=annotate(b,ids,packets,ptr,sources,excluded,keep,lambda i:print('ALIGN',mode,i,flush=True));np.save(OUT/f'{mode}_gold_roots.npy',gold);np.save(OUT/f'{mode}_active_candidates.npy',active)
    # Do not apply alignment to an official-final-incorrect candidate even if an
    # unusual parser/rounding discrepancy makes its source trace equivalent.
    report['active_final_incorrect_candidates']=int((active&(y.ravel()==0)).sum());active &= y.ravel()>0;np.save(OUT/f'{mode}_active_candidates.npy',active)
    if mode=='fit':
        x=sparse.load_npz(ROOT/'experiments/round107/train_root_features.npz');counts=np.diff(ptr);correct=total=0;chance=0.
        for fold in (0,1):
            with np.load(ROOT/f'experiments/round107/latent/nested{fold}.npz') as w:s=np.asarray(x@w['weight']).ravel()
            for c in np.flatnonzero(active&np.repeat(f==fold,4)):
                lo,hi=ptr[c:c+2];correct+=bool(gold[lo+np.argmax(s[lo:hi])]);total+=1;chance+=gold[lo:hi].mean()
        report.update(nested_model_reference_choice_correct=correct,nested_model_reference_choice_n=total,uniform_expected_correct=chance)
    report.update(fit_only=mode=='fit',test_opened=False,seconds=time.perf_counter()-tick);dump(OUT/f'{mode}_stats.json',report);print({k:v for k,v in report.items() if k!='alignment_parent_ids'},flush=True)
