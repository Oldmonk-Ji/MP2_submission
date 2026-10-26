from round22_common import ROOT,load,read,dump,sha
from round130_renaming import build,checks
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round130'
if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();OUT.mkdir();dump(OUT/'plan.json',dict(checks=checks(),method='Reverse only internal source-quantity identifiers and their program/context references. Keep question wording, numerical values, candidate texts and final labels fixed. Audit fit-only cross-fitted prediction invariance before augmentation.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round130*.py')},test_opened=False))
    b,a,y,ids=load('train');packets=read(ROOT/'experiments/round102/train_programs.json')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    ix=np.flatnonzero(f!=2);x=build([b[i] for i in ix],[packets[i] for i in ix]);sparse.save_npz(OUT/'fit_reversed_features.npz',x)
    old=sparse.load_npz(ROOT/'experiments/round103/train_local.npz')[(ix[:,None]*4+np.arange(4)).ravel()];control=np.zeros((len(ix),4));scores=np.zeros_like(control)
    for fold in (0,1):
        path=ROOT/f'experiments/round103/local/nested{fold}.npz';assert not set(ids[ix][f[ix]==fold])&set(read(path.with_suffix('.json'))['training_ids'])
        with np.load(path) as w:weights=w['weight']
        keep=np.flatnonzero(f[ix]==fold);rr=(keep[:,None]*4+np.arange(4)).ravel();control[keep]=np.asarray(old[rr]@weights).reshape(-1,4);scores[keep]=np.asarray(x[rr]@weights).reshape(-1,4)
    c=control.argmax(1);p=scores.argmax(1);mixed=y[ix].min(1)==0;ok=y[ix][np.arange(len(ix)),c]>0;new=y[ix][np.arange(len(ix)),p]>0
    r=dict(fit_only=True,questions=len(ix),mixed_questions=int(mixed.sum()),changed_choices=int((p!=c).sum()),mixed_changed_choices=int(((p!=c)&mixed).sum()),old_correct=int(ok.sum()),renamed_correct=int(new.sum()),correctness_flips=int((ok!=new).sum()),test_opened=False,seconds=time.perf_counter()-tick)
    dump(OUT/'fit_audit.json',r);np.savez(OUT/'fit_scores.npz',ids=ids[ix],control=control,renamed=scores,labels=y[ix]);print(r,flush=True)
