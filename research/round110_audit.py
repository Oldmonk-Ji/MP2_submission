"""Evaluate held-out intermediate decisions without altering the frozen rules."""
from round22_common import ROOT,load,read,dump,sha
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np
OUT=ROOT/'experiments/round110'
if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('train');base=ROOT/'experiments/round107'
    x=sparse.load_npz(base/'train_root_features.npz');ptr=np.load(base/'train_root_ptr.npy')
    gold=np.load(ROOT/'experiments/round109/full_gold_roots.npy');active=np.load(ROOT/'experiments/round109/full_active_candidates.npy')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:folds=z['folds'];assert np.array_equal(ids,z['ids'])
    reports=[]
    for part,fs in [('fit_nested',(0,1)),('outer_held',(2,))]:
        for arm in ('latent','aligned'):
            correct=n=0;prob_sum=0.;parents=set()
            for fold in fs:
                stage=f'nested{fold}' if fold!=2 else 'oof2'
                path=base/'latent'/f'{stage}.npz' if arm=='latent' else OUT/f'{stage}.npz'
                training=set(read(path.with_suffix('.json'))['training_ids'])
                with np.load(path) as w:s=np.asarray(x@w['weight']).ravel()
                for c in np.flatnonzero(active&np.repeat(folds==fold,4)):
                    assert int(ids[c//4]) not in training
                    lo,hi=ptr[c:c+2];g=gold[lo:hi];ss=s[lo:hi];p=np.exp(ss-ss.max());p/=p.sum()
                    correct+=bool(g[ss.argmax()]);n+=1;prob_sum+=p[g].sum();parents.add(int(ids[c//4]))
            r=dict(part=part,arm=arm,correct=correct,n=n,parents=len(parents),mean_reference_probability=float(prob_sum/max(n,1)));reports.append(r);print(r,flush=True)
    no_pair=(y.min(1)>0);r=dict(aligned_all_correct_parents=int((no_pair&active.reshape(-1,4).any(1)).sum()),note='All-correct parents have zero candidate ranking loss and receive alignment mass 1; mixed parents retain total mass 2.',source_sha256=sha(__file__),reports=reports,test_opened=False)
    dump(OUT/'intermediate_decision_audit.json',r)
