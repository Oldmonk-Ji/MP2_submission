"""Equalize influence-class training mass after parent balancing; fixed ablation."""
from round22_common import ROOT,read,dump,sha
from sklearn.linear_model import LogisticRegression
from collections import Counter
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round100'

def fit(x,y,parents,mask,path):
    count=Counter(parents[mask]);ww=np.asarray([1/count[k] for k in parents[mask]]);yy=y[mask];mass=np.bincount(yy,weights=ww,minlength=3);factors=ww.sum()/(3*mass);ww*=factors[yy]
    m=LogisticRegression(C=1,solver='lbfgs',max_iter=700,random_state=82082).fit(x[mask],yy,sample_weight=ww)
    assert np.array_equal(m.classes_,np.arange(3));np.savez(path,weight=m.coef_.astype(np.float32),bias=m.intercept_.astype(np.float32));dump(path.with_suffix('.json'),dict(training_parent_ids=sorted(map(int,set(parents[mask]))),training_quantity_labels=int(mask.sum()),C=1,iterations=m.n_iter_.tolist(),class_factors=factors.tolist(),class_weights_fit_only=True));return m

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(control='round98 occurrence, same labels/features/optimizer; only equal class mass after parent balancing',fit_audit=read(ROOT/'experiments/round98/fit_class_balance_audit.json'),source_sha256=sha(__file__),C=1,max_iter=700,test_opened=False))
    rows=read(ROOT/'experiments/round98/labels.json');x=sparse.load_npz(ROOT/'experiments/round98/quantity_features.npz');y=np.array([r['label'] for r in rows]);parents=np.array([r['train_index'] for r in rows]);f=np.array([r['fold'] for r in rows]);oof=np.zeros((len(y),3))
    for held in range(3):
        m=fit(x,y,parents,f!=held,OUT/f'oof{held}.npz');oof[f==held]=m.predict_proba(x[f==held])
    for held in (0,1):fit(x,y,parents,f==1-held,OUT/f'nested{held}.npz')
    fit(x,y,parents,np.ones(len(y),bool),OUT/'full.npz');np.savez(OUT/'oof_scores.npz',probability=oof,labels=y,parents=parents,folds=f)
    te=f==2;report=dict(held_total=int(te.sum()),held_correct=int((oof[te].argmax(1)==y[te]).sum()),by_class={str(k):dict(n=int((te&(y==k)).sum()),correct=int((te&(y==k)&(oof.argmax(1)==k)).sum())) for k in range(3)},test_opened=False)
    dump(OUT/'held_results.json',report);print(report,flush=True);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,auxiliary_supervision_only=True,test_opened=False))
