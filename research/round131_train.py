"""TRAIN-only equivalent source-ID views with unchanged total parent weight."""
from round22_common import ROOT,load,read,dump,sha
from round130_renaming import build,checks
from round3_train import pair_data
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round131'

def fit(x,rev,y,ix,ids,stage):
    assert read(ROOT/f'experiments/round103/local/{stage}.json')['training_ids']==ids[ix].tolist()
    rr=(ix[:,None]*4+np.arange(4)).ravel();features=sparse.vstack([x[rr],rev[rr]],format='csr');labels=np.vstack([y[ix],y[ix]])
    xx,yy,ww=pair_data(features,labels)
    model=LogisticRegression(C=1,fit_intercept=False,solver='liblinear',max_iter=1500,random_state=80080).fit(xx,yy,sample_weight=ww*.5)
    assert int(model.n_iter_[0])<1500;w=model.coef_[0].astype(np.float32);np.savez(OUT/f'{stage}.npz',weight=w)
    dump(OUT/f'{stage}.json',dict(training_ids=ids[ix].tolist(),views_per_parent=2,view_weight=.5,C=1,iterations=model.n_iter_.tolist(),same_parent_all_views_in_same_partition=True,test_opened=False));print('COMPLETE',stage,flush=True);return w

if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();OUT.mkdir();audit=read(ROOT/'experiments/round130/fit_audit.json');assert audit['fit_only']
    for src,h in read(ROOT/'experiments/round130/plan.json')['source_hashes'].items():assert sha(ROOT/src)==h,src
    dump(OUT/'plan.json',dict(method='Two TRAIN-only feature views: original source IDs and reversed source IDs with all context/program references remapped together. Original candidate labels retained. Half weight per view keeps original parent loss mass. Same C1,dimension65536,solver and inference extraction as round103.',checks=checks(),fit_evidence=audit,source_sha256=sha(__file__),test_opened=False))
    b,a,y,ids=load('train');x=sparse.load_npz(ROOT/'experiments/round103/train_local.npz');vx=sparse.load_npz(ROOT/'experiments/round103/validation_local.npz');rev=build(b,read(ROOT/'experiments/round102/train_programs.json'));sparse.save_npz(OUT/'train_reversed_features.npz',rev)
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    fitix=np.flatnonzero(f!=2);delta=rev[(fitix[:,None]*4+np.arange(4)).ravel()]-sparse.load_npz(ROOT/'experiments/round130/fit_reversed_features.npz');assert not delta.nnz
    outputs=dict(oof=np.zeros((len(b),4)),nested=np.zeros((len(b),4)))
    stages=[(f'oof{k}',np.flatnonzero(f!=k),np.flatnonzero(f==k),'oof') for k in range(3)]+[(f'nested{k}',np.flatnonzero(f==1-k),np.flatnonzero(f==k),'nested') for k in (0,1)]+[('full',np.arange(len(b)),None,'validation')]
    for stage,ix,te,kind in stages:
        w=fit(x,rev,y,ix,ids,stage);xx=vx if te is None else x[(te[:,None]*4+np.arange(4)).ravel()];s=np.asarray(xx@w).reshape(-1,4)
        if te is None:outputs[kind]=s
        else:outputs[kind][te]=s
    outputs['nested'][f==2]=outputs['oof'][f==2];np.savez(OUT/'invariant_scores.npz',**outputs)
    te=f==2;c=outputs['oof'][te].argmax(1);r=dict(correct=int(y[te][np.arange(te.sum()),c].sum()),n=int(te.sum()),parse_gate_applied=False);dump(OUT/'held_standalone.json',r);print(r,flush=True)
    # Audit representation sensitivity with untouched outer-held parents.
    with np.load(OUT/'oof2.npz') as w:reversed_score=np.asarray(rev[(np.flatnonzero(te)[:,None]*4+np.arange(4)).ravel()]@w['weight']).reshape(-1,4)
    rc=reversed_score.argmax(1);dump(OUT/'held_invariance.json',dict(changed_choices=int((rc!=c).sum()),questions=int(te.sum()),correctness_flips=int((y[te][np.arange(te.sum()),rc]!=y[te][np.arange(te.sum()),c]).sum()),test_opened=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
