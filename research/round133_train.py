from round22_common import ROOT,load,read,dump,sha
from round133_context import question_context,probabilities,checks
from sklearn.linear_model import LogisticRegression
from collections import Counter
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round133'

def fit(x,q,owners,y,parents,mask,stage):
    count=Counter(parents[mask]);ww=np.asarray([1/count[p] for p in parents[mask]]);context_ix=np.unique(owners[mask]);mean=q[context_ix].mean(0)
    lexical_rms=float(np.sqrt(np.average(np.asarray(x[mask].power(2).sum(1)).ravel(),weights=ww)));semantic_rms=float(np.sqrt(np.square(q[context_ix]-mean).sum(1).mean()));scale=lexical_rms/max(semantic_rms,1e-12)
    xx=sparse.hstack([x[mask],sparse.csr_matrix(((q-mean)*scale)[owners[mask]])],format='csr');tick=time.perf_counter()
    m=LogisticRegression(C=1,solver='lbfgs',max_iter=700,random_state=82082).fit(xx,y[mask],sample_weight=ww)
    assert np.array_equal(m.classes_,np.arange(6));assert int(m.n_iter_[0])<700
    w=dict(weight=m.coef_[:,:x.shape[1]].astype(np.float32),context_weight=m.coef_[:,x.shape[1]:].astype(np.float32),bias=m.intercept_.astype(np.float32),context_mean=mean,context_scale=np.array(scale))
    np.savez(OUT/f'{stage}.npz',**w);meta=dict(training_parent_ids=sorted(map(int,set(parents[mask]))),context_training_parent_ids=sorted(map(int,set(parents[mask]))),training_pairs=int(mask.sum()),C=1,iterations=m.n_iter_.tolist(),context_scale=scale,lexical_block_rms=lexical_rms,semantic_block_rms=semantic_rms,seconds=time.perf_counter()-tick,test_opened=False)
    dump(OUT/f'{stage}.json',meta);print('COMPLETE',stage,{k:v for k,v in meta.items() if 'parent_ids' not in k},flush=True);return w

if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();OUT.mkdir();dump(OUT/'plan.json',dict(checks=checks(),method='Matched round120 conservative consensus labels and C1 six-way logistic classifier. Add mean supplied candidate embedding as a question semantic context block; fit centering and norm matching only on stage training parents.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round133*.py')},test_opened=False))
    b,a,y,ids=load('train');q=question_context(a);index={int(idx):i for i,idx in enumerate(ids)};rows=read(ROOT/'experiments/round120/labels.json');x=sparse.load_npz(ROOT/'experiments/round120/pair_features.npz')
    targets=np.array([r['label'] for r in rows]);parents=np.array([r['train_index'] for r in rows]);f=np.array([r['fold'] for r in rows]);owners=np.array([index[int(p)] for p in parents]);oof=np.zeros((len(rows),6));nested=np.zeros_like(oof)
    for fold in range(3):
        w=fit(x,q,owners,targets,parents,f!=fold,f'oof{fold}');oof[f==fold]=probabilities(x[f==fold],q,owners[f==fold],w)
    for fold in (0,1):
        w=fit(x,q,owners,targets,parents,f==1-fold,f'nested{fold}');nested[f==fold]=probabilities(x[f==fold],q,owners[f==fold],w)
    fit(x,q,owners,targets,parents,np.ones(len(rows),bool),'full');nested[f==2]=oof[f==2]
    np.savez(OUT/'oof_scores.npz',probability=oof,nested_probability=nested,labels=targets,parents=parents,folds=f)
    with np.load(ROOT/'experiments/round120/oof_scores.npz') as z:old=z['probability'];assert np.array_equal(z['parents'],parents)
    te=f==2;report=dict(n=int(te.sum()),old_correct=int((old[te].argmax(1)==targets[te]).sum()),new_correct=int((oof[te].argmax(1)==targets[te]).sum()),old_log_loss=float(-np.log(old[te][np.arange(te.sum()),targets[te]].clip(1e-12)).mean()),new_log_loss=float(-np.log(oof[te][np.arange(te.sum()),targets[te]].clip(1e-12)).mean()),test_opened=False)
    dump(OUT/'held_relations.json',report);print(report,flush=True);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False,auxiliary_supervision_only=True))
