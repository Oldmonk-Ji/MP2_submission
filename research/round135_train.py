from round22_common import ROOT,load,read,dump,sha
from round135_interaction import question_context,interaction,probabilities,checks
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA,TruncatedSVD
from collections import Counter
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round135'
RANK=16

def fit(x,q,owners,y,parents,mask,stage):
    count=Counter(parents[mask]);ww=np.asarray([1/count[p] for p in parents[mask]]);context_ix=np.unique(owners[mask]);mean=q[context_ix].mean(0);tick=time.perf_counter()
    lexical_rms=float(np.sqrt(np.average(np.asarray(x[mask].power(2).sum(1)).ravel(),weights=ww)));semantic_rms=float(np.sqrt(np.square(q[context_ix]-mean).sum(1).mean()));scale=lexical_rms/max(semantic_rms,1e-12)
    svd=TruncatedSVD(n_components=RANK,n_iter=5,random_state=135135).fit(x[mask].multiply(np.sqrt(ww)[:,None]));pair_pc=svd.components_.astype(np.float32)
    pca=PCA(n_components=RANK,svd_solver='randomized',random_state=135135).fit(q[context_ix]-mean);context_pc=pca.components_.astype(np.float32)
    lp=np.asarray(x[mask]@pair_pc.T);lm=np.average(lp,axis=0,weights=ww);ls=np.sqrt(np.average((lp-lm)**2,axis=0,weights=ww)).clip(1e-6)
    cp=(q[context_ix]-mean)@context_pc.T;cs=np.sqrt(np.square(cp).mean(0)).clip(1e-6)
    w=dict(pair_pc=pair_pc,pair_mean=lm,pair_std=ls,context_pc=context_pc,context_std=cs,context_mean=mean,context_scale=np.array(scale),interaction_scale=np.array(1.))
    z=interaction(x[mask],q,owners[mask],w);cross_rms=float(np.sqrt(np.average(np.square(z).sum(1),weights=ww)));w['interaction_scale']=np.array(lexical_rms/max(cross_rms,1e-12));z*=float(w['interaction_scale'])
    xx=sparse.hstack([x[mask],sparse.csr_matrix(((q-mean)*scale)[owners[mask]]),sparse.csr_matrix(z)],format='csr')
    m=LogisticRegression(C=1,solver='lbfgs',max_iter=700,random_state=82082).fit(xx,y[mask],sample_weight=ww);assert np.array_equal(m.classes_,np.arange(6));assert int(m.n_iter_[0])<700
    d=x.shape[1];c=q.shape[1];w.update(weight=m.coef_[:,:d].astype(np.float32),context_weight=m.coef_[:,d:d+c].astype(np.float32),interaction_weight=m.coef_[:,d+c:].astype(np.float32),bias=m.intercept_.astype(np.float32))
    np.savez(OUT/f'{stage}.npz',**w);meta=dict(training_parent_ids=sorted(map(int,set(parents[mask]))),preprocessing_training_parent_ids=sorted(map(int,set(parents[mask]))),training_pairs=int(mask.sum()),C=1,rank=RANK,iterations=m.n_iter_.tolist(),context_scale=scale,interaction_scale=float(w['interaction_scale']),lexical_block_rms=lexical_rms,semantic_block_rms=semantic_rms,interaction_block_rms=cross_rms,seconds=time.perf_counter()-tick,test_opened=False)
    dump(OUT/f'{stage}.json',meta);print('COMPLETE',stage,{k:v for k,v in meta.items() if 'parent_ids' not in k},flush=True);return w

if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();OUT.mkdir();dump(OUT/'plan.json',dict(checks=checks(),method='Keep round133 additive semantic context and conservative consensus labels; add an outer product of16-dimensional pair SVD and16-dimensional semantic PCA projections, trained only on allowed stage parents. Same C1 logistic classifier; fixed rank16 from asset envelope, no sweep.',fit_evidence=read(ROOT/'experiments/round133/fit_context_audit.json'),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round135*.py')},expected_extra_projection_bytes=16*(16384+1024)*4,test_opened=False))
    b,a,y,ids=load('train');q=question_context(a);index={int(idx):i for i,idx in enumerate(ids)};rows=read(ROOT/'experiments/round120/labels.json');x=sparse.load_npz(ROOT/'experiments/round120/pair_features.npz')
    targets=np.array([r['label'] for r in rows]);parents=np.array([r['train_index'] for r in rows]);f=np.array([r['fold'] for r in rows]);owners=np.array([index[int(p)] for p in parents]);oof=np.zeros((len(rows),6));nested=np.zeros_like(oof)
    for fold in range(3):
        w=fit(x,q,owners,targets,parents,f!=fold,f'oof{fold}');oof[f==fold]=probabilities(x[f==fold],q,owners[f==fold],w)
    for fold in (0,1):
        w=fit(x,q,owners,targets,parents,f==1-fold,f'nested{fold}');nested[f==fold]=probabilities(x[f==fold],q,owners[f==fold],w)
    fit(x,q,owners,targets,parents,np.ones(len(rows),bool),'full');nested[f==2]=oof[f==2];np.savez(OUT/'oof_scores.npz',probability=oof,nested_probability=nested,labels=targets,parents=parents,folds=f)
    with np.load(ROOT/'experiments/round133/oof_scores.npz') as z:old=z['probability'];assert np.array_equal(z['parents'],parents)
    te=f==2;report=dict(n=int(te.sum()),old_correct=int((old[te].argmax(1)==targets[te]).sum()),new_correct=int((oof[te].argmax(1)==targets[te]).sum()),old_log_loss=float(-np.log(old[te][np.arange(te.sum()),targets[te]].clip(1e-12)).mean()),new_log_loss=float(-np.log(oof[te][np.arange(te.sum()),targets[te]].clip(1e-12)).mean()),test_opened=False)
    dump(OUT/'held_relations.json',report);print(report,flush=True);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False,auxiliary_supervision_only=True))
