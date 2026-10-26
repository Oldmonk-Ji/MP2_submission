"""Fixed factorization ablation using the original frozen relation labels."""
from round22_common import ROOT,read,dump,sha
from round90_relations import FAMILY,direction_matrix,probabilities
from round82_relations import LABELS
from sklearn.linear_model import LogisticRegression
from collections import Counter
from threadpoolctl import threadpool_limits
from scipy import sparse
import numpy as np,time
OUT=ROOT/'experiments/round90'

def fit(x,y,parents,mask,path):
    xx=x[mask];yy=y[mask];pp=parents[mask];count=Counter(pp);ww=np.asarray([1/count[p] for p in pp])
    args=dict(C=1,solver='lbfgs',max_iter=700,random_state=82082)
    family=LogisticRegression(**args).fit(xx,FAMILY[yy],sample_weight=ww)
    dm=np.isin(yy,[1,2,4,5]);direction=LogisticRegression(**args).fit(direction_matrix(xx[dm],yy[dm]>=4),np.isin(yy[dm],[2,5]).astype(int),sample_weight=ww[dm])
    assert np.array_equal(family.classes_,np.arange(4));assert np.array_equal(direction.classes_,[0,1])
    state=dict(family_weight=family.coef_.astype(np.float32),family_bias=family.intercept_.astype(np.float32),direction_weight=direction.coef_.astype(np.float32),direction_bias=direction.intercept_.astype(np.float32))
    p=probabilities(xx,state);assert np.allclose(p.sum(1),1)
    # Independently verify the shared and operation-specific direction decomposition.
    direct=direction.predict_proba(direction_matrix(xx[dm],yy[dm]>=4))[:,1]
    reverse=np.where(yy[dm]>=4,p[dm,5]/(p[dm,4]+p[dm,5]),p[dm,2]/(p[dm,1]+p[dm,2]))
    assert np.allclose(direct,reverse,atol=1e-5)
    np.savez(path,**state);dump(path.with_suffix('.json'),dict(training_parent_ids=sorted(map(int,set(pp))),training_pairs=len(yy),direction_pairs=int(dm.sum()),C=1,iterations_family=family.n_iter_.tolist(),iterations_direction=direction.n_iter_.tolist()))
    return state

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter()
    dump(OUT/'plan.json',dict(hypothesis='Learn ADD/SUB/MUL/DIV families separately, with shared and family-specific direction weights; compose normalized six-class probabilities.',training_label_source='round82',direction_weights='Original per-parent weights restricted to noncommutative pairs',C=1,max_iter=700,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round90*.py')},fit_only_diagnostic=read(ROOT/'experiments/train_fit_relation_factorization_audit_20261026.json'),test_opened=False))
    rows=read(ROOT/'experiments/round82/labels.json');x=sparse.load_npz(ROOT/'experiments/round82/pair_features.npz');y=np.asarray([r['label'] for r in rows]);parents=np.asarray([r['train_index'] for r in rows]);f=np.asarray([r['fold'] for r in rows]);oof=np.zeros((len(y),6))
    for held in range(3):
        state=fit(x,y,parents,f!=held,OUT/f'oof{held}.npz');oof[f==held]=probabilities(x[f==held],state);print('OOF',held,int((oof[f==held].argmax(1)==y[f==held]).sum()),int((f==held).sum()),flush=True)
    for held in (0,1):fit(x,y,parents,f==1-held,OUT/f'nested{held}.npz')
    fit(x,y,parents,np.ones(len(y),bool),OUT/'full.npz');np.savez(OUT/'oof_scores.npz',probability=oof,labels=y,parents=parents,folds=f)
    te=f==2;pred=oof[te].argmax(1);report=dict(held_pairs=int(te.sum()),held_correct=int((pred==y[te]).sum()),by_class={LABELS[k]:dict(n=int((y[te]==k).sum()),correct=int(((y[te]==k)&(pred==k)).sum())) for k in range(6)},test_opened=False)
    dump(OUT/'held_relation_classification.json',report);print('HELD',report,flush=True);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False,auxiliary_supervision_only=True))
