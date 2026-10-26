from round22_common import ROOT,load,read,dump,sha
from round139_features import pair_features,matrix,probabilities,checks
from round138_literals import LABELS,ABSENT
from sklearn.linear_model import LogisticRegression
from collections import Counter
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round139'

def weights(y,parents):
    count=Counter((int(p),bool(t!=ABSENT)) for p,t in zip(parents,y));groups=Counter(p for p,k in count)
    return np.array([1/(groups[int(p)]*count[(int(p),bool(t!=ABSENT))]) for p,t in zip(parents,y)])

def fit(x,y,parents,mask,stage):
    ww=weights(y[mask],parents[mask]);m=LogisticRegression(C=1,solver='lbfgs',max_iter=700,random_state=139139).fit(x[mask],y[mask],sample_weight=ww)
    assert np.array_equal(m.classes_,np.arange(5));assert int(m.n_iter_[0])<700
    w=dict(weight=m.coef_.astype(np.float32),bias=m.intercept_.astype(np.float32));np.savez(OUT/f'{stage}.npz',**w)
    dump(OUT/f'{stage}.json',dict(training_parent_ids=sorted(map(int,set(parents[mask]))),training_pairs=int(mask.sum()),class_counts={LABELS[k]:int((y[mask]==k).sum()) for k in range(5)},C=1,iterations=m.n_iter_.tolist(),weight_sum=float(ww.sum()),test_opened=False));print('COMPLETE',stage,m.n_iter_.tolist(),flush=True);return w

if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();OUT.mkdir()
    for src,h in read(ROOT/'experiments/round138/completion_manifest.json')['hashes'].items():assert sha(ROOT/src)==h,src
    dump(OUT/'plan.json',dict(checks=checks(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round139*.py')},method='Fixed5-class C1 source/literal classifier from frozen138 labels. Per parent, equal mass on available present and absent groups; no external inputs or semantic gold claims.',test_opened=False))
    b,a,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)};rows=read(ROOT/'experiments/round138/full_labels.json');x=matrix([pair_features(b[index[r['train_index']]]['question'],r['quantity'],r['constant']) for r in rows]);sparse.save_npz(OUT/'pair_features.npz',x)
    targets=np.array([r['label'] for r in rows]);parents=np.array([r['train_index'] for r in rows]);f=np.array([r['fold'] for r in rows]);oof=np.zeros((len(rows),5));nested=np.zeros_like(oof)
    for fold in range(3):
        w=fit(x,targets,parents,f!=fold,f'oof{fold}');oof[f==fold]=probabilities(x[f==fold],w)
    for fold in (0,1):
        w=fit(x,targets,parents,f==1-fold,f'nested{fold}');nested[f==fold]=probabilities(x[f==fold],w)
    fit(x,targets,parents,np.ones(len(rows),bool),'full');nested[f==2]=oof[f==2];np.savez(OUT/'oof_scores.npz',probability=oof,nested_probability=nested,labels=targets,parents=parents,folds=f)
    te=f==2;yy=targets[te];pred=oof[te].argmax(1);ww=weights(yy,parents[te]);report=dict(n=int(te.sum()),correct=int((pred==yy).sum()),present_n=int((yy!=ABSENT).sum()),present_correct=int(((pred==yy)&(yy!=ABSENT)).sum()),absent_n=int((yy==ABSENT).sum()),absent_correct=int(((pred==yy)&(yy==ABSENT)).sum()),parent_group_balanced_accuracy=float(np.average(pred==yy,weights=ww)),parent_group_balanced_log_loss=float(np.average(-np.log(oof[te][np.arange(te.sum()),yy].clip(1e-12)),weights=ww)),by_class={LABELS[k]:dict(n=int((yy==k).sum()),correct=int(((yy==k)&(pred==k)).sum())) for k in range(5)},test_opened=False)
    dump(OUT/'held_relations.json',report);print(report,flush=True);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False,auxiliary_supervision_only=True))
