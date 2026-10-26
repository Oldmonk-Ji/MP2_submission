"""Matched source-supervision ablation; constant/scope remain frozen159."""
from round22_common import ROOT,load,read,dump,sha
from round159_factors import matrix,probabilities
from round172_target_source import source_features,checks
from round159_train import weights
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round173'

def fit(x,y,parents,mask):
    ww=weights(y[mask],parents[mask],True);total=ww.sum()
    for label in (0,1):ww[y[mask]==label]*=total/(2*ww[y[mask]==label].sum())
    m=LogisticRegression(C=1,solver='lbfgs',max_iter=700,random_state=139139).fit(x[mask],y[mask],sample_weight=ww);assert m.n_iter_[0]<700;assert np.array_equal(m.classes_,[0,1])
    w=dict(source_weight=np.r_[np.zeros_like(m.coef_),m.coef_].astype(np.float32),source_bias=np.array([0,m.intercept_[0]],np.float32));assert np.allclose(probabilities(x[mask][:20],w,'source'),m.predict_proba(x[mask][:20]),atol=1e-6)
    return w,dict(training_parent_ids=sorted(map(int,set(parents[mask]))),training_labels=int(mask.sum()),iterations=m.n_iter_.tolist(),weight_sum=float(ww.sum()),class_weight_sum=[float(ww[y[mask]==k].sum()) for k in (0,1)])

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter()
    for src,h in read(ROOT/'experiments/round169_refined/plan.json')['source_hashes'].items():assert sha(ROOT/src)==h
    dump(OUT/'plan.json',dict(arms=['interaction'],checks=checks(),method='Matched170 reference_mask with target/local feature products; same labels, loss, hash size, C1 and frozen159 constant/scope.',source_sha256=sha(__file__),test_opened=False))
    b,a,y,ids=load('train');index={int(v):i for i,v in enumerate(ids)};original=read(ROOT/'experiments/round158/full_source_labels.json');ox=matrix([source_features(b[index[r['train_index']]]['question'],r['source']) for r in original]);oy=np.array([r['label'] for r in original]);op=np.array([r['train_index'] for r in original]);of=np.array([r['fold'] for r in original]);refined=read(ROOT/'experiments/round169_refined/full_source_labels.json');valid={(r['train_index'],r['source']) for r in refined};trusted=np.array([(r['train_index'],r['source']) in valid for r in original])
    for arm,rows in [('interaction',refined)]:
        folder=OUT/arm;folder.mkdir();x=matrix([source_features(b[index[r['train_index']]]['question'],r['source']) for r in rows]);yy=np.array([r['label'] for r in rows]);pp=np.array([r['train_index'] for r in rows]);ff=np.array([r['fold'] for r in rows]);oof=np.zeros((len(original),2),np.float32);nested=np.zeros_like(oof)
        for stage in ('oof0','oof1','oof2','nested0','nested1','full'):
            held=int(stage[-1]) if stage!='full' else None;mask=np.ones(len(rows),bool) if held is None else ff==1-held if stage.startswith('nested') else ff!=held
            sw,meta=fit(x,yy,pp,mask)
            with np.load(ROOT/f'experiments/round159/{stage}.npz') as z:w={k:z[k] for k in z.files}
            w.update(sw);metadata=read(ROOT/f'experiments/round159/{stage}.json');metadata['heads']['source']=meta;metadata['training_parent_ids']=sorted(set(p for h in metadata['heads'].values() for p in h['training_parent_ids']));metadata['source_arm']=arm
            np.savez(folder/f'{stage}.npz',**w);dump(folder/f'{stage}.json',metadata)
            if held is not None:
                test=of==held;assert not set(op[test])&set(metadata['training_parent_ids']);(nested if stage.startswith('nested') else oof)[test]=probabilities(ox[test],w,'source')
            print('COMPLETE',arm,stage,meta['iterations'],flush=True)
        nested[of==2]=oof[of==2];np.savez(folder/'source_oof_scores.npz',probability=oof,nested_probability=nested,labels=oy,parents=op,folds=of,retained_by_reference_mask=trusted)
        report={}
        for name,mask in [('original_reference_labels',of==2),('common_retained_labels',(of==2)&trusted)]:
            pred=oof[mask].argmax(1);target=oy[mask];report[name]=dict(n=int(mask.sum()),correct=int((pred==target).sum()),by_class={str(k):dict(n=int((target==k).sum()),correct=int(((target==k)&(pred==k)).sum())) for k in (0,1)})
        dump(folder/'held_source.json',report);print('HELD',arm,report,flush=True)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
