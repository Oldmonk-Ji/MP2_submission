from round22_common import ROOT,load,read,dump,sha
from round159_factors import source_features,constant_features,pair_features,matrix,probabilities,predict,checks
from sklearn.linear_model import LogisticRegression
from collections import Counter
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round159'

def weights(y,parents,binary):
    if not binary:
        count=Counter(parents);return np.array([1/count[p] for p in parents])
    count=Counter(zip(parents,y));groups=Counter(p for p,c in count)
    return np.array([1/(groups[p]*count[(p,c)]) for p,c in zip(parents,y)])

def fit(data,mask,name):
    x,y,parents,folds=data;ww=weights(y[mask],parents[mask],name!='scope');k=4 if name=='scope' else 2
    m=LogisticRegression(C=1,solver='lbfgs',max_iter=700,random_state=139139).fit(x[mask],y[mask],sample_weight=ww);assert m.n_iter_[0]<700
    coef=np.zeros((k,x.shape[1]),np.float32);bias=np.full(k,-30,np.float32)
    if len(m.classes_)==2:
        bias[m.classes_[0]]=0;coef[m.classes_[1]]=m.coef_[0];bias[m.classes_[1]]=m.intercept_[0]
    else:coef[m.classes_]=m.coef_;bias[m.classes_]=m.intercept_
    w={name+'_weight':coef,name+'_bias':bias};p=probabilities(x[mask][:20],w,name);assert np.allclose(p[:,m.classes_],m.predict_proba(x[mask][:20]),atol=1e-6)
    return w,dict(training_parent_ids=sorted(map(int,set(parents[mask]))),training_labels=int(mask.sum()),iterations=m.n_iter_.tolist(),classes=m.classes_.tolist())

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter()
    for src,h in read(ROOT/'experiments/round158/completion_manifest.json')['hashes'].items():assert sha(ROOT/src)==h,src
    dump(OUT/'plan.json',dict(checks=checks(),method='Fixed C1 lexical-only factorization; source and constant binary labels balanced by available classes per parent; scope labels normalized per parent. Same original OOF/nested folds. No parameter search.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round159*.py')},test_opened=False))
    b,a,y,ids=load('train');index={int(idx):i for i,idx in enumerate(ids)};data={};oof_heads={}
    for name in ('source','constant','scope'):
        rows=read(ROOT/f'experiments/round158/full_{name}_labels.json');fs=[]
        for r in rows:
            q=b[index[r['train_index']]]['question'];fs.append(source_features(q,r['source']) if name=='source' else constant_features(q,r['constant']) if name=='constant' else pair_features(q,r['quantity'],r['constant']))
        x=matrix(fs);sparse.save_npz(OUT/f'{name}_features.npz',x);targets=np.array([r['label'] for r in rows]);parents=np.array([r['train_index'] for r in rows]);folds=np.array([r['fold'] for r in rows]);data[name]=(x,targets,parents,folds);oof_heads[name]=np.zeros((len(rows),4 if name=='scope' else 2),np.float32)
    pair_rows=read(ROOT/'experiments/round148/full_labels.json');parents=np.array([r['train_index'] for r in pair_rows]);f=np.array([r['fold'] for r in pair_rows]);targets=np.array([r['label'] for r in pair_rows]);sx=matrix([source_features(b[index[r['train_index']]]['question'],r['quantity']) for r in pair_rows]);cx=matrix([constant_features(b[index[r['train_index']]]['question'],r['constant']) for r in pair_rows]);px=sparse.load_npz(ROOT/'experiments/round149/pair_features.npz');oof=np.zeros((len(pair_rows),5),np.float32);nested=np.zeros_like(oof)
    for stage in ('oof0','oof1','oof2','nested0','nested1','full'):
        w={};metadata={};held=int(stage[-1]) if stage!='full' else None
        for name,d in data.items():
            mask=np.ones(len(d[1]),bool) if stage=='full' else d[3]==1-held if stage.startswith('nested') else d[3]!=held
            ww,meta=fit(d,mask,name);w.update(ww);metadata[name]=meta
            if stage.startswith('oof'):oof_heads[name][d[3]==held]=probabilities(d[0][d[3]==held],w,name)
        np.savez(OUT/f'{stage}.npz',**w);union=sorted(set(p for m in metadata.values() for p in m['training_parent_ids']));dump(OUT/f'{stage}.json',dict(training_parent_ids=union,heads=metadata,test_opened=False))
        if held is not None:
            mask=f==held;assert not set(parents[mask])&set(union);prob=predict(px[mask],sx[mask],cx[mask],w)
            (nested if stage.startswith('nested') else oof)[mask]=prob
        print('COMPLETE',stage,{k:v['iterations'] for k,v in metadata.items()},flush=True)
    nested[f==2]=oof[f==2];np.savez(OUT/'oof_scores.npz',probability=oof,nested_probability=nested,labels=targets,parents=parents,folds=f)
    reports={}
    for name,d in data.items():
        mask=d[3]==2;pred=oof_heads[name][mask].argmax(1);yy=d[1][mask];reports[name]=dict(n=int(mask.sum()),correct=int((pred==yy).sum()),by_class={str(k):dict(n=int((yy==k).sum()),correct=int(((yy==k)&(pred==k)).sum())) for k in range(oof_heads[name].shape[1])})
    with np.load(ROOT/'experiments/round149/oof_scores.npz') as z:old=z['probability'];assert np.array_equal(z['parents'],parents)
    te=f==2;yy=targets[te];pp=oof[te];pred=pp.argmax(1);present=yy!=4
    reports['composed']=dict(n=int(te.sum()),old_correct=int((old[te].argmax(1)==yy).sum()),new_correct=int((pred==yy).sum()),present_n=int(present.sum()),old_present_correct=int(((old[te].argmax(1)==yy)&present).sum()),new_present_correct=int(((pred==yy)&present).sum()),old_log_loss=float(-np.log(old[te][np.arange(te.sum()),yy].clip(1e-12)).mean()),new_log_loss=float(-np.log(pp[np.arange(te.sum()),yy].clip(1e-12)).mean()))
    dump(OUT/'held_heads.json',reports);print(reports,flush=True);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
