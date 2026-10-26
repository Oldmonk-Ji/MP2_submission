from round22_common import ROOT,load,dump,sha
from round59_selector import features
from round62_selector import compact,transform,kernel
from round3_train import pair_data
from round25_selector import scores as baseline
from round22_selector import normalized
from round4_gates import apply_gate
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import numpy as np,time,shutil
OUT=ROOT/'experiments/round62'

def fit(x,y,ids,ex,path):
    path.mkdir(parents=True);rng=np.random.default_rng(62062)
    p=PCA(n_components=64,svd_solver='randomized',random_state=62062).fit(x[:,:,:1024].reshape(-1,1024))
    w=dict(pmean=p.mean_.astype(np.float32),pc=p.components_.astype(np.float32),pscale=np.sqrt(p.explained_variance_.clip(p.explained_variance_[0]*.01)).astype(np.float32))
    xx=compact(x,w);w.update(mean=xx.mean(0),scale=xx.std(0).clip(.001),blockscale=np.r_[np.full(192,1/8),np.full(xx.shape[1]-192,1/np.sqrt(xx.shape[1]-192))].astype(np.float32))
    xx=transform(x,w);w['anchors']=xx[rng.choice(len(xx),768,replace=False)];b=w['anchors'];sm=xx[rng.choice(len(xx),768,replace=False)]
    w['median']=np.asarray(np.median(np.maximum((sm*sm).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*sm@b.T,0)),np.float32)
    k=kernel(xx,w);ek=kernel(transform(ex,w),w)
    ev,u=np.linalg.eigh(kernel(b,w).astype(np.float64));norm=(u/np.sqrt(np.maximum(ev,1e-6)))@u.T
    scores={}
    for arm in ('raw','nystrom'):
        z=k if arm=='raw' else (k@norm).astype(np.float32);scale=z.std(0).clip(.001)
        xx,yy,ww=pair_data(sparse.csr_matrix(z/scale),y)
        m=LogisticRegression(C=1,solver='liblinear',fit_intercept=False,max_iter=1500,random_state=62062).fit(xx,yy,sample_weight=ww)
        coef=m.coef_[0]/scale;coef=coef if arm=='raw' else norm@coef
        np.savez(path/f'{arm}.npz',**w,weight=coef.astype(np.float32));scores[arm]=(ek@coef).reshape(-1,4)
        assert np.allclose(k[:16]@coef,(z[:16]/scale)@m.coef_[0],atol=1e-4)
        print('FIT',path.name,arm,'iterations',m.n_iter_,flush=True)
    dump(path/'training.json',dict(training_ids=ids.tolist(),seed=62062,C=1,anchors=768,pca=64))
    return scores

if __name__=='__main__':
    OUT.mkdir();start=time.perf_counter();dump(OUT/'plan.json',dict(arms=['raw','nystrom'],C=1,anchors=768,pca=64,alpha=[.1,.25,.5,1.],test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round62*.py')}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');x,_=features(b,a);vx,_=features(v,va)
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    hs=fit(x[f!=2],y[f!=2],ids[f!=2],x[f==2],OUT/'diagnostic');ss=fit(x,y,ids,vx,OUT/'full')
    bs=normalized(baseline(v,va,ROOT/'experiments/round25/selected_bundle'));bc=apply_gate(v,bs,'parse_and_repair',4);records=[]
    for arm,s in ss.items():
        hb=[r for r,k in zip(b,f) if k==2];hc=apply_gate(hb,hs[arm],'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(correct=int(y[f==2][np.arange(len(hb)),hc].sum()),n=len(hb)))
        np.savez(OUT/f'{arm}_scores.npz',validation=s,held=hs[arm])
        for alpha in (.1,.25,.5,1.):
            c=apply_gate(v,(1-alpha)*bs+alpha*normalized(s),'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
            r=dict(arm=arm,alpha=alpha,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()),predictions=c.tolist());records.append(r);print({k:v for k,v in r.items() if k!='predictions'},flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['alpha']))
    bundle=OUT/'bundle';bundle.mkdir();shutil.copy2(OUT/'full'/f"{best['arm']}.npz",bundle/'model.npz')
    if best['alpha']<1:shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
    dump(bundle/'config.json',dict(alpha=best['alpha']));size=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert size<=16*1024**2
    dump(OUT/f"validation_{best['arm']}_predictions.json",dict(selected=best['predictions']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle='experiments/round62/bundle'),selected='round62' if best['correct']>752 else 'round25',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-start,asset_bytes=size,test_opened=False))
