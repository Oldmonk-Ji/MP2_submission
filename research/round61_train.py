"""Nonlinear rank interactions on compact TRAIN-only PCA representations."""
from round22_common import ROOT,load,dump,sha
from round59_selector import features
from round25_selector import scores as baseline
from round22_selector import normalized
from round4_gates import apply_gate
from sklearn.decomposition import PCA
import lightgbm as lgb
import numpy as np,time
OUT=ROOT/'experiments/round61'

def design(x,space,context):
    e=(x[:,:,:1024]-space['mean'])@space['components'].T;s=x[:,:,1024:]
    if not context:return np.concatenate([e,s],2)
    return np.concatenate([e,e-e.mean(1,keepdims=True),e*e.mean(1,keepdims=True),s,s-s.mean(1,keepdims=True)],2)

def fit(x,y,ids,path,context):
    path.mkdir(parents=True);p=PCA(n_components=64,svd_solver='randomized',random_state=61061).fit(x[:,:,:1024].reshape(-1,1024))
    space=dict(mean=p.mean_.astype(np.float32),components=p.components_.astype(np.float32));np.savez(path/'space.npz',**space)
    xx=design(x,space,context)
    ds=lgb.Dataset(xx.reshape(-1,xx.shape[-1]),label=y.ravel(),group=np.full(len(x),4))
    params=dict(objective='lambdarank',metric='None',num_leaves=15,min_data_in_leaf=50,learning_rate=.03,lambda_l2=10,feature_fraction=.8,num_threads=4,seed=61061,verbosity=-1,deterministic=True,force_col_wise=True)
    m=lgb.train(params,ds,num_boost_round=600);m.save_model(str(path/'model.txt'))
    dump(path/'training.json',dict(training_ids=ids.tolist(),context=context,params=params,rounds=600))
    return lambda x:m.predict(design(x,space,context).reshape(-1,xx.shape[-1])).reshape(-1,4)

if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(test_opened=False,arms=['independent','context'],rounds=600,alpha=[.1,.25,.5,1.],source_hash=sha(__file__)))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');x,_=features(b,a);vx,_=features(v,va)
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    bs=normalized(baseline(v,va,ROOT/'experiments/round25/selected_bundle'));bc=apply_gate(v,bs,'parse_and_repair',4);records=[]
    for context in (False,True):
        arm='context' if context else 'independent';fn=fit(x[f!=2],y[f!=2],ids[f!=2],OUT/'diagnostic'/arm,context)
        hb=[r for r,k in zip(b,f) if k==2];h=fn(x[f==2]);hc=apply_gate(hb,h,'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(n=len(hb),correct=int(y[f==2][np.arange(len(hb)),hc].sum())))
        print('HELD',arm,int(y[f==2][np.arange(len(hb)),hc].sum()),flush=True)
        fn=fit(x,y,ids,OUT/'full'/arm,context);s=normalized(fn(vx));np.savez(OUT/f'{arm}_scores.npz',validation=s,held=h)
        for alpha in (.1,.25,.5,1.):
            c=apply_gate(v,(1-alpha)*bs+alpha*s,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
            r=dict(arm=arm,alpha=alpha,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()),predictions=c.tolist());records.append(r);print({k:v for k,v in r.items() if k!='predictions'},flush=True)
    dump(OUT/'validation_records.json',records);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
