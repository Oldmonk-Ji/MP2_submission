from round22_common import ROOT,load,read,dump,sha
from round80_features import extract,matrix
from round80_selector import augment
from round79_selector import design,project
from round68_selector import components
from round3_features import grounding
from round3_train import pair_data
from round4_gates import apply_gate
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round80'

def fit(x,y,ix,ids,path):
    rows=(ix[:,None]*4+np.arange(4)).ravel();xx,yy,ww=pair_data(x[rows],y[ix]);m=LogisticRegression(C=1,fit_intercept=False,solver='liblinear',max_iter=1500,random_state=80080).fit(xx,yy,sample_weight=ww)
    path.parent.mkdir(parents=True,exist_ok=True);w=m.coef_[0].astype(np.float32);np.savez(path,weight=w);dump(path.with_suffix('.json'),dict(training_ids=ids[ix].tolist(),C=1,iterations=m.n_iter_.tolist(),labels='candidate final-answer correctness; not human semantic gold'))
    return w

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['unsigned','signed'],C=1,hashed_dimension=16384,meta_rounds=200,meta_leaves=7,test_opened=False,nested=True,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round80*.py')},hypothesis='Learn conditional quantity influence from TRAIN candidate contrasts; remove derivative sign as matched control.'))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']];cache={}
    for split,rows in [('train',b),('validation',v)]:
        rr=[];ee=[];fs={k:[] for k in ('unsigned','signed')}
        for i,row in enumerate(rows):
            r,e,f=extract(row);rr.append(r);ee.append(e)
            for arm in fs:fs[arm].extend(f[arm])
            if i%500==0:print('FEATURES',split,i,flush=True)
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:assert np.array_equal(rr,z['root']);assert np.array_equal(ee,z['extra'])
        cache[split]={arm:matrix(x) for arm,x in fs.items()}
        for arm,x in cache[split].items():sparse.save_npz(OUT/f'{split}_{arm}.npz',x)
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nested=z['components'];folds=z['folds']
    roots={}
    for split in ('train','validation'):
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:roots[split]=(z['root'],z['extra'])
    with np.load(ROOT/'experiments/round79/full_space.npz') as w:context=project(a,w);vc=project(va,w)
    with np.load(ROOT/'experiments/round79/diagnostic_space.npz') as w:nc=project(a,w)
    g=grounding(b);vs,vg=components(v,va,ROOT/'experiments/round25/selected_bundle')
    x,bs=design(b,ss,weights,g,*roots['train'],context,'context_influence');nx,nbs=design(b,nested,weights,g,*roots['train'],nc,'context_influence');vx,vbs=design(v,vs,weights,vg,*roots['validation'],vc,'context_influence')
    records=[];params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True)
    for arm in ('unsigned','signed'):
        xx=cache['train'][arm];oof=np.zeros((len(b),4));ns=np.zeros_like(oof)
        for held in range(3):
            ix=np.flatnonzero(folds!=held);te=np.flatnonzero(folds==held);w=fit(xx,y,ix,ids,OUT/arm/f'oof{held}.npz');rows=(te[:,None]*4+np.arange(4)).ravel();oof[te]=np.asarray(xx[rows]@w).reshape(-1,4)
        ns[folds==2]=oof[folds==2]
        for held in (0,1):
            ix=np.flatnonzero(folds==1-held);te=np.flatnonzero(folds==held);w=fit(xx,y,ix,ids,OUT/arm/f'nested{held}.npz');rows=(te[:,None]*4+np.arange(4)).ravel();ns[te]=np.asarray(xx[rows]@w).reshape(-1,4)
        w=fit(xx,y,np.arange(len(b)),ids,OUT/arm/'full.npz');vscores=np.asarray(cache['validation'][arm]@w).reshape(-1,4);np.savez(OUT/f'{arm}_scores.npz',oof=oof,nested=ns,validation=vscores)
        fitmask=folds!=2;held=folds==2;ax=augment(nx,ns);ds=lgb.Dataset(ax[fitmask].reshape(-1,ax.shape[-1]),label=y[fitmask].ravel(),group=np.full(fitmask.sum(),4),init_score=nbs[fitmask].ravel());hm=lgb.train(params,ds,num_boost_round=200)
        hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(ax[held].reshape(-1,ax.shape[-1])).reshape(-1,4),'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True))
        ax=augment(x,oof);av=augment(vx,vscores);ds=lgb.Dataset(ax.reshape(-1,ax.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel());m=lgb.train(params,ds,num_boost_round=200)
        c=apply_gate(v,vbs+m.predict(av.reshape(-1,av.shape[-1])).reshape(-1,4),'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');m.save_model(str(bundle/'meta.txt'));shutil.copy2(ROOT/'experiments/round79/full_space.npz',bundle/'space.npz');shutil.copy2(OUT/arm/'full.npz',bundle/'flow.npz');dump(bundle/'config.json',dict(arm=arm,weights=weights))
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2;records.append(r);print('VAL',r,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round80/bundle_{best['arm']}"),selected='round80' if best['correct']>754 else 'round79',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
