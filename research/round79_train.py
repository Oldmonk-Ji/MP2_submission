from round22_common import ROOT,load,read,dump,sha
from round79_selector import design,project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round79'

def space(a,mask):
    p=PCA(n_components=64,svd_solver='randomized',random_state=79079).fit(a['embeddings'][mask].reshape(-1,1024))
    return dict(mean=p.mean_.astype(np.float32),pc=p.components_.astype(np.float32),scale=np.sqrt(p.explained_variance_.clip(p.explained_variance_[0]*.01)).astype(np.float32))

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['context','context_influence'],rounds=200,leaves=7,seed=68068,truncation=1,pca_dimensions=64,pca_seed=79079,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round79*.py')},hypothesis='Quantity gain/loss can mean different things for asked remaining/spent/total. Condition residual correction on supplied semantic context; compare with and without program influence.'))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');source=ROOT/'experiments/round25/selected_bundle';weights=[m['weight'] for m in read(source/'config.json')['members']]
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nested=z['components'];folds=z['folds']
    cache={}
    for split in ('train','validation'):
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:cache[split]=(z['root'],z['extra'])
    fit=folds!=2;held=~fit;nw=space(a,fit);w=space(a,np.ones(len(b),bool));np.savez(OUT/'diagnostic_space.npz',**nw);np.savez(OUT/'full_space.npz',**w)
    nxcontext=project(a,nw);context=project(a,w);vc=project(va,w);g=grounding(b);vs,vg=components(v,va,source);records=[]
    params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True)
    for arm in ('context','context_influence'):
        x,bs=design(b,ss,weights,g,*cache['train'],context,arm);nx,nbs=design(b,nested,weights,g,*cache['train'],nxcontext,arm);vx,vbs=design(v,vs,weights,vg,*cache['validation'],vc,arm)
        ds=lgb.Dataset(nx[fit].reshape(-1,nx.shape[-1]),label=y[fit].ravel(),group=np.full(fit.sum(),4),init_score=nbs[fit].ravel());hm=lgb.train(params,ds,num_boost_round=200)
        hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(nx[held].reshape(-1,nx.shape[-1])).reshape(-1,4),'parse_and_repair',4)
        dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True,pca_fit_ids=ids[fit].tolist()))
        ds=lgb.Dataset(x.reshape(-1,x.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel());m=lgb.train(params,ds,num_boost_round=200)
        c=apply_gate(v,vbs+m.predict(vx.reshape(-1,vx.shape[-1])).reshape(-1,4),'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
        r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()));bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(source,bundle/'baseline');m.save_model(str(bundle/'meta.txt'));shutil.copy2(OUT/'full_space.npz',bundle/'space.npz');dump(bundle/'config.json',dict(arm=arm,weights=weights))
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2;records.append(r);print('VAL',r,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round79/bundle_{best['arm']}"),selected='round79' if best['correct']>753 else 'round75',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
