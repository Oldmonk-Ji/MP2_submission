"""Matched uniform/pooling/learned strict-root decisions in exact104 readout."""
from round22_common import ROOT,load,read,dump,sha
from round83_selector import extend
from round80_selector import augment
from round79_selector import design,project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from threadpoolctl import threadpool_limits
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round199'

if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();OUT.mkdir();assert read(ROOT/'experiments/round198/training_complete.json')['test_opened']==False
    arms=['control','mean','pool_only','latent'];dump(OUT/'plan.json',dict(arms=arms,source_sha256=sha(__file__),method='Exact104 base plus three strict-root score summaries. Mean frozen181, pool-only same weights, latent learned184. Same fixed200-tree7-leaf fusion; strict improvement over759 required before TEST.',test_opened=False))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']];cache={}
    for split in ('train','validation'):
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:cache[split]=(z['root'],z['extra'])
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nested=z['components'];folds=z['folds']
    for fold in range(3):
        for stage in ([f'oof{fold}',f'nested{fold}'] if fold!=2 else ['oof2']):assert not set(ids[folds==fold])&set(read(ROOT/f'experiments/round198/{stage}.json')['training_ids'])
    with np.load(ROOT/'experiments/round79/full_space.npz') as w:context=project(a,w);vc=project(va,w)
    with np.load(ROOT/'experiments/round79/diagnostic_space.npz') as w:nc=project(a,w)
    g=grounding(b);vs,vg=components(v,va,ROOT/'experiments/round25/selected_bundle');x,bs=design(b,ss,weights,g,*cache['train'],context,'context_influence');nx,nbs=design(b,nested,weights,g,*cache['train'],nc,'context_influence');vx,vbs=design(v,vs,weights,vg,*cache['validation'],vc,'context_influence')
    with np.load(ROOT/'experiments/round80/unsigned_scores.npz') as z:x=augment(x,z['oof']);nx=augment(nx,z['nested']);vx=augment(vx,z['validation'])
    with np.load(ROOT/'experiments/round83/relation_scores.npz') as z:x=extend(x,z['oof']);nx=extend(nx,z['nested']);vx=extend(vx,z['validation'])
    with np.load(ROOT/'experiments/round103/local_scores.npz') as z:x=augment(x,z['oof']);nx=augment(nx,z['nested']);vx=augment(vx,z['validation'])
    params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True);records=[];fit=folds!=2;held=~fit
    for arm in arms:
        ax,an,av=x,nx,vx
        if arm!='control':
            path=ROOT/('experiments/round195/scores.npz' if arm=='mean' else f'experiments/round198/{arm}_scores.npz')
            with np.load(path) as z:ax=augment(x,z['oof']);an=augment(nx,z['nested']);av=augment(vx,z['validation'])
        ds=lgb.Dataset(an[fit].reshape(-1,an.shape[-1]),label=y[fit].ravel(),group=np.full(fit.sum(),4),init_score=nbs[fit].ravel());hm=lgb.train(params,ds,num_boost_round=200);hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(an[held].reshape(-1,an.shape[-1])).reshape(-1,4),'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True))
        ds=lgb.Dataset(ax.reshape(-1,ax.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel());m=lgb.train(params,ds,num_boost_round=200);c=apply_gate(v,vbs+m.predict(av.reshape(-1,av.shape[-1])).reshape(-1,4),'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()),feature_dimension=int(ax.shape[-1]))
        if arm=='control':assert r['correct']==759;assert c.tolist()==read(ROOT/'experiments/round104/validation_local_predictions.json')['selected']
        if arm=='mean':assert r['correct']==read(ROOT/'experiments/round196/validation_records.json')[1]['correct'];assert c.tolist()==read(ROOT/'experiments/round196/validation_strict_predictions.json')['selected']
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');m.save_model(str(bundle/'meta.txt'))
        for source,name in [('round79/full_space.npz','space.npz'),('round80/unsigned/full.npz','flow.npz'),('round82/full.npz','relation.npz'),('round103/local/full.npz','structure.npz')]:shutil.copy2(ROOT/'experiments'/source,bundle/name)
        if arm!='control':shutil.copy2(ROOT/('experiments/round198/full.npz' if arm=='latent' else 'experiments/round195/full.npz'),bundle/'strict.npz')
        dump(bundle/'config.json',dict(arm='relations',weights=weights,structure_arm='local',strict_arm=arm));pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred)
        r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2;records.append(r);print('VAL',r,flush=True)
    dump(OUT/'validation_records.json',records);new=max(records[1:],key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']));dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=new['arm'],correct=new['correct'],bundle='experiments/round199/bundle_'+new['arm']),selected='round199' if new['correct']>759 else 'round104',test_target_fixed_before_test=True,test_used_for_selection=False));dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
