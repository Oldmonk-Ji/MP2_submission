"""Fixed candidate-selector ablation of symbolic trace consistency."""
from round22_common import ROOT,load,read,dump,sha
from round83_features import summarize
from round83_selector import extend
from round92_consistency import build as consistency
from round80_selector import augment
from round79_selector import design,project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round92_selection'

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['control','unscoped','conserved'],rounds=200,leaves=7,seed=68068,nested=True,test_opened=False,source_sha256=sha(__file__),candidate_extraction_unchanged=True))
    for src,h in read(ROOT/'experiments/round83/completion_manifest.json')['hashes'].items():assert sha(ROOT/src)==h,src
    for src,h in read(ROOT/'experiments/round92/rule_freeze.json')['hashes'].items():assert sha(ROOT/src)==h,src
    b,a,y,ids=load('train');v,va,vy,_=load('validation');weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']];cache={}
    for split in ('train','validation'):
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:cache[split]=(z['root'],z['extra'])
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nested=z['components'];folds=z['folds']
    evidence={}
    for arm in ('unscoped','conserved'):
        evidence[arm]=(consistency(b,arm),consistency(v,arm));np.savez(OUT/f'{arm}_features.npz',train=evidence[arm][0],validation=evidence[arm][1])
    with np.load(ROOT/'experiments/round79/full_space.npz') as w:context=project(a,w);vc=project(va,w)
    with np.load(ROOT/'experiments/round79/diagnostic_space.npz') as w:nc=project(a,w)
    g=grounding(b);vs,vg=components(v,va,ROOT/'experiments/round25/selected_bundle');x,bs=design(b,ss,weights,g,*cache['train'],context,'context_influence');nx,nbs=design(b,nested,weights,g,*cache['train'],nc,'context_influence');vx,vbs=design(v,vs,weights,vg,*cache['validation'],vc,'context_influence')
    with np.load(ROOT/'experiments/round80/unsigned_scores.npz') as z:x=augment(x,z['oof']);nx=augment(nx,z['nested']);vx=augment(vx,z['validation'])
    with np.load(ROOT/'experiments/round83/relation_scores.npz') as z:old=(z['oof'],z['nested'],z['validation'])
    params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True);records=[];fit=folds!=2;held=~fit
    for arm in ('control','unscoped','conserved'):
        features=old if arm=='control' else tuple(np.concatenate((z,evidence[arm][int(i==2)]),2) for i,z in enumerate(old))
        ax=extend(x,features[0]);an=extend(nx,features[1]);av=extend(vx,features[2]);ds=lgb.Dataset(an[fit].reshape(-1,an.shape[-1]),label=y[fit].ravel(),group=np.full(fit.sum(),4),init_score=nbs[fit].ravel());hm=lgb.train(params,ds,num_boost_round=200);hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(an[held].reshape(-1,an.shape[-1])).reshape(-1,4),'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True))
        ds=lgb.Dataset(ax.reshape(-1,ax.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel());m=lgb.train(params,ds,num_boost_round=200);c=apply_gate(v,vbs+m.predict(av.reshape(-1,av.shape[-1])).reshape(-1,4),'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
        if arm=='control':assert r['correct']==758;assert c.tolist()==read(ROOT/'experiments/round83/validation_relations_predictions.json')['selected']
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');m.save_model(str(bundle/'meta.txt'));shutil.copy2(ROOT/'experiments/round79/full_space.npz',bundle/'space.npz');shutil.copy2(ROOT/'experiments/round80/unsigned/full.npz',bundle/'flow.npz');shutil.copy2(ROOT/'experiments/round82/full.npz',bundle/'relation.npz');dump(bundle/'config.json',dict(arm='relations',weights=weights,consistency_arm=arm))
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2;records.append(r);print('VAL',r,flush=True)
    dump(OUT/'validation_records.json',records);new=max(records[1:],key=lambda r:r['correct']);dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=new['arm'],correct=new['correct'],bundle='experiments/round92_selection/bundle_'+new['arm']),selected='round92' if new['correct']>758 else 'round83',test_target_fixed_before_test=True,test_used_for_selection=False));dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
