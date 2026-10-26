"""Match ranking training's action set to the unchanged numeric-parser gate."""
from round22_common import ROOT,load,read,dump,sha
from round83_selector import extend
from round80_selector import augment
from round79_selector import design,project
from round68_selector import components
from round3_features import grounding
from features import metadata
from round4_gates import apply_gate
from threadpoolctl import threadpool_limits
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round89'

def dataset(x,y,bs,mask):
    assert mask.any(1).all();assert not ((y>.5)&~mask).any()
    return lgb.Dataset(x[mask],label=y[mask],group=mask.sum(1),init_score=bs[mask])

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['control','valid_actions'],rounds=200,leaves=7,seed=68068,nested=True,validation_incumbent=758,test_opened=False,source_sha256=sha(__file__),fit_audit_sha256=sha(ROOT/'experiments/train_fit_eligible_action_audit_20261026.json'),hypothesis='Remove ranking comparisons against parser-ineligible actions, retain every eligible TRAIN question and all correct actions; no inference gate change.'))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']];valid=metadata(b)['valid'];assert not ((y>.5)&~valid).any();cache={}
    for split in ('train','validation'):
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:cache[split]=(z['root'],z['extra'])
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nested=z['components'];folds=z['folds']
    with np.load(ROOT/'experiments/round79/full_space.npz') as w:ctx=project(a,w);vc=project(va,w)
    with np.load(ROOT/'experiments/round79/diagnostic_space.npz') as w:nc=project(a,w)
    g=grounding(b);vs,vg=components(v,va,ROOT/'experiments/round25/selected_bundle');x,bs=design(b,ss,weights,g,*cache['train'],ctx,'context_influence');nx,nbs=design(b,nested,weights,g,*cache['train'],nc,'context_influence');vx,vbs=design(v,vs,weights,vg,*cache['validation'],vc,'context_influence')
    with np.load(ROOT/'experiments/round80/unsigned_scores.npz') as z:x=augment(x,z['oof']);nx=augment(nx,z['nested']);vx=augment(vx,z['validation'])
    with np.load(ROOT/'experiments/round83/relation_scores.npz') as z:x=extend(x,z['oof']);nx=extend(nx,z['nested']);vx=extend(vx,z['validation'])
    params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True);records=[];fit=folds!=2;held=~fit
    for arm in ('control','valid_actions'):
        mask=np.ones_like(valid) if arm=='control' else valid;hm=lgb.train(params,dataset(nx[fit],y[fit],nbs[fit],mask[fit]),num_boost_round=200);hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(nx[held].reshape(-1,nx.shape[-1])).reshape(-1,4),'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True))
        m=lgb.train(params,dataset(x,y,bs,mask),num_boost_round=200);c=apply_gate(v,vbs+m.predict(vx.reshape(-1,vx.shape[-1])).reshape(-1,4),'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()),training_questions=len(b),training_candidate_actions=int(mask.sum()))
        if arm=='control':assert r['correct']==758;assert c.tolist()==read(ROOT/'experiments/round83/validation_relations_predictions.json')['selected']
        bundle=OUT/f'bundle_{arm}';shutil.copytree(ROOT/'experiments/round83/bundle_relations',bundle);m.save_model(str(bundle/'meta.txt'));pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2;records.append(r);print('VAL',r,flush=True)
    dump(OUT/'validation_records.json',records);new=records[1];dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm='valid_actions',correct=new['correct'],bundle='experiments/round89/bundle_valid_actions'),selected='round89' if new['correct']>758 else 'round83',test_target_fixed_before_test=True,test_used_for_selection=False));dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
