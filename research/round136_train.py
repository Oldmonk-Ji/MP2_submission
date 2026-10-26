"""Matched candidate selector: only ambiguity-preserving reference relation supervision replaces old labels."""
from round22_common import ROOT,load,read,dump,sha
from round83_features import summarize
from round83_selector import extend
from round136_inference import probabilities,question_context
from round80_selector import augment
from round79_selector import design,project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round136'

def relation_scores(split,arrays,folds=None,nested=False):
    x=sparse.load_npz(ROOT/f'experiments/round83/{split}_pair_features.npz');ptr=np.load(ROOT/f'experiments/round83/{split}_pair_ptr.npy');packets=read(ROOT/f'experiments/round83/{split}_packets.json')
    context=question_context(arrays);owners=np.repeat(np.arange(len(ptr)-1),np.diff(ptr))
    if split=='validation':
        with np.load(ROOT/'experiments/round135/full.npz') as w:prob=probabilities(x,context,owners,w)
    else:
        prob=np.zeros((x.shape[0],6))
        for fold in range(3):
            which=f'nested{fold}' if nested and fold!=2 else f'oof{fold}';mask=np.zeros(x.shape[0],bool)
            for i in np.flatnonzero(folds==fold):mask[ptr[i]:ptr[i+1]]=True
            with np.load(ROOT/f'experiments/round135/{which}.npz') as w:prob[mask]=probabilities(x[mask],context,owners[mask],w)
    return summarize(prob,ptr,packets)

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['control','interaction'],rounds=200,leaves=7,seed=68068,nested=True,test_opened=False,source_sha256=sha(__file__),candidate_extraction_unchanged=True))
    for src,h in read(ROOT/'experiments/round83/completion_manifest.json')['hashes'].items():assert sha(ROOT/src)==h,src
    assert read(ROOT/'experiments/round135/training_complete.json')['auxiliary_supervision_only']
    b,a,y,ids=load('train');v,va,vy,_=load('validation');weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']];cache={}
    for split in ('train','validation'):
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:cache[split]=(z['root'],z['extra'])
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nested=z['components'];folds=z['folds']
    rel=relation_scores('train',a,folds);nrel=relation_scores('train',a,folds,True);vrel=relation_scores('validation',va);np.savez(OUT/'relation_scores.npz',oof=rel,nested=nrel,validation=vrel)
    with np.load(ROOT/'experiments/round79/full_space.npz') as w:context=project(a,w);vc=project(va,w)
    with np.load(ROOT/'experiments/round79/diagnostic_space.npz') as w:nc=project(a,w)
    g=grounding(b);vs,vg=components(v,va,ROOT/'experiments/round25/selected_bundle');x,bs=design(b,ss,weights,g,*cache['train'],context,'context_influence');nx,nbs=design(b,nested,weights,g,*cache['train'],nc,'context_influence');vx,vbs=design(v,vs,weights,vg,*cache['validation'],vc,'context_influence')
    with np.load(ROOT/'experiments/round80/unsigned_scores.npz') as z:x=augment(x,z['oof']);nx=augment(nx,z['nested']);vx=augment(vx,z['validation'])
    with np.load(ROOT/'experiments/round83/relation_scores.npz') as z:old=(z['oof'],z['nested'],z['validation'])
    params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True);records=[];fit=folds!=2;held=~fit
    for arm,features in [('control',old),('interaction',(rel,nrel,vrel))]:
        ax=extend(x,features[0]);an=extend(nx,features[1]);av=extend(vx,features[2])
        with np.load(ROOT/'experiments/round103/local_scores.npz') as z:ax=augment(ax,z['oof']);an=augment(an,z['nested']);av=augment(av,z['validation'])
        ds=lgb.Dataset(an[fit].reshape(-1,an.shape[-1]),label=y[fit].ravel(),group=np.full(fit.sum(),4),init_score=nbs[fit].ravel());hm=lgb.train(params,ds,num_boost_round=200);hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(an[held].reshape(-1,an.shape[-1])).reshape(-1,4),'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True))
        ds=lgb.Dataset(ax.reshape(-1,ax.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel());m=lgb.train(params,ds,num_boost_round=200);c=apply_gate(v,vbs+m.predict(av.reshape(-1,av.shape[-1])).reshape(-1,4),'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
        if arm=='control':assert r['correct']==759;assert c.tolist()==read(ROOT/'experiments/round104/validation_local_predictions.json')['selected']
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');m.save_model(str(bundle/'meta.txt'));shutil.copy2(ROOT/'experiments/round79/full_space.npz',bundle/'space.npz');shutil.copy2(ROOT/'experiments/round80/unsigned/full.npz',bundle/'flow.npz');shutil.copy2(ROOT/f"experiments/round{82 if arm=='control' else 135}/full.npz",bundle/'relation.npz');dump(bundle/'config.json',dict(arm='relations',weights=weights,structure_arm='local'))
        shutil.copy2(ROOT/'experiments/round103/local/full.npz',bundle/'structure.npz')
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2;records.append(r);print('VAL',r,flush=True)
    dump(OUT/'validation_records.json',records);new=records[1];dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm='interaction',correct=new['correct'],bundle='experiments/round136/bundle_interaction'),selected='round136' if new['correct']>759 else 'round104',test_target_fixed_before_test=True,test_used_for_selection=False));dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
