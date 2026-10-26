"""Matched influence supervision using occurrence-preserving candidate programs."""
from round22_common import ROOT,load,read,dump,sha
from round99_features import build as extract, influence_summary as summarize
from round83_selector import extend
from round82_relations import probabilities
from round80_selector import augment
from round79_selector import design,project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round99'

def relation_scores(split,arm,folds=None,nested=False):
    x=sparse.load_npz(OUT/f'{split}_quantity_features.npz');ptr=np.load(OUT/f'{split}_quantity_ptr.npy');packets=read(OUT/f'{split}_influence_packets.json')
    if split=='validation':
        with np.load(ROOT/f'experiments/round98/{arm}/full.npz') as w:prob=probabilities(x,w)
    else:
        prob=np.zeros((x.shape[0],3))
        for fold in range(3):
            which=f'nested{fold}' if nested and fold!=2 else f'oof{fold}';mask=np.zeros(x.shape[0],bool)
            for i in np.flatnonzero(folds==fold):mask[ptr[i]:ptr[i+1]]=True
            with np.load(ROOT/f'experiments/round98/{arm}/{which}.npz') as w:prob[mask]=probabilities(x[mask],w)
    return summarize(prob,ptr,packets)

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['control','filtered','occurrence'],rounds=200,leaves=7,seed=68068,nested=True,test_opened=False,source_sha256=sha(__file__),candidate_extraction_unchanged=True))
    for src,h in read(ROOT/'experiments/round83/completion_manifest.json')['hashes'].items():assert sha(ROOT/src)==h,src
    assert read(ROOT/'experiments/round98/training_complete.json')['auxiliary_supervision_only']
    b,a,y,ids=load('train');v,va,vy,_=load('validation');weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']];cache={}
    for split in ('train','validation'):
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:cache[split]=(z['root'],z['extra'])
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nested=z['components'];folds=z['folds']
    for split,rows in [('train',b),('validation',v)]:
        root,extra,flow,pairs,ptr,packets,qx,qp,inf=extract(rows,lambda i:print('FEATURES',split,i,flush=True))
        assert np.array_equal(root,cache[split][0]);assert np.array_equal(extra,cache[split][1]);assert (flow-sparse.load_npz(ROOT/f'experiments/round80/{split}_unsigned.npz')).nnz==0
        assert packets==read(ROOT/f'experiments/round83/{split}_packets.json')
        sparse.save_npz(OUT/f'{split}_quantity_features.npz',qx);np.save(OUT/f'{split}_quantity_ptr.npy',qp);dump(OUT/f'{split}_influence_packets.json',inf)
    evidence={}
    for arm in ('filtered','occurrence'):
        evidence[arm]=(relation_scores('train',arm,folds),relation_scores('train',arm,folds,True),relation_scores('validation',arm));np.savez(OUT/f'{arm}_influence_scores.npz',oof=evidence[arm][0],nested=evidence[arm][1],validation=evidence[arm][2])
    with np.load(ROOT/'experiments/round79/full_space.npz') as w:context=project(a,w);vc=project(va,w)
    with np.load(ROOT/'experiments/round79/diagnostic_space.npz') as w:nc=project(a,w)
    g=grounding(b);vs,vg=components(v,va,ROOT/'experiments/round25/selected_bundle');x,bs=design(b,ss,weights,g,*cache['train'],context,'context_influence');nx,nbs=design(b,nested,weights,g,*cache['train'],nc,'context_influence');vx,vbs=design(v,vs,weights,vg,*cache['validation'],vc,'context_influence')
    with np.load(ROOT/'experiments/round80/unsigned_scores.npz') as z:x=augment(x,z['oof']);nx=augment(nx,z['nested']);vx=augment(vx,z['validation'])
    with np.load(ROOT/'experiments/round83/relation_scores.npz') as z:old=(z['oof'],z['nested'],z['validation'])
    params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True);records=[];fit=folds!=2;held=~fit
    for arm in ('control','filtered','occurrence'):
        features=old if arm=='control' else tuple(np.concatenate((a,b),2) for a,b in zip(old,evidence[arm]))
        ax=extend(x,features[0]);an=extend(nx,features[1]);av=extend(vx,features[2]);ds=lgb.Dataset(an[fit].reshape(-1,an.shape[-1]),label=y[fit].ravel(),group=np.full(fit.sum(),4),init_score=nbs[fit].ravel());hm=lgb.train(params,ds,num_boost_round=200);hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(an[held].reshape(-1,an.shape[-1])).reshape(-1,4),'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True))
        ds=lgb.Dataset(ax.reshape(-1,ax.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel());m=lgb.train(params,ds,num_boost_round=200);c=apply_gate(v,vbs+m.predict(av.reshape(-1,av.shape[-1])).reshape(-1,4),'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
        if arm=='control':assert r['correct']==758;assert c.tolist()==read(ROOT/'experiments/round83/validation_relations_predictions.json')['selected']
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');m.save_model(str(bundle/'meta.txt'));shutil.copy2(ROOT/'experiments/round79/full_space.npz',bundle/'space.npz');shutil.copy2(ROOT/'experiments/round80/unsigned/full.npz',bundle/'flow.npz');shutil.copy2(ROOT/'experiments/round82/full.npz',bundle/'relation.npz');dump(bundle/'config.json',dict(arm='relations',weights=weights,influence_arm=arm))
        if arm!='control':shutil.copy2(ROOT/f'experiments/round98/{arm}/full.npz',bundle/'influence.npz')
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2;records.append(r);print('VAL',r,flush=True)
    dump(OUT/'validation_records.json',records);new=max(records[1:],key=lambda r:r['correct']);dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=new['arm'],correct=new['correct'],bundle='experiments/round99/bundle_'+new['arm']),selected='round99' if new['correct']>758 else 'round83',test_target_fixed_before_test=True,test_used_for_selection=False));dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
