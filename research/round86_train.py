from round22_common import ROOT,load,read,dump,sha
from round86_features import build,summarize
from round86_selector import extend
from round82_relations import probabilities
from round80_selector import augment
from round79_selector import design,project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round86'

def relations_for(x,ptr,packets,folds,nested):
    prob=np.zeros((x.shape[0],6))
    for fold in range(3):
        which=f'nested{fold}' if nested and fold!=2 else f'oof{fold}';mask=np.zeros(x.shape[0],bool)
        for i in np.flatnonzero(folds==fold):mask[ptr[i]:ptr[i+1]]=True
        with np.load(ROOT/f'experiments/round85/{which}.npz') as w:prob[mask]=probabilities(x[mask],w)
    return summarize(prob,ptr,packets)

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();
    for src,h in read(ROOT/'experiments/round84/fit_plan.json')['source_hashes'].items():assert sha(ROOT/src)==h,src
    dump(OUT/'plan.json',dict(arms=['control','relations'],rounds=200,leaves=7,seed=68068,nested=True,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round86*.py')},auxiliary='round85 expanded supervision, exact algebraic recovery applied at inference; control retains previous candidate packets'))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');cache={};weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']]
    for split,rows in [('train',b),('validation',v)]:
        root,extra,flow,pairs,ptr,packets=build(rows,lambda i:print('FEATURES',split,i,flush=True))
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:assert np.array_equal(root,z['root']);assert np.array_equal(extra,z['extra'])
        old=sparse.load_npz(ROOT/f'experiments/round80/{split}_unsigned.npz');assert (flow-old).nnz==0
        sparse.save_npz(OUT/f'{split}_pair_features.npz',pairs);np.save(OUT/f'{split}_pair_ptr.npy',ptr);dump(OUT/f'{split}_packets.json',packets);cache[split]=(root,extra,pairs,ptr,packets)
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nested=z['components'];folds=z['folds']
    rel=relations_for(*cache['train'][2:],folds,False);nrel=relations_for(*cache['train'][2:],folds,True)
    with np.load(ROOT/'experiments/round85/full.npz') as w:vp=probabilities(cache['validation'][2],w)
    vrel=summarize(vp,*cache['validation'][3:]);np.savez(OUT/'relation_scores.npz',oof=rel,nested=nrel,validation=vrel)
    with np.load(ROOT/'experiments/round79/full_space.npz') as w:context=project(a,w);vc=project(va,w)
    with np.load(ROOT/'experiments/round79/diagnostic_space.npz') as w:nc=project(a,w)
    g=grounding(b);vs,vg=components(v,va,ROOT/'experiments/round25/selected_bundle');x,bs=design(b,ss,weights,g,*cache['train'][:2],context,'context_influence');nx,nbs=design(b,nested,weights,g,*cache['train'][:2],nc,'context_influence');vx,vbs=design(v,vs,weights,vg,*cache['validation'][:2],vc,'context_influence')
    with np.load(ROOT/'experiments/round80/unsigned_scores.npz') as z:x=augment(x,z['oof']);nx=augment(nx,z['nested']);vx=augment(vx,z['validation'])
    with np.load(ROOT/'experiments/round85_selection/relation_scores.npz') as z:old_rel=(z['oof'],z['nested'],z['validation'])
    params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True);records=[];fit=folds!=2;held=~fit
    for arm in ('control','relations'):
        ax=extend(x,rel) if arm=='relations' else extend(x,old_rel[0]);an=extend(nx,nrel) if arm=='relations' else extend(nx,old_rel[1]);av=extend(vx,vrel) if arm=='relations' else extend(vx,old_rel[2])
        ds=lgb.Dataset(an[fit].reshape(-1,an.shape[-1]),label=y[fit].ravel(),group=np.full(fit.sum(),4),init_score=nbs[fit].ravel());hm=lgb.train(params,ds,num_boost_round=200);hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(an[held].reshape(-1,an.shape[-1])).reshape(-1,4),'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True))
        ds=lgb.Dataset(ax.reshape(-1,ax.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel());m=lgb.train(params,ds,num_boost_round=200);c=apply_gate(v,vbs+m.predict(av.reshape(-1,av.shape[-1])).reshape(-1,4),'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
        if arm=='control':assert r['correct']==755;assert c.tolist()==read(ROOT/'experiments/round85_selection/validation_recovered_predictions.json')['selected']
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');m.save_model(str(bundle/'meta.txt'));shutil.copy2(ROOT/'experiments/round79/full_space.npz',bundle/'space.npz');shutil.copy2(ROOT/'experiments/round80/unsigned/full.npz',bundle/'flow.npz')
        if True:shutil.copy2(ROOT/'experiments/round85/full.npz',bundle/'relation.npz')
        dump(bundle/'config.json',dict(arm='relations',weights=weights));pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2;records.append(r);print('VAL',r,flush=True)
    dump(OUT/'validation_records.json',records);new=records[1];dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm='relations',correct=new['correct'],bundle='experiments/round86/bundle_relations'),selected='round86' if new['correct']>758 else 'round83',test_target_fixed_before_test=True,test_used_for_selection=False));dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
