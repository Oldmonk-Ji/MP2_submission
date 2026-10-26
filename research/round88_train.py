from round22_common import ROOT,load,read,dump,sha
from round83_features import summarize
from round83_selector import extend
from round87_model import local_matrix,context,probabilities
from round80_selector import augment
from round79_selector import design,project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round88'

def predict_relations(x,q,ptr,packets,arm,folds=None,nested=False):
    if folds is None:
        with np.load(ROOT/f'experiments/round87/{arm}/full.npz') as w:prob=probabilities(x,q,w)
    else:
        prob=np.zeros((x.shape[0],6))
        for fold in range(3):
            which=f'nested{fold}' if nested and fold!=2 else f'oof{fold}';mask=np.zeros(x.shape[0],bool)
            for i in np.flatnonzero(folds==fold):mask[ptr[i]:ptr[i+1]]=True
            with np.load(ROOT/f'experiments/round87/{arm}/{which}.npz') as w:prob[mask]=probabilities(x[mask],q[mask],w)
    return summarize(prob,ptr,packets)

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();assert read(ROOT/'experiments/round87/training_complete.json')['auxiliary_supervision_only'];dump(OUT/'plan.json',dict(arms=['local','context'],rounds=200,leaves=7,seed=68068,nested=True,validation_incumbent=758,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round88*.py')},unchanged_programs='round83 original read-once packets; no new inference algebraic rewriting'))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']];cache={};pair_cache={}
    for split,arrays in [('train',a),('validation',va)]:
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:cache[split]=(z['root'],z['extra'])
        pairs=local_matrix(sparse.load_npz(ROOT/f'experiments/round83/{split}_pair_features.npz'));ptr=np.load(ROOT/f'experiments/round83/{split}_pair_ptr.npy');packets=read(ROOT/f'experiments/round83/{split}_packets.json');q=np.repeat(context(arrays),np.diff(ptr),axis=0);pair_cache[split]=(pairs,q,ptr,packets)
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nested=z['components'];folds=z['folds']
    with np.load(ROOT/'experiments/round79/full_space.npz') as w:ctx=project(a,w);vc=project(va,w)
    with np.load(ROOT/'experiments/round79/diagnostic_space.npz') as w:nc=project(a,w)
    g=grounding(b);vs,vg=components(v,va,ROOT/'experiments/round25/selected_bundle');x,bs=design(b,ss,weights,g,*cache['train'],ctx,'context_influence');nx,nbs=design(b,nested,weights,g,*cache['train'],nc,'context_influence');vx,vbs=design(v,vs,weights,vg,*cache['validation'],vc,'context_influence')
    with np.load(ROOT/'experiments/round80/unsigned_scores.npz') as z:x=augment(x,z['oof']);nx=augment(nx,z['nested']);vx=augment(vx,z['validation'])
    params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True);records=[];fit=folds!=2;held=~fit
    for arm in ('local','context'):
        rel=predict_relations(*pair_cache['train'],arm,folds);nrel=predict_relations(*pair_cache['train'],arm,folds,True);vrel=predict_relations(*pair_cache['validation'],arm);np.savez(OUT/f'{arm}_relation_scores.npz',oof=rel,nested=nrel,validation=vrel)
        ax=extend(x,rel);an=extend(nx,nrel);av=extend(vx,vrel);ds=lgb.Dataset(an[fit].reshape(-1,an.shape[-1]),label=y[fit].ravel(),group=np.full(fit.sum(),4),init_score=nbs[fit].ravel());hm=lgb.train(params,ds,num_boost_round=200);hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(an[held].reshape(-1,an.shape[-1])).reshape(-1,4),'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True))
        ds=lgb.Dataset(ax.reshape(-1,ax.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel());m=lgb.train(params,ds,num_boost_round=200);c=apply_gate(v,vbs+m.predict(av.reshape(-1,av.shape[-1])).reshape(-1,4),'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');m.save_model(str(bundle/'meta.txt'));shutil.copy2(ROOT/'experiments/round79/full_space.npz',bundle/'space.npz');shutil.copy2(ROOT/'experiments/round80/unsigned/full.npz',bundle/'flow.npz');shutil.copy2(ROOT/f'experiments/round87/{arm}/full.npz',bundle/'relation.npz');dump(bundle/'config.json',dict(arm=arm,weights=weights));pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2;records.append(r);print('VAL',r,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']));dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round88/bundle_{best['arm']}"),selected='round88' if best['correct']>758 else 'round83',test_target_fixed_before_test=True,test_used_for_selection=False));dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
