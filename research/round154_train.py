"""Matched markup extraction repair; all auxiliary models and selector settings fixed."""
from round22_common import ROOT,load,read,dump,sha
from round153_features import summarize
from round82_relations import probabilities
from round83_selector import extend
from round80_selector import augment
from round79_selector import design,project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round154_corrected'

def scores(split,folds=None,ids=None,nested=False):
    base=ROOT/'experiments/round153';flow=sparse.load_npz(base/f'{split}_flow.npz');structure=sparse.load_npz(base/f'{split}_structure.npz');pair=sparse.load_npz(base/f'{split}_pairs.npz');ptr=np.load(base/f'{split}_ptr.npy');packets=read(base/f'{split}_packets.json');n=len(ptr)-1
    fs=np.zeros(n*4,dtype=np.float32 if split=='validation' else np.float64);ss=np.zeros_like(fs);prob=np.zeros((pair.shape[0],6),dtype=np.float32);owner=np.repeat(np.arange(n),np.diff(ptr))
    for fold in (range(3) if split=='train' else [None]):
        stage=('nested'+str(fold) if nested and fold!=2 else 'oof'+str(fold)) if fold is not None else 'full'
        mask=folds==fold if fold is not None else np.ones(n,bool);cmask=np.repeat(mask,4);pmask=mask[owner]
        for source,x,out,rows,key in [('round80/unsigned',flow,fs,cmask,'training_ids'),('round103/local',structure,ss,cmask,'training_ids'),('round82',pair,prob,pmask,'training_parent_ids')]:
            path=ROOT/f'experiments/{source}/{stage}.npz'
            if fold is not None:assert not set(ids[mask])&set(read(path.with_suffix('.json'))[key])
            with np.load(path) as w:out[rows]=probabilities(x[rows],w) if source=='round82' else np.asarray(x[rows]@w['weight']).ravel()
    return fs.reshape(n,4),summarize(prob,ptr,packets),ss.reshape(n,4)

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();assert (ROOT/'experiments/round153/preparation_complete.json').exists()
    dump(OUT/'plan.json',dict(arms=['control','markup'],rounds=200,leaves=7,auxiliary_models='Frozen80,82,103, with original strict OOF and nested stages',source_sha256=sha(__file__),test_opened=False))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']]
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(ids,z['ids']);base=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(ids,z['ids']);nested=z['components'];folds=z['folds']
    with np.load(ROOT/'experiments/round79/full_space.npz') as w:context=project(a,w);vc=project(va,w)
    with np.load(ROOT/'experiments/round79/diagnostic_space.npz') as w:nc=project(a,w)
    g=grounding(b);vs,vg=components(v,va,ROOT/'experiments/round25/selected_bundle');fit=folds!=2;held=~fit;records=[]
    params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True)
    for arm in ('control','markup'):
        directory='round78' if arm=='control' else 'round153'
        with np.load(ROOT/f'experiments/{directory}/train_features.npz') as z:r,e=z['root'],z['extra']
        with np.load(ROOT/f'experiments/{directory}/validation_features.npz') as z:vr,ve=z['root'],z['extra']
        x,bs=design(b,base,weights,g,r,e,context,'context_influence');nx,nbs=design(b,nested,weights,g,r,e,nc,'context_influence');vx,vbs=design(v,vs,weights,vg,vr,ve,vc,'context_influence')
        if arm=='control':
            with np.load(ROOT/'experiments/round80/unsigned_scores.npz') as z:fl=[z[k] for k in ('oof','nested','validation')]
            with np.load(ROOT/'experiments/round83/relation_scores.npz') as z:rel=[z[k] for k in ('oof','nested','validation')]
            with np.load(ROOT/'experiments/round103/local_scores.npz') as z:st=[z[k] for k in ('oof','nested','validation')]
        else:
            allscores=[scores('train',folds,ids),scores('train',folds,ids,True),scores('validation')];fl,rel,st=zip(*allscores)
            for name,values in [('flow',fl),('relation',rel),('structure',st)]:np.savez(OUT/f'{name}_scores.npz',**dict(zip(('oof','nested','validation'),values)))
        x,nx,vx=[augment(extend(augment(xx,f),rr),s) for xx,f,rr,s in zip((x,nx,vx),fl,rel,st)]
        ds=lgb.Dataset(nx[fit].reshape(-1,nx.shape[-1]),label=y[fit].ravel(),group=np.full(fit.sum(),4),init_score=nbs[fit].ravel());hm=lgb.train(params,ds,num_boost_round=200);hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(nx[held].reshape(-1,nx.shape[-1])).reshape(-1,4),'parse_and_repair',4)
        dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True))
        ds=lgb.Dataset(x.reshape(-1,x.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel());m=lgb.train(params,ds,num_boost_round=200);c=apply_gate(v,vbs+m.predict(vx.reshape(-1,vx.shape[-1])).reshape(-1,4),'parse_and_repair',4)
        control=read(ROOT/'experiments/round104/validation_local_predictions.json')['selected'];ok=vy[np.arange(len(v)),c];co=vy[np.arange(len(v)),control]
        rec=dict(arm=arm,correct=int(ok.sum()),repairs_vs104=int(((ok==1)&(co==0)).sum()),harms_vs104=int(((ok==0)&(co==1)).sum()),feature_dimension=int(x.shape[-1]))
        if arm=='control':assert c.tolist()==control
        bundle=OUT/f'bundle_{arm}';shutil.copytree(ROOT/'experiments/round104/bundle_local',bundle);m.save_model(str(bundle/'meta.txt'))
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);rec['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert rec['asset_bytes']<=16*1024**2;records.append(rec);print('VAL',rec,flush=True)
    dump(OUT/'validation_records.json',records);dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm='markup',correct=records[1]['correct'],bundle='experiments/round154_corrected/bundle_markup'),selected='round154' if records[1]['correct']>759 else 'round104',test_target_fixed_before_test=True,test_used_for_selection=False));dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False,training_ids=ids.tolist()))
