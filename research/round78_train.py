from round22_common import ROOT,load,read,dump,sha
from round78_features import extract
from round78_selector import design
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round78'

if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['control','influence'],rounds=200,leaves=7,seed=68068,truncation=1,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round78*.py')},hypothesis='Conditional signed contribution captures operation scope/direction beyond quantity coverage; dimensional roles are hypotheses, not hard exclusions.'))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');source=ROOT/'experiments/round25/selected_bundle';weights=[m['weight'] for m in read(source/'config.json')['members']]
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nested=z['components'];folds=z['folds']
    cache={}
    for split,rows in [('train',b),('validation',v)]:
        rr=[];ee=[]
        for i,r in enumerate(rows):
            x,e=extract(r);rr.append(x);ee.append(e)
            if i%500==0:print('FEATURES',split,i,flush=True)
        root=np.asarray(rr);extra=np.asarray(ee);assert np.array_equal(root,np.load(ROOT/f'experiments/round73/{split}_root_features.npy'))
        np.savez(OUT/f'{split}_features.npz',root=root,extra=extra);cache[split]=(root,extra)
    g=grounding(b);vs,vg=components(v,va,source);records=[]
    params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True)
    for arm in ('control','influence'):
        x,bs=design(b,ss,weights,g,*cache['train'],arm);nx,nbs=design(b,nested,weights,g,*cache['train'],arm);vx,vbs=design(v,vs,weights,vg,*cache['validation'],arm)
        fit=folds!=2;held=folds==2;ds=lgb.Dataset(nx[fit].reshape(-1,nx.shape[-1]),label=y[fit].ravel(),group=np.full(fit.sum(),4),init_score=nbs[fit].ravel());hm=lgb.train(params,ds,num_boost_round=200)
        hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(nx[held].reshape(-1,nx.shape[-1])).reshape(-1,4),'parse_and_repair',4)
        dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True))
        ds=lgb.Dataset(x.reshape(-1,x.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel());m=lgb.train(params,ds,num_boost_round=200)
        c=apply_gate(v,vbs+m.predict(vx.reshape(-1,vx.shape[-1])).reshape(-1,4),'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
        r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
        if arm=='control':assert r['correct']==753
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(source,bundle/'baseline');m.save_model(str(bundle/'meta.txt'));dump(bundle/'config.json',dict(arm=arm,weights=weights))
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2
        records.append(r);print('VAL',r,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],r['arm']=='control'))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round78/bundle_{best['arm']}"),selected='round78' if best['correct']>753 else 'round75',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
