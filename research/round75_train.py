"""Matched ranking-focus ablation, not direct optimization of TEST accuracy."""
from round22_common import ROOT,load,read,dump,sha
from round68_selector import components
from round73_selector import augmented
from round3_features import grounding
from round4_gates import apply_gate
import numpy as np,lightgbm as lgb,shutil,time
OUT=ROOT/'experiments/round75'
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(features=['control','root'],truncation=[30,1],rounds=200,leaves=7,l2=50,seed=68068,documentation='https://lightgbm.readthedocs.io/en/v4.5.0/Parameters.html#lambdarank_truncation_level',qualification='Truncation changes top-rank focus; it is still a ranking surrogate, not exact accuracy optimization.',test_opened=False,source_hash=sha(__file__)))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');source=ROOT/'experiments/round25/selected_bundle';weights=[m['weight'] for m in read(source/'config.json')['members']]
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    g=grounding(b);vs,vg=components(v,va,source);root=np.load(ROOT/'experiments/round73/train_root_features.npy');vroot=np.load(ROOT/'experiments/round73/validation_root_features.npy');records=[]
    for feature in ('control','root'):
        x,bs=augmented(b,ss,weights,g,root,feature);vx,vbs=augmented(v,vs,weights,vg,vroot,feature)
        for truncation in (30,1):
            arm=f'{feature}_top{truncation}';ds=lgb.Dataset(x.reshape(-1,x.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel())
            params=dict(objective='lambdarank',lambdarank_truncation_level=truncation,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True)
            m=lgb.train(params,ds,num_boost_round=200);s=vbs+m.predict(vx.reshape(-1,vx.shape[-1])).reshape(-1,4);c=apply_gate(v,s,'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4)
            ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
            if truncation==30:assert r['correct']==(749 if feature=='control' else 751)
            bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(source,bundle/'baseline');m.save_model(str(bundle/'meta.txt'));dump(bundle/'config.json',dict(arm=feature,weights=weights))
            pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred)
            r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2;records.append(r);print(r,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round75/bundle_{best['arm']}"),selected='round75' if best['correct']>752 else 'round25',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
