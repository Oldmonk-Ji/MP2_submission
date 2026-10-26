from round22_common import ROOT,load,read,dump,sha
from round68_selector import components
from round73_selector import augmented
from round73_features import extract
from round3_features import grounding
from round4_gates import apply_gate
import numpy as np,lightgbm as lgb,shutil,time,json
OUT=ROOT/'experiments/round73'
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['control','root'],rounds=200,leaves=7,l2=50,seed=68068,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round73*.py')}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');source=ROOT/'experiments/round25/selected_bundle';weights=[m['weight'] for m in read(source/'config.json')['members']]
    root={}
    for split,rows in [('train',b),('validation',v)]:
        fs=[]
        with (OUT/f'{split}_root_usage.jsonl').open('w',encoding='utf-8') as fh:
            for i,r in enumerate(rows):
                f,u=extract(r);fs.append(f);fh.write(json.dumps(dict(index=int(ids[i]) if split=='train' else i,usage=u))+'\n')
                if i%500==0:print('ROOT FEATURES',split,i,'seconds',time.perf_counter()-tick,flush=True)
        root[split]=np.asarray(fs);np.save(OUT/f'{split}_root_features.npy',root[split])
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    g=grounding(b);vs,vg=components(v,va,source);records=[]
    for arm in ('control','root'):
        x,bs=augmented(b,ss,weights,g,root['train'],arm);vx,vbs=augmented(v,vs,weights,vg,root['validation'],arm)
        ds=lgb.Dataset(x.reshape(-1,x.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel())
        params=dict(objective='lambdarank',metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True)
        m=lgb.train(params,ds,num_boost_round=200);s=vbs+m.predict(vx.reshape(-1,vx.shape[-1])).reshape(-1,4);c=apply_gate(v,s,'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4)
        ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
        if arm=='control':assert r['correct']==749
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(source,bundle/'baseline');m.save_model(str(bundle/'meta.txt'));dump(bundle/'config.json',dict(arm=arm,weights=weights))
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred)
        r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2
        records.append(r);print('VAL',r,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round73/bundle_{best['arm']}"),selected='round73' if best['correct']>752 else 'round25',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
