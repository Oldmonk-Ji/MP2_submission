from round22_common import ROOT,load,read,dump,sha
from round68_selector import components
from round69_selector import design
from round9_group_features import group_labels,candidate_scores
from round3_features import grounding
from round4_gates import apply_gate
import numpy as np,lightgbm as lgb,shutil,time
OUT=ROOT/'experiments/round69'
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();source=ROOT/'experiments/round25/selected_bundle'
    for p,h in read(ROOT/'experiments/round68/completion_manifest.json')['hashes'].items():assert sha(ROOT/p)==h,p
    dump(OUT/'plan.json',dict(arms=['groups','diversity'],rounds=200,leaves=7,l2=50,seed=69069,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round69*.py')}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');weights=[m['weight'] for m in read(source/'config.json')['members']]
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    g=grounding(b);vs,vg=components(v,va,source);records=[]
    for diversity in (False,True):
        arm='diversity' if diversity else 'groups';x,base,mapping,sizes,bs=design(b,a,ss,weights,g,diversity);target=group_labels(y,mapping)
        assert len(sizes)==4743 and (sizes>0).all()
        vx,vbase,vm,_,vbs=design(v,va,vs,weights,vg,diversity)
        ds=lgb.Dataset(x,label=target,group=sizes,init_score=base)
        params=dict(objective='lambdarank',metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=69069,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True)
        model=lgb.train(params,ds,num_boost_round=200)
        s=candidate_scores(vbase+model.predict(vx),vm,vbs,v);c=apply_gate(v,s,'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4)
        ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
        rec=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()),training_questions=len(sizes),training_answer_groups=len(x))
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(source,bundle/'baseline');model.save_model(str(bundle/'meta.txt'));dump(bundle/'config.json',dict(diversity=diversity,weights=weights))
        rec['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert rec['asset_bytes']<=16*1024**2
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred)
        records.append(rec);print(rec,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round69/bundle_{best['arm']}"),selected='round69' if best['correct']>752 else 'round25',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
