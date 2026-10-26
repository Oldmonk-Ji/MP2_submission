from round22_common import ROOT,load,read,dump,sha
from round70_train import fit
from round70_selector import scores
from round59_selector import features
from round68_selector import components
from round71_selector import augmented
from round3_features import grounding
from round4_gates import apply_gate
from round14_prepare import family
import torch,numpy as np,lightgbm as lgb,shutil,time
OUT=ROOT/'experiments/round71'
if __name__=='__main__':
    OUT.mkdir();torch.set_num_threads(4);tick=time.perf_counter();source=ROOT/'experiments/round25/selected_bundle'
    arms=[('confidence8',False,True),('normalized9',True,False),('confidence9',True,True)]
    dump(OUT/'plan.json',dict(arms=arms,meta_rounds=200,meta_leaves=7,meta_l2=50,meta_seed=68068,attention_recipe='Frozen round70 120epoch seed70070',test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round71*.py')}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');ix={int(k):i for i,k in enumerate(ids)};x,o=features(b,a);oof=np.zeros((len(b),4));seen=np.zeros(len(b),int)
    for fold in range(3):
        part=read(ROOT/f'experiments/round22/oof/fold{fold}/partition.json');tr=np.array([ix[k] for k in part['fit_ids']]);te=np.array([ix[k] for k in part['held_ids']])
        assert not {family(b[i]['question']) for i in tr}&{family(b[i]['question']) for i in te}
        p=OUT/f'fold{fold}'/'attention.npz';fit(x[tr],o[tr],y[tr],ids[tr],p,True);oof[te]=scores(x[te],o[te],p,True);seen[te]+=1
        dump(p.parent/'partition.json',part);print('ATTENTION OOF',fold,'seconds',time.perf_counter()-tick,flush=True)
    assert (seen==1).all();np.savez(OUT/'attention_oof.npz',ids=ids,scores=oof)
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(ids,z['ids']);old=z['components']
    vx,vo=features(v,va);av=scores(vx,vo,ROOT/'experiments/round70/full/attention.npz',True);vs,vg=components(v,va,source);g=grounding(b)
    weights=[m['weight'] for m in read(source/'config.json')['members']];records=[]
    for arm,attention,confidence in arms:
        ss=np.concatenate([old,oof[:,:,None]],2) if attention else old;vv=np.concatenate([vs,av[:,:,None]],2) if attention else vs;ww=weights+[0] if attention else weights
        xx,bs=augmented(b,ss,ww,g,confidence);ex,vbs=augmented(v,vv,ww,vg,confidence)
        ds=lgb.Dataset(xx.reshape(-1,xx.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel())
        params=dict(objective='lambdarank',metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True)
        m=lgb.train(params,ds,num_boost_round=200);s=vbs+m.predict(ex.reshape(-1,ex.shape[-1])).reshape(-1,4);c=apply_gate(v,s,'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4)
        ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(source,bundle/'baseline');m.save_model(str(bundle/'meta.txt'))
        if attention:shutil.copy2(ROOT/'experiments/round70/full/attention.npz',bundle/'attention.npz')
        dump(bundle/'config.json',dict(attention=attention,confidence=confidence,weights=ww));r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);records.append(r);print(r,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round71/bundle_{best['arm']}"),selected='round71' if best['correct']>752 else 'round25',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),oof_question_disjoint=True,test_opened=False))
