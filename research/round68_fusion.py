from round22_common import ROOT,load,read,dump,sha
from round68_selector import components,design
from round3_features import grounding
from round3_train import pair_data
from round4_gates import apply_gate
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import numpy as np,lightgbm as lgb,shutil,time
OUT=ROOT/'experiments/round68'
if __name__=='__main__':
    tick=time.perf_counter();assert (OUT/'oof_complete.json').exists();source=ROOT/'experiments/round25/selected_bundle'
    dump(OUT/'fusion_plan.json',dict(arms=['linear','residual_tree'],linear_C=.1,tree_rounds=200,tree_leaves=7,tree_l2=50,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round68*.py')}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');weights=[m['weight'] for m in read(source/'config.json')['members']]
    with np.load(OUT/'oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    x,bs=design(b,ss,weights,grounding(b));vs,vg=components(v,va,source);vx,vbs=design(v,vs,weights,vg)
    bc=apply_gate(v,vbs,'parse_and_repair',4);assert vy[np.arange(len(v)),bc].sum()==752
    mean=x.mean((0,1));scale=x.std((0,1)).clip(.001);xx,yy,ww=pair_data(sparse.csr_matrix(((x-mean)/scale).reshape(-1,x.shape[-1])),y)
    m=LogisticRegression(C=.1,fit_intercept=False,solver='liblinear',max_iter=1000,random_state=68068).fit(xx,yy,sample_weight=ww)
    np.savez(OUT/'linear.npz',mean=mean,scale=scale,weight=m.coef_[0].astype(np.float32));scores={'linear':((vx-mean)/scale)@m.coef_[0]}
    ds=lgb.Dataset(x.reshape(-1,x.shape[-1]),label=y.ravel(),group=np.full(len(x),4),init_score=bs.ravel())
    params=dict(objective='lambdarank',metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True)
    m=lgb.train(params,ds,num_boost_round=200);m.save_model(str(OUT/'residual_tree.txt'));scores['residual_tree']=vbs+m.predict(vx.reshape(-1,x.shape[-1])).reshape(-1,4)
    records=[]
    for arm,s in scores.items():
        c=apply_gate(v,s,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
        r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(source,bundle/'baseline')
        if arm=='linear':shutil.copy2(OUT/'linear.npz',bundle/'meta.npz')
        else:shutil.copy2(OUT/'residual_tree.txt',bundle/'meta.txt')
        dump(bundle/'config.json',dict(arm=arm,weights=weights));r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred)
        records.append(r);print('FUSION',r,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round68/bundle_{best['arm']}"),selected='round68' if best['correct']>752 else 'round25',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'fusion_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
