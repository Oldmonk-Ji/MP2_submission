from round22_common import ROOT,load,read,dump,sha
from round68_selector import components
from round73_selector import augmented
from round74_quantity import matrix,probabilities,weighted
from round44_programs import locations
from round3_features import grounding
from round4_gates import apply_gate
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import numpy as np,lightgbm as lgb,shutil,time,json
OUT=ROOT/'experiments/round74'
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['root_control','learned_relevance'],quantity_C=1,quantity_label_status='weak reference trace usage, not independent semantic gold',meta_rounds=200,meta_leaves=7,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round74*.py')}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');qx,ptr=matrix(b);vqx,vptr=matrix(v)
    parents=np.repeat(ids,np.diff(ptr));target=np.full(len(parents),-1,int);index={int(k):i for i,k in enumerate(ids)}
    for r in read(ROOT/'experiments/round63/accepted_reference_programs.json'):
        i=index[r['train_index']];used={n[1] for _,n in locations(r['program']) if n[0]=='N'};target[ptr[i]:ptr[i+1]]=[int(k in used) for k in range(ptr[i+1]-ptr[i])]
    oof=np.zeros(len(parents));seen=np.zeros(len(parents),int);checks=[]
    for fold in range(3):
        part=read(ROOT/f'experiments/round22/oof/fold{fold}/partition.json');tr=np.isin(parents,part['fit_ids'])&(target>=0);te=np.isin(parents,part['held_ids'])
        m=LogisticRegression(C=1,solver='liblinear',max_iter=1000,random_state=74074).fit(qx[tr],target[tr]);oof[te]=m.predict_proba(qx[te])[:,1];seen[te]+=1
        check=te&(target>=0);checks.append(dict(fold=fold,n=int(check.sum()),correct=int(((oof[check]>=.5)==target[check]).sum()),positive_n=int(target[check].sum())))
    assert (seen==1).all();m=LogisticRegression(C=1,solver='liblinear',max_iter=1000,random_state=74074).fit(qx[target>=0],target[target>=0])
    qw=dict(weight=m.coef_[0].astype(np.float32),bias=m.intercept_.astype(np.float32));np.savez(OUT/'quantity.npz',**qw)
    dump(OUT/'quantity_diagnostic.json',dict(covered_reference_parents=len(set(parents[target>=0])),labeled_quantities=int((target>=0).sum()),folds=checks,interpretation='agreement with reference-derived usage labels, not human semantic accuracy'))
    usage={}
    for split in ('train','validation'):
        with (ROOT/f'experiments/round73/{split}_root_usage.jsonl').open(encoding='utf-8') as f:rows=[json.loads(line) for line in f]
        if split=='train':assert [r['index'] for r in rows]==ids.tolist()
        usage[split]=[r['usage'] for r in rows]
    wf=weighted(usage['train'],[oof[a:b] for a,b in zip(ptr,ptr[1:])]);vwf=weighted(usage['validation'],probabilities(vqx,vptr,qw))
    source=ROOT/'experiments/round25/selected_bundle';weights=[m['weight'] for m in read(source/'config.json')['members']]
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(ids,z['ids']);ss=z['components']
    g=grounding(b);vs,vg=components(v,va,source);root=np.load(ROOT/'experiments/round73/train_root_features.npy');vroot=np.load(ROOT/'experiments/round73/validation_root_features.npy');records=[]
    for relevance in (False,True):
        arm='learned_relevance' if relevance else 'root_control';x,bs=augmented(b,ss,weights,g,root,'root');vx,vbs=augmented(v,vs,weights,vg,vroot,'root')
        if relevance:x=np.concatenate([x,wf,wf-wf.mean(1,keepdims=True)],2);vx=np.concatenate([vx,vwf,vwf-vwf.mean(1,keepdims=True)],2)
        ds=lgb.Dataset(x.reshape(-1,x.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel())
        params=dict(objective='lambdarank',metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True)
        m=lgb.train(params,ds,num_boost_round=200);s=vbs+m.predict(vx.reshape(-1,vx.shape[-1])).reshape(-1,4);c=apply_gate(v,s,'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4)
        ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
        if not relevance:assert r['correct']==751
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(source,bundle/'baseline');m.save_model(str(bundle/'meta.txt'))
        if relevance:shutil.copy2(OUT/'quantity.npz',bundle/'quantity.npz')
        dump(bundle/'config.json',dict(relevance=relevance,weights=weights));pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred)
        r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2
        records.append(r);print(r,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round74/bundle_{best['arm']}"),selected='round74' if best['correct']>752 else 'round25',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),auxiliary_quantity_supervision_partial_only=True,test_opened=False))
