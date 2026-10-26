"""Additional train-only confirmation of two weak but positive screening leads.
This does not rewrite the failed round9/10 screening gates. It uses a new,
fixed meta-crossfit protocol, selected before any validation access this turn.
"""
from data_access import ROOT,load
import json,time,hashlib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from round9_group_features import build,group_labels,subset,candidate_scores
from round9_train import train_model,metric
from round10_fusion import arrays,fit,fuse,gated
from features import metadata
from round4_gates import apply_gate
OUT=ROOT/'experiments/round12'

def main():
    OUT.mkdir(exist_ok=True);tick=time.perf_counter()
    # Both candidates fixed from the preceding internal screen, no new search.
    plan=dict(folds=3,seed=20261015,methods=['round9 groups 7 leaves/120 rounds/alpha .25','round10 unnormalized probability mean-group pooling'],why='Each screened +1/951. Too weak to select immediately; confirm on all train questions with fixed 3-fold meta-crossfit.',gate='Net >=3 correct over all 4755 train OOF questions, and at least 2 of 3 folds nonnegative, then one validation comparison only. Otherwise keep baseline.',caveat='Method choice used the earlier overlapping train screen; base OOF and meta folds are not fully nested. This is supporting development evidence, not an unbiased generalization estimate.',test_access=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    bank,_,y=load('train');cache=np.load(ROOT/'experiments/round3/stack_scores.npz');raw=cache['oof_members'];base=cache['meta_oof'];meta=metadata(bank);arith=np.load(ROOT/'experiments/round4/arithmetic_train_features.npz')['features'];cfg=json.loads((ROOT/'experiments/round4/selected_bundle/config.json').read_text())
    baseline=apply_gate(bank,base,cfg['gate_kind'],cfg['margin_limit']);baselineok=y[np.arange(len(y)),baseline]>0
    x,gb,mapping,sizes,_=build(bank,raw,base,arith,'groups');gy=group_labels(y,mapping);z,eq,valid=arrays(bank,raw,base,False)
    folds=list(StratifiedKFold(3,shuffle=True,random_state=20261015).split(np.arange(len(y)),y.sum(1)))
    selected={name:np.zeros(len(y),int) for name in ('group','probability')};records=[]
    for k,(tr,ho) in enumerate(folds):
        xx,bb,yy,gg=subset(x,gb,gy,mapping,sizes,tr);model=train_model(xx,bb,yy,gg,OUT/f'group_fold{k}.txt',7,120);delta=model.predict(x,num_threads=4);s=candidate_scores(gb+.25*delta,mapping,base,bank)
        choice=apply_gate(bank,s,cfg['gate_kind'],cfg['margin_limit']);selected['group'][ho]=choice[ho]
        rec=dict(method='group',fold=k,**metric(choice[ho],y[ho],meta['hard'][ho],baseline[ho]));records.append(rec);print(json.dumps(rec),flush=True)
        params,info=fit(z[tr],eq[tr],valid[tr],y[tr],'probability','mean');np.savez(OUT/f'probability_fold{k}.npz',params=params)
        ps=fuse(z,eq,valid,params,'probability','mean');choice=gated(bank,ps,base);selected['probability'][ho]=choice[ho]
        rec=dict(method='probability',fold=k,**metric(choice[ho],y[ho],meta['hard'][ho],baseline[ho]),optimization=info);records.append(rec);print(json.dumps(rec),flush=True)
    summary={}
    for method,choice in selected.items():
        m=metric(choice,y,meta['hard'],baseline);nonnegative=sum(r['fixed']>=r['lost'] for r in records if r['method']==method)
        m['nonnegative_folds']=nonnegative;m['gate_passed']=m['fixed']-m['lost']>=3 and nonnegative>=2;summary[method]=m
    decision=dict(baseline=metric(baseline,y,meta['hard']),summary=summary,folds=records,validation_read=False,test_access=False,seconds=time.perf_counter()-tick)
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2));np.savez(OUT/'train_meta_oof.npz',baseline=baseline,**selected)
    paths=[ROOT/'research/round12_confirm.py',OUT/'plan.json',OUT/'decision.json']
    (OUT/'frozen.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2));print('DECISION',json.dumps(summary),flush=True)

if __name__=='__main__':main()
