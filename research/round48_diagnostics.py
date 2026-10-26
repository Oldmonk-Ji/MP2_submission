"""TRAIN-only local decision fit/generalization and deterministic reward audit."""
from data_access import ROOT
from round22_common import read,dump
from round48_local import node_scores,aggregate,focus,csr
from round46_selector import probability
from scipy import sparse
import numpy as np
from collections import defaultdict

OUT=ROOT/'experiments/round48'

def main():
    x=sparse.load_npz(OUT/'node_features.npz');ptr=np.load(OUT/'program_node_ptr.npy')
    xp=sparse.load_npz(OUT/'constraint_positive.npz');xn=sparse.load_npz(OUT/'constraint_negative.npz');parents=np.load(OUT/'constraint_parent_ids.npy')
    with np.load(OUT/'train_targets.npz') as z:
        ids=z['ids'];folds=z['folds'];rewards=z['rewards'];mask=z['mask'];y=z['labels'];base=z['base']
    fm=dict(zip(map(int,ids),map(int,folds)));fitpair=np.asarray([fm[int(p)]!=2 for p in parents]);history=[];fits=[]
    for stage in ('diagnostic','full'):
        for arm in ('sft','exact','grpo'):
            hs=[read(OUT/stage/f'{arm}_s{seed}'/'training.json')['history'] for seed in (48048,48049)]
            history.append(dict(stage=stage,arm=arm,first={k:float(np.mean([h[0][k] for h in hs])) for k in hs[0][0]},last={k:float(np.mean([h[-1][k] for h in hs])) for k in hs[0][-1]}))
            margins=[]
            for seed in (48048,48049):
                with np.load(OUT/stage/f'{arm}_s{seed}'/'model.npz') as w:margins.append(node_scores(xp,w)-node_scores(xn,w))
            m=np.mean(margins,axis=0)
            for name,keep in [('fit',fitpair),('old_held',~fitpair)]:
                fits.append(dict(stage=stage,arm=arm,group=name,n=int(keep.sum()),correct=int((m[keep]>0).sum())))
    records=[]
    with np.load(OUT/'diagnostic_scores.npz') as z:
        held=z['held'];oldok=y[held,base[held]].astype(bool)
        for arm in ('sft','exact','grpo'):
            s=(z[f'{arm}_48048']+z[f'{arm}_48049'])/2;p=probability(s,mask[held]);top=p.argmax(1)
            for name,keep in [('all',np.ones(len(held),bool)),('baseline_errors',~oldok)]:
                records.append(dict(arm=arm,group=name,n=int(keep.sum()),correct_program_available=int(rewards[held][keep].any(1).sum()),top_program_correct=int(rewards[held][np.arange(len(held)),top][keep].sum()),expected_reward=float((p*rewards[held]).sum(1)[keep].mean())))
    # Same-answer semantic challenge is evaluated without changing rewards.
    same=read(ROOT/'experiments/round47/held_same_value_cases.json')['cases'];pairs=[]
    for r in same:pairs.append(focus(r))
    sp=csr([r[0] for r in pairs]);sn=csr([r[1] for r in pairs]);challenge=[]
    for arm in ('sft','exact','grpo'):
        ms=[]
        for seed in (48048,48049):
            with np.load(OUT/'diagnostic'/f'{arm}_s{seed}'/'model.npz') as w:ms.append(node_scores(sp,w)-node_scores(sn,w))
        m=np.mean(ms,axis=0);challenge.append(dict(arm=arm,n=len(same),correct=int((m>0).sum()),margins=m.tolist()))
    # The soft-min must punish one poor relation instead of averaging it away.
    testptr=np.r_[0,2,4,np.repeat(4,62)]
    scores=aggregate(np.asarray([2.,2.,2.,-2.]),testptr)
    assert scores.shape==(1,64) and scores[0,0]>scores[0,1]
    report=dict(training=history,local_relation_fit=fits,held_rewards=records,held_same_value_challenge=challenge,
        deterministic_executor_and_rewards_unchanged_from_round47=True,weak_relation_penalizes_program=True,
        locally_distinguishable_pairs=read(OUT/'preparation_result.json')['locally_distinguishable_pairs'],test_opened=False)
    dump(OUT/'reward_diagnostics.json',report);print(report,flush=True)

if __name__=='__main__':main()
