"""Post-experiment TRAIN-only reward/coverage diagnostics; no tuning."""
from data_access import ROOT
from round22_common import read,dump
from round45_rlvr import ARMS,SEEDS
from round44_policy import probability
import numpy as np

OUT=ROOT/'experiments/round45'
def main():
    with np.load(OUT/'diagnostic_scores.npz') as z:pred={k:z[k] for k in z.files}
    held=pred['held']
    with np.load(ROOT/'experiments/round44/train_targets.npz') as z:
        r=z['rewards'][held];mask=z['mask'][held];y=z['labels'][held]
        coverage=dict(training_n=len(z['labels']),all_four_correct=int(z['labels'].all(1).sum()),candidate_mixed_correctness=int((~z['labels'].all(1)).sum()),
            relation_has_correct_action=int(z['rewards'].any(1).sum()),relation_has_no_correct_action=int((~z['rewards'].any(1)).sum()))
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:base=z['choices'][held]
    baseok=y[np.arange(len(held)),base].astype(bool);records=[];histories=[]
    for level in ('candidate','relation'):
        for arm in ['warm']+ARMS[level]:
            target=y if level=='candidate' else r;valid=np.ones_like(y,dtype=bool) if level=='candidate' else mask
            p=sum(probability(pred[f'{level}_{arm}_{seed}'],valid) for seed in SEEDS[level])/2
            exp=(p*target).sum(1);top=target[np.arange(len(held)),p.argmax(1)].astype(bool)
            for name,g in [('all',np.ones(len(held),bool)),('baseline_errors',~baseok)]:
                records.append(dict(level=level,arm=arm,group=name,n=int(g.sum()),oracle=int(target[g].any(1).sum()),
                    expected_verified_reward=float(exp[g].mean()),top_action_correct=int(top[g].sum())))
            if arm=='warm':continue
            for stage in ('diagnostic','full'):
                hh=[read(OUT/f'{stage}/{level}_{arm}_s{seed}/training.json')['history'] for seed in SEEDS[level]]
                histories.append(dict(stage=stage,level=level,arm=arm,first={k:float(np.mean([h[0][k] for h in hh])) for k in hh[0][0]},
                    last={k:float(np.mean([h[-1][k] for h in hh])) for k in hh[0][-1]}))
    result=dict(held_reward=records,rollout_training=histories,training_reward_coverage=coverage,test_opened=False,
        caveat='TRAIN-only grouped held diagnostics; previously explored TRAIN schemas. Relation action space unchanged from round44; no arbitrary new program generation.')
    dump(OUT/'reward_diagnostics.json',result)
    for r in records:
        if r['group']=='baseline_errors':print(r,flush=True)
    for r in histories:
        if r['stage']=='full':print('FULL_ROLLOUT',r,flush=True)

if __name__=='__main__':main()
