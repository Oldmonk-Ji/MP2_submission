"""Post-experiment TRAIN diagnostics only; does not alter any model or rule."""
from data_access import ROOT
from round22_common import read,dump,load
from round44_policy import probability,safe_argmax,forward
from round44_selector import select
from scipy import sparse
from collections import Counter
import numpy as np

def main():
    out=ROOT/'experiments/round44';b,a,y,ids=load('train')
    with np.load(out/'train_targets.npz') as z:mask=z['mask'];matches=z['matches'];rewards=z['rewards']
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:base=z['choices'];bs=z['scores']
    with np.load(out/'diagnostic_scores.npz') as z:pred={k:z[k] for k in z.files}
    held=pred['held'];baseok=y[held,base[held]].astype(bool);records=[]
    for arm in ('sft','reward','reward_relations'):
        s=(pred[f'{arm}_44044']+pred[f'{arm}_44045'])/2
        top=safe_argmax(s,mask[held]);topok=rewards[held,top].astype(bool)
        for group,g in [('all',np.ones(len(held),bool)),('baseline_errors',~baseok),('baseline_correct',baseok)]:
            records.append(dict(arm=arm,group=group,n=int(g.sum()),proposal_n=int(mask[held][g].any(1).sum()),
                correct_program_available=int(rewards[held][g].any(1).sum()),top_program_correct=int(topok[g].sum())))
    pairs=read(out/'constraint_pairs.json');xp=sparse.load_npz(out/'constraint_positive.npz');xn=sparse.load_npz(out/'constraint_negative.npz')
    relation=[]
    for stage in ('diagnostic','full'):
        for arm in ('sft','reward','reward_relations'):
            margin=np.zeros(len(pairs))
            for seed in (44044,44045):
                with np.load(out/f'{stage}/{arm}_s{seed}/model.npz') as z:w={k:z[k] for k in z.files}
                margin+=(forward(xp,w)-forward(xn,w))/2
            heldparents=set(read(out/'diagnostic_partition.json')['held_constraint_parents'])
            for group in ('fit','held') if stage=='diagnostic' else ('all',):
                ix=[j for j,p in enumerate(pairs) if group=='all' or (p['train_index'] in heldparents)==(group=='held')]
                relation.append(dict(stage=stage,arm=arm,group=group,pairs=len(ix),correct=int((margin[ix]>0).sum())))
    annotations=read(ROOT/'experiments/round38/root_cause_annotations.json')['records'];byid={int(i):j for j,i in enumerate(ids)};cases=[]
    for r in annotations:
        k=byid[r['train_index']]
        if k not in set(held.tolist()):continue
        j=int(np.flatnonzero(held==k)[0]);case=dict(train_index=r['train_index'],root_cause=r['root_cause'],oracle=bool(rewards[k].any()),arms={})
        for arm in ('sft','reward','reward_relations'):
            s=(pred[f'{arm}_44044']+pred[f'{arm}_44045'])/2
            for mode in ('execute_top','fixed_residual'):
                c=select([b[k]],bs[k:k+1],s[j:j+1],mask[k:k+1],matches[k:k+1],mode)[0]
                case['arms'][arm+'_'+mode]=dict(candidate=int(c),correct=bool(y[k,c]))
        cases.append(case)
    result=dict(held_program_coverage=records,relation_pair_accuracy=relation,previously_reviewed_held_errors=cases,
        limitation='Historical TRAIN diagnostics. Only three reviewed parent questions contribute 21 held relation pairs; these pairs are correlated and not independent human gold.',test_opened=False)
    dump(out/'postexperiment_train_diagnostics.json',result);print(result,flush=True)

if __name__=='__main__':main()
