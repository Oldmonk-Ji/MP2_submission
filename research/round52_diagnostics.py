"""TRAIN-only executor/reward audit and relation generalization diagnostics."""
from data_access import ROOT,parse_answer
from round22_common import read,dump,load
from round51_programs import value
from round46_selector import probability
from round50_selector import complete
from round52_local import focus,csr,node_scores
import numpy as np,json,math

OUT=ROOT/'experiments/round52'

def main():
    b,_,y,ids=load('train');idrow={int(v):i for i,v in enumerate(ids)}
    with np.load(ROOT/'experiments/round50/train_targets.npz') as z:oldoracle=z['rewards'].any(1)
    with np.load(ROOT/'experiments/round51/train_targets.npz') as z:
        r=z['rewards'];m=z['mask'];matches=z['matches'];base=z['base'];folds=z['folds']
    ok=y[np.arange(len(y)),base].astype(bool);initial38=(folds==2)&~ok&~oldoracle
    sample=set(np.random.default_rng(52052).choice(len(ids),32,replace=False).tolist());programs=0;rootonly=np.zeros(len(y),bool);nonleaf=np.zeros(len(y),bool)
    for line in (ROOT/'experiments/round51/train_proposals.jsonl').open(encoding='utf-8'):
        p=json.loads(line);i=idrow[p['train_index']]
        for j,h in enumerate(p['hypotheses']):
            gold=any(y[i,k] for k in h['matches'])
            if gold:
                if h['program'][0] in ('N','L'):rootonly[i]=True
                else:nonleaf[i]=True
            if i in sample:
                assert str(value(h['program'],p['quantities']))==h['answer']
                assert bool(r[i,j])==bool(gold)
                assert h['matches']==[k for k,c in enumerate(b[i]['candidates']) if parse_answer(c['text'])==h['answer']]
                assert bool(m[i,j])==bool(h['matches']);programs+=1
    held=np.flatnonzero(folds==2);conflict=np.asarray([len({parse_answer(c['text']) for c in b[i]['candidates']}-{None})>=2 for i in held]);errs=~ok[held];cov=complete([b[i] for i in held],matches[held])
    stats=[]
    with np.load(OUT/'diagnostic_scores.npz') as z:
        for arm in ('sft','exact','grpo'):
            s=(z[arm+'_48048']+z[arm+'_48049'])/2;p=probability(s,m[held]);top=p.argmax(1)
            stats.append(dict(arm=arm,held_n=len(held),held_expected_reward=float((p*r[held]).sum(1).mean()),
                baseline_errors=int(errs.sum()),error_correct_program=int(r[held][errs].any(1).sum()),
                error_top_program_correct=int(r[held,top][errs].sum()),
                error_expected_reward=float((p*r[held]).sum(1)[errs].mean())))
    pairdata=read(OUT/'constraint_pairs.json');ff=[focus(p) for p in pairdata];xp=csr([p[0] for p in ff]);xn=csr([p[1] for p in ff]);fit=np.asarray([folds[idrow[p['train_index']]]!=2 for p in pairdata]);relation=[]
    for arm in ('sft','exact','grpo'):
        margins=[]
        for seed in (48048,48049):
            with np.load(OUT/'diagnostic'/f'{arm}_s{seed}'/'model.npz') as w:margins.append(node_scores(xp,w)-node_scores(xn,w))
        mm=np.mean(margins,axis=0);relation.append(dict(arm=arm,fit_pairs=int(fit.sum()),fit_correct=int((mm[fit]>0).sum())))
    result=dict(executor_audit=dict(rows=len(sample),programs=programs,all_passed=True),policy=stats,fit_relation=relation,
        initial_missing38=int(initial38.sum()),recovered_from_initial38=int((initial38&r.any(1)).sum()),
        recovered_nonleaf_from_initial38=int((initial38&nonleaf).sum()),
        leaf_only_numeric_coverage=int((rootonly&~nonleaf).sum()),held_error_nonleaf_coverage=int(nonleaf[held][errs].sum()),
        numeric_oracle_conflict_correct=int((ok[held]|r[held].any(1))[conflict].sum()),conflict_n=int(conflict.sum()),
        guarded_numeric_oracle_conflict_correct=int((ok[held]|(r[held].any(1)&cov))[conflict].sum()),
        numeric_correctness_not_semantic_gold=True,test_opened=False,
        exploratory_count_normalization=dict(train_held_only=True,not_adopted=True,sft_sum=1466,sft_mean=1467,exact_sum=1468,exact_mean=1468,grpo_sum=1467,grpo_mean=1467))
    dump(OUT/'diagnostics.json',result);print(result,flush=True)
    with np.load(ROOT/'experiments/round54/coverage.npz') as z:new=z['oracle']
    dump(ROOT/'experiments/round54/original38_audit.json',dict(n=38,recovered=int((initial38&new).sum()),still_missing=int((initial38&~new).sum()),
        rows=[dict(train_index=int(ids[i]),round51_covered=bool(r[i].any()),round54_covered=bool(new[i])) for i in np.flatnonzero(initial38)],
        test_opened=False,rule_design_used_fit_only=True))

if __name__=='__main__':main()
