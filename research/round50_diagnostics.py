"""TRAIN-only causal ablation of missing-evidence protection and reward audit."""
from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump,load
from round44_programs import value
from round46_selector import select as unguarded
from round50_selector import select,complete
from round46_train import metrics
import round48_diagnostics
import numpy as np,json,math

OUT=ROOT/'experiments/round50'

if __name__=='__main__':
    round48_diagnostics.OUT=OUT;round48_diagnostics.main()
    b,_,y,ids=load('train')
    with np.load(OUT/'train_targets.npz') as z:
        rewards=z['rewards'];matches=z['matches'];mask=z['mask'];base=z['base'];bs=z['base_scores'];folds=z['folds']
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');sample=set(np.random.default_rng(50050).choice(len(ids),32,replace=False).tolist());programs=0;new=0
    with (ROOT/'experiments/round49/train_proposals.jsonl').open(encoding='utf-8') as f:
        for k,line in enumerate(f):
            if k not in sample:continue
            p=json.loads(line);gold=parse_answer(refs[int(ids[k])]['answer'])
            for j,h in enumerate(p['hypotheses']):
                executed=str(value(h['program'],p['quantities']));assert executed==h['answer']
                assert bool(rewards[k,j])==(executed==gold and bool(h['matches']))
                assert bool(rewards[k,j])==bool((matches[k,j]&y[k].astype(bool)).any())
                assert bool(mask[k,j])==bool(h['matches'])
                programs+=1;new+=h['action']=='unit_separated_arithmetic_graph'
    dump(OUT/'reward_audit.json',dict(rows=len(sample),programs=programs,new_parser_programs=new,official_parser_agrees=True,legal_actions_only=True,test_opened=False))
    records=[]
    with np.load(OUT/'diagnostic_scores.npz') as z:
        held=z['held'];bb=[b[i] for i in held];conflict=np.asarray([len({parse_answer(c['text']) for c in r['candidates']}-{None})>=2 for r in bb]);cov=complete(bb,matches[held])
        for arm in ('sft','exact','grpo'):
            s=(z[f'{arm}_48048']+z[f'{arm}_48049'])/2
            for guard,func in ((False,unguarded),(True,select)):
                c=func(bb,bs[held],s,mask[held],matches[held]);rec=metrics(y[held],c,base[held],conflict);rec.update(arm=arm,unknown_guard=guard);records.append(rec)
    oldok=y[held,base[held]].astype(bool);oracle=rewards[held].any(1)
    # Upper bound only for selecting an executable program with baseline fallback.
    upper=oldok|oracle;guardupper=oldok|(oracle&cov)
    ceiling=dict(held_conflict_n=int(conflict.sum()),baseline_conflict_correct=int(oldok[conflict].sum()),
        executable_or_baseline_oracle=int(upper[conflict].sum()),executable_or_baseline_oracle_percent=100*float(upper[conflict].mean()),
        guarded_oracle=int(guardupper[conflict].sum()),minimum_correct_for97=math.ceil(.97*int(conflict.sum())),
        caveat='TRAIN historical held subset, not TEST. Numeric oracle is not semantic proof. Bound applies to program-choice plus baseline fallback, not arbitrary future candidate models.')
    dump(OUT/'unknown_guard_ablation.json',dict(records=records,held_complete_answer_coverage=int(cov.sum()),oracle_ceiling=ceiling,test_opened=False))
    print('UNKNOWN_GUARD_ABLATION',records,flush=True);print('TRAIN_ORACLE_CEILING',ceiling,flush=True)
