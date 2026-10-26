"""Independent reward/source audit and mechanisms, TRAIN only."""
from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump
from round46_selector import probability
from round44_programs import value
from round46_graph import graph,features
import numpy as np,json

OUT=ROOT/'experiments/round47'

def main():
    with np.load(OUT/'train_targets.npz') as z:
        ids=z['ids'];rewards=z['rewards'];mask=z['mask'];matches=z['matches'];base=z['base'];y=z['labels'];folds=z['folds']
    old=json.loads((ROOT/'experiments/round46/discovery_review_before_reference.json').read_text(encoding='utf-8'))
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    sampled=set(np.random.default_rng(47047).choice(len(ids),32,replace=False).tolist());program_count=0;rows=0
    with (ROOT/'experiments/round46/train_proposals.jsonl').open(encoding='utf-8') as f:
        for k,line in enumerate(f):
            if k not in sampled:continue
            p=json.loads(line);assert p['train_index']==int(ids[k]);gold=parse_answer(refs[int(ids[k])]['answer']);rows+=1
            for j,h in enumerate(p['hypotheses']):
                executed=str(value(h['program'],p['quantities']));assert executed==h['answer']
                assert bool(rewards[k,j])==(executed==gold and bool(h['matches']))
                assert bool(rewards[k,j])==bool((matches[k,j]&y[k].astype(bool)).any())
                program_count+=1
    alias_checks=[]
    for r in read(OUT/'constraint_pairs.json'):
        if not r.get('decision',{}).get('same_numeric_answer'):continue
        q=r['question'];qs=graph(q)['quantities'];assert value(r['positive'],qs)==value(r['negative'],qs)
        assert features(q,qs,r['positive'])!=features(q,qs,r['negative'])
        alias_checks.append(r['train_index'])
    records=[]
    with np.load(ROOT/'experiments/round44/train_targets.npz') as z:old_oracle=z['rewards'].any(1)
    baseline_ok=y[np.arange(len(y)),base].astype(bool)
    budget_ablation={}
    for label,ix in [('all_train',np.ones(len(ids),bool)),('held_errors',(folds==2)&~baseline_ok)]:
        budget_ablation[label]=dict(n=int(ix.sum()),old_32=int(old_oracle[ix].sum()),new_graph_first_32=int(rewards[ix,:32].any(1).sum()),new_graph_64=int(rewards[ix].any(1).sum()))
    with np.load(OUT/'diagnostic_scores.npz') as z:
        held=z['held'];oldok=y[held,base[held]].astype(bool)
        for arm in ('sft','exact','grpo'):
            s=(z[f'{arm}_46046']+z[f'{arm}_46047'])/2;p=probability(s,mask[held]);top=p.argmax(1)
            for name,keep in [('all',np.ones(len(held),bool)),('baseline_errors',~oldok)]:
                records.append(dict(arm=arm,group=name,n=int(keep.sum()),correct_program_available=int(rewards[held][keep].any(1).sum()),
                    top_program_correct=int(rewards[held][np.arange(len(held)),top][keep].sum()),
                    expected_reward=float((p*rewards[held]).sum(1)[keep].mean())))
    history=[]
    for stage in ('diagnostic','full'):
        for arm in ('sft','exact','grpo'):
            hs=[read(OUT/stage/f'{arm}_s{seed}'/'training.json')['history'] for seed in (46046,46047)]
            history.append(dict(stage=stage,arm=arm,first={k:float(np.mean([h[0][k] for h in hs])) for k in hs[0][0]},last={k:float(np.mean([h[-1][k] for h in hs])) for k in hs[0][-1]}))
    report=dict(reward_audit_rows=rows,reexecuted_programs=program_count,official_parser_agreement=True,
        source_equal_value_constraints=len(alias_checks),source_features_distinguish_same_value_bindings=True,
        held_rewards=records,proposal_budget_ablation=budget_ablation,training=history,test_opened=False)
    dump(OUT/'reward_diagnostics.json',report);print(report,flush=True)

if __name__=='__main__':main()
