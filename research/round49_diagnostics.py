"""Same diagnostics plus reexecution of newly extracted arithmetic evidence."""
from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump
from round44_programs import value
import round48_diagnostics
import numpy as np,json

OUT=ROOT/'experiments/round49'

if __name__=='__main__':
    round48_diagnostics.OUT=OUT;round48_diagnostics.main()
    with np.load(OUT/'train_targets.npz') as z:ids=z['ids'];rewards=z['rewards'];matches=z['matches'];y=z['labels']
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');sample=set(np.random.default_rng(49049).choice(len(ids),32,replace=False).tolist());rows=0;programs=0;new=0
    with (OUT/'train_proposals.jsonl').open(encoding='utf-8') as f:
        for k,line in enumerate(f):
            if k not in sample:continue
            p=json.loads(line);gold=parse_answer(refs[int(ids[k])]['answer']);rows+=1
            for j,h in enumerate(p['hypotheses']):
                executed=str(value(h['program'],p['quantities']));assert executed==h['answer']
                assert bool(rewards[k,j])==(executed==gold and bool(h['matches']))
                assert bool(rewards[k,j])==bool((matches[k,j]&y[k].astype(bool)).any())
                programs+=1;new+=h['action']=='unit_separated_arithmetic_graph'
    dump(OUT/'reward_audit.json',dict(rows=rows,programs=programs,new_parser_programs=new,official_parser_agrees=True,test_opened=False))
    print('PARSER_REWARD_AUDIT',dict(rows=rows,programs=programs,new_parser_programs=new),flush=True)
