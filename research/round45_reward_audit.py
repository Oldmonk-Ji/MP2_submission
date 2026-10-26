"""Independently re-execute sampled TRAIN programs and verify cached rewards."""
from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump
from round29_programs import value
import numpy as np,json

def main():
    out=ROOT/'experiments/round45'
    with np.load(ROOT/'experiments/round44/train_targets.npz') as z:ids=z['ids'];rewards=z['rewards'];labels=z['labels'];mask=z['mask']
    rng=np.random.default_rng(45045);positions=set(rng.choice(len(ids),32,replace=False).tolist())
    for source_id in (670,3635):positions.add(int(np.flatnonzero(ids==source_id)[0]))
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl');executed=0
    with (ROOT/'experiments/round44/train_proposals.jsonl').open(encoding='utf-8') as f:
        for k,line in enumerate(f):
            if k not in positions:continue
            packet=json.loads(line);i=int(ids[k]);assert packet['train_index']==i
            gold=parse_answer(refs[i]['answer']);answers=[parse_answer(c['text']) for c in bank[i]['candidates']]
            assert np.array_equal(labels[k],np.asarray([a==gold for a in answers],np.float32))
            assert mask[k].sum()==len(packet['hypotheses'])
            for j,h in enumerate(packet['hypotheses']):
                result=str(value(h['program'],packet['quantities']));assert result==h['answer']
                matches=[j for j,a in enumerate(answers) if a is not None and a==result];assert matches==h['matches']
                assert float(result==gold and bool(matches))==rewards[k,j];executed+=1
    result=dict(train_questions_checked=len(positions),programs_reexecuted=executed,candidate_rewards_match_official_parser=True,
        relation_rewards_match_execution_and_train_reference=True,test_opened=False,train_indices=sorted(int(ids[k]) for k in positions))
    dump(out/'reward_audit.json',result);print(result,flush=True)

if __name__=='__main__':main()
