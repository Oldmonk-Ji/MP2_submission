from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump,sha
from round39_study import GOLD,KINDS
from round40_budget import compile
import numpy as np,datetime,sys

OUT=ROOT/'experiments/round40'
NEW={
1095:('item_total_money','box laundry detergent','11',[('c','shower gel','4'),('b',None,'60'),('p','shower gel','4'),('c','tube toothpaste','1'),('p','tube toothpaste','3'),('r',None,'30'),('c','box laundry detergent','1')]),
1798:('shortfall_money',None,'43',[('c','black printer ink','2'),('c','red printer ink','3'),('c','yellow printer ink','2'),('b',None,'50'),('p','black printer ink','11'),('p','red printer ink','15'),('p','yellow printer ink','13')]),
1048:('change_money',None,'5',[('c','lollipop','4'),('c','pack gummy','2'),('p','lollipop','3/2'),('p','pack gummy','2'),('b',None,'15')]),
453:('average_unit_price','movie','14',[('c','movie dvd','8'),('c','movie blu ray','4'),('p','movie dvd','12'),('p','movie blu ray','18')])}

def prepare():
    assert not (OUT/'freeze.json').exists()
    rows=read(OUT/'discovery_extractions.json');supervision=[]
    for r in rows:
        i=r['train_index'];q=r['question'];o=compile(q)
        if i in GOLD or i in NEW:
            if i in GOLD:k,s,a,tuples=GOLD[i];k=KINDS[k]
            else:k,s,a,tuples=NEW[i]
            assert o['answer']==a and o['target']['kind']==k and o['target']['scope']==s,(i,o)
            assert {(f['kind'],f['scope'],f['value']) for f in o['facts']}=={(KINDS[kind],s,v) for kind,s,v in tuples},i
            for f in o['facts']:
                e=f['evidence'];assert q[e['start']:e['end']]==e['text']
            supervision.append(dict(train_index=i,question=q,relation_labels=o['facts'],target=o['target'],answer=a,
                source='assistant adjudication; exact evidence and semantic slots; not independently human validated'))
        if i==3232:assert o['answer'] is None
    # Recheck all existing contrast interventions; compare expected outputs,
    # not just the compiler's own outputs.
    tests=[]
    for t in read(ROOT/'experiments/round39/counterfactual_tests.json'):
        if 'question' in t:assert compile(t['question'])['answer']==t['expected'];tests.append(t)
    byid={r['train_index']:r['question'] for r in rows}
    changes=[('target_average_to_total',byid[453].replace('What is the average price he paid per movie?','How much did he spend in total?'),'168'),
        ('target_total_to_difference',byid[4737].replace('How much did everything cost?','How much more did the shirts cost than the pants?'),None),
        ('change_to_total',byid[1048].replace('how much was she left with after spending on the candies?','how much did she spend in all?'),'10'),
        ('cash_vs_residual',byid[1095].replace('$30 remaining','$20 remaining'),'21'),
        ('same_repeated_purchase_scope','Mira bought 2 pens for $3 each. She also bought 2 pens for $3 each. How much did she spend in total?',None)]
    for name,q,a in changes:
        o=compile(q);assert o['answer']==a,(name,o)
        tests.append(dict(intervention=name,question=q,expected=a,passed=True))
    dump(OUT/'relation_supervision.json',supervision);dump(OUT/'counterfactual_tests.json',tests)
    dump(OUT/'freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research/round40_budget.py',ROOT/'research/round40_study.py',OUT/'relation_supervision.json',OUT/'counterfactual_tests.json']},
        discovery_ids=[r['train_index'] for r in rows],
        former_audit_now_discovery=[453,1657,1890,2729,3232,4173],
        audit='Only previously uninspected natural TRAIN questions in folds1+2; every new proved output reviewed before reference comparison. Old audit questions explicitly become discovery/regression cases, not fresh evidence.',
        selection='Single fixed answer restriction by complete ledger; baseline on unknown or no candidate matching original final answer; no margin sweep.',
        validation_role='Full official validation selects versus frozen round25 after TRAIN relation audit; no validation-derived labels or code changes.',
        test_opened=False))
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl');discovery={r['train_index'] for r in rows}
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:ids=z['ids'];folds=z['folds']
    extracted=[];packet=[]
    for i,fold in zip(ids,folds):
        r=bank[int(i)];o=compile(r['question']);item=dict(train_index=int(i),fold=int(fold),question=r['question'],extraction=o);extracted.append(item)
        if fold!=0 and int(i) not in discovery and o['answer'] is not None:packet.append(item)
    dump(OUT/'train_extractions.json',extracted);dump(OUT/'natural_relation_audit_packet.json',packet)
    print(dict(supervision_questions=len(supervision),facts=sum(len(r['relation_labels']) for r in supervision),contrast_tests=len(tests),
        train_proved=sum(r['extraction']['answer'] is not None for r in extracted),fresh_held_proved=len(packet)))
    for r in packet:print(r)

if __name__=='__main__':prepare()
