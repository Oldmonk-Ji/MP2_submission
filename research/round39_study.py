"""TRAIN-only relation supervision and frozen natural-language audit."""
from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump,sha,load
from round39_budget import compile
from fractions import Fraction
import numpy as np,datetime,json,sys

OUT=ROOT/'experiments/round39'
# Manually adjudicated semantic tuples, before natural held-fold inspection.
# c=count, p=price per item, n=total count, s=total spent, b=cash, r=change.
KINDS=dict(c='item_count',p='unit_price',n='total_item_count',s='total_spent_money',b='available_money',r='change_money')
GOLD={
3002:('p','top','5', [('c','pair short','5'),('c','pair shoe','2'),('c','top','4'),('p','pair short','7'),('p','pair shoe','10'),('s',None,'75')]),
275:('p','candy bar','4',[('c','soft drink','2'),('c','candy bar','5'),('p','soft drink','4'),('s',None,'28')]),
4737:('s',None,'100',[('c','shirt','10'),('c','pants','5'),('p','shirt','6'),('p','pants','8')]),
4402:('r',None,'50',[('c','tumbler','10'),('p','tumbler','45'),('b',None,'500')]),
4537:('s',None,'17',[('c','box taco shell','1'),('c','bell pepper','4'),('c','pound meat','2'),('p','box taco shell','5'),('p','bell pepper','3/2'),('p','pound','3')]),
201:('c','yard fabric','5',[('p','yard','24'),('c','pattern','1'),('p','pattern','15'),('c','spool silver thread','2'),('p','spool silver thread','3'),('s',None,'141')]),
3259:('c','large glass','5',[('c','small glass','8'),('p','small glass','3'),('p','large glass','5'),('b',None,'50'),('r',None,'1')]),
3635:('p','blue notebook','3',[('n','notebook','12'),('c','red notebook','3'),('c','green notebook','2'),('s',None,'37'),('p','red notebook','4'),('p','green notebook','2')]),
}

def build_supervision():
    assert not (OUT/'freeze.json').exists()
    rows=read(OUT/'discovery18.json')+[r for r in read(ROOT/'experiments/round38/discovery24.json') if r['train_index'] in (3259,3635)];records=[];tests=[]
    for row in rows:
        i=row['train_index'];q=row['question'];out=compile(q)
        for f in out['facts']:
            e=f['evidence'];assert q[e['start']:e['end']]==e['text']
        if i not in GOLD:
            assert out['status']=='unknown',(i,out)
            continue
        k,scope,answer,tuples=GOLD[i]
        expected={(KINDS[t],s,str(Fraction(v))) for t,s,v in tuples}
        actual={(f['kind'],f['scope'],f['value']) for f in out['facts']}
        assert expected==actual,(i,expected-actual,actual-expected)
        assert out['status']=='proved_within_grammar' and out['target']['kind']==KINDS[k] and out['target']['scope']==scope
        assert out['answer']==answer==parse_answer(row['reference']),(i,out)
        records.append(dict(train_index=i,question=q,relation_labels=out['facts'],target_label=out['target'],executable_answer=answer,
            provenance='assistant adjudicated semantic tuples from original TRAIN question; exact evidence checked; reference arithmetic used only as a separate sanity check',
            independent_human_gold=False))
        # Name substitutions must leave the ledger unchanged.
        import re
        renamed=re.sub(r'\b(?:Ann|Benny|James|Louis|Peter|Mike|Pauline|Petersons)\b','Morgan',q)
        z=compile(renamed);assert z['answer']==answer,(i,z)
        tests.append(dict(source_train_id=i,intervention='rename_person',expected=answer,passed=True))
    q={r['train_index']:r['question'] for r in rows}
    cases=[
      ('different_count',q[275].replace('5 candy bars','4 candy bars'),'5'),
      ('different_total',q[275].replace('28 dollars','38 dollars'),'6'),
      ('same_number_different_scope',q[275].replace('5 candy bars','2 candy bars'),'10'),
      ('banknote_count',q[4402].replace('five $100 bills','six $100 bills'),'150'),
      ('no_spending_evidence',q[3259].replace(' and leaves with $1 in change',''),None),
      ('contradictory_amount',q[4737].replace('How much','He spent $101. How much'),None),
      ('batch_price_is_not_unit_price',q[275].replace('$ 4 each','$ 4'),None),
      ('unit_price_same_numbers',q[3635].replace('4 dollars each','2 dollars each'),'27/7'),
    ]
    for name,question,answer in cases:
        z=compile(question);assert z['answer']==answer,(name,z,answer)
        tests.append(dict(intervention=name,question=question,expected=answer,passed=True))
    dump(OUT/'relation_supervision.json',records);dump(OUT/'counterfactual_tests.json',tests)
    dump(OUT/'discovery_extractions_final.json',[dict(train_index=r['train_index'],question=r['question'],extraction=compile(r['question'])) for r in rows])
    print(dict(relation_questions=len(records),facts=sum(len(r['relation_labels']) for r in records),counterfactual_checks=len(tests)))

def freeze():
    assert not (OUT/'freeze.json').exists()
    sources=[ROOT/'research/round39_budget.py',ROOT/'research/round39_study.py',OUT/'relation_supervision.json',OUT/'counterfactual_tests.json']
    dump(OUT/'freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
      discovery='root38 fold0 errors plus hash-selected purchase discovery18 fold0',
      natural_relation_audit='All proved outputs in folds1+2, source questions and facts reviewed before numeric reference comparison. No label or reference supplied to compiler.',
      primary_selection='One fixed rule: a complete purchase ledger answer may restrict candidates by their ORIGINAL final numeric answer; retain baseline on unknown/no matching candidate. Among matches use historical baseline scores. No threshold/parameter search.',
      baseline='historical round22 honest OOF predictions for TRAIN effects; frozen round25 for official validation',
      quality='Check relation truth and full-scope completeness separately from final-number agreement; any semantic audit error prevents deploying this version as proof authority.',
      validation='One full 798-question evaluation of the frozen module after TRAIN semantic audit; official validation selects winner versus round25; no validation labels become relation supervision.',
      test_opened=False,external_training_data=False))
    print('FROZEN before held-fold question inspection')

def extract_train():
    p=read(OUT/'freeze.json')
    for file,h in p['source_hashes'].items():assert sha(ROOT/file)==h
    # Candidate bank only: no reference/label needed for extraction or audit packet.
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl')
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:ids=z['ids'];folds=z['folds']
    allrows=[];packet=[]
    for i,fold in zip(ids,folds):
        r=bank[int(i)];o=compile(r['question']);record=dict(train_index=int(i),fold=int(fold),question=r['question'],extraction=o);allrows.append(record)
        if fold!=0 and o['status']=='proved_within_grammar':packet.append(record)
    dump(OUT/'train_extractions.json',allrows);dump(OUT/'natural_relation_audit_packet.json',packet)
    print(dict(all_proved=sum(r['extraction']['status']=='proved_within_grammar' for r in allrows),held_proved=len(packet),held_ids=[r['train_index'] for r in packet]))

if __name__=='__main__':globals()[sys.argv[1]]()
