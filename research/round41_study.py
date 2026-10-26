from data_access import ROOT,CODE,read_jsonl
from round22_common import read,dump,sha
from round41_relations import solve
import numpy as np,datetime

OUT=ROOT/'experiments/round41'
# Directed relationships adjudicated from natural TRAIN questions, before audit.
GOLD={
1257:('63',[('lily','petal','6'),('tulip','petal','3')],[('lily','8'),('tulip','5')]),
4335:('58',[('car','people','4'),('taxi','people','6'),('van','people','5')],[('car','3'),('taxi','6'),('van','2')]),
2430:('480',[('carton','pack','5'),('pack','stick','3'),('brown box','carton','4')],[('brown box','8')]),
112:('900',[('page','donut','1/2'),('donut','calory','150')],[('page','12')]),
407:('1000',[('bank teller','roll','10'),('roll','coin','25')],[('bank teller','4')]),
3243:('1008',[('apartment building','floor','12'),('floor','apartment','6'),('apartment','door','7')],[('apartment building','2')])}

def main():
    assert not (OUT/'freeze.json').exists()
    rows=read(OUT/'discovery.json');supervision=[];lookup={r['train_index']:r['question'] for r in rows}
    for r in rows:
        i=r['train_index'];o=solve(r['question'])
        if i not in GOLD:assert o['answer'] is None,(i,o);continue
        a,edges,bases=GOLD[i];assert o['answer']==a
        assert {(f['parent'],f['child'],f['value']) for f in o['facts'] if f['kind']=='per_relation'}==set(edges)
        assert {(f['unit'],f['value']) for f in o['facts'] if f['kind']=='total_count'}==set(bases)
        for f in o['facts']:
            e=f['evidence'];assert r['question'][e['start']:e['end']]==e['text']
        supervision.append(dict(**r,labels=o,annotation='assistant source-adjudicated relation tuples with occurrence evidence; not independent human gold'))
    tests=[
      ('missing_level',lookup[3243].replace('with 12 floors each',''),None),
      ('same_value_distinct_units',lookup[3243].replace('2 apartment buildings','6 apartment buildings'),'3024'),
      ('change_local_rate',lookup[3243].replace('12 floors each','10 floors each'),'840'),
      ('different_container_count',lookup[2430].replace('8 brown boxes','6 brown boxes'),'360'),
      ('different_denominator',lookup[112].replace('per 2 pages','per 3 pages'),'600'),
      ('different_flower_rate',lookup[1257].replace('6 petals','4 petals'),'47'),
      ('unrelated_numeric_fact',lookup[1257].replace('How many','She has 3 stones. How many'),None),
      ('target_part_vs_total',lookup[407].replace('have in all','have each'),None),
      ('noun_head_not_modifier','There are 2 bird cages with 3 birds each. Each bird has 4 feathers. How many feathers are there?','24'),
      ('owner_collision','Mia has 2 boxes. Lee has 2 boxes. Each box has 3 pens. How many pens do they have?',None)]
    checked=[]
    for name,q,a in tests:
        o=solve(q);assert o['answer']==a,(name,o,a);checked.append(dict(intervention=name,question=q,expected=a,passed=True))
    dump(OUT/'relation_supervision.json',supervision);dump(OUT/'counterfactual_tests.json',checked)
    dump(OUT/'freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),discovery_ids=list(lookup),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research/round41_relations.py',ROOT/'research/round41_study.py',OUT/'relation_supervision.json',OUT/'counterfactual_tests.json']},
        primary='A source-grounded directed quantity graph with one path per source to requested total; every relation must connect; no parameter sweep.',
        audit='Review every accepted proof in TRAIN folds1+2 before comparing references. Exact source evidence and target/aggregation truth judged separately from numeric matching.',
        validation='Evaluate one frozen rule on full 798 validation questions against round25, only after audit; no validation-specific parser edits.',test_opened=False))
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl')
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:ids=z['ids'];folds=z['folds']
    allrows=[];packet=[]
    for i,fold in zip(ids,folds):
        o=solve(bank[int(i)]['question']);r=dict(train_index=int(i),fold=int(fold),question=bank[int(i)]['question'],extraction=o);allrows.append(r)
        if fold!=0 and o['answer'] is not None:packet.append(r)
    dump(OUT/'train_extractions.json',allrows);dump(OUT/'natural_relation_audit_packet.json',packet)
    print(dict(supervision=len(supervision),relation_facts=sum(len(r['labels']['facts']) for r in supervision),counterfactuals=len(checked),
        proved_train=sum(r['extraction']['answer'] is not None for r in allrows),proved_held=len(packet)))
    for r in packet:print(r)

if __name__=='__main__':main()
