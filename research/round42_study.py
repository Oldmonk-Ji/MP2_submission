from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump,sha
from round42_relations import solve
from round41_study import GOLD
import numpy as np,datetime

OUT=ROOT/'experiments/round42'
REVIEW={2383:('123',True,'Disjoint indoor and outdoor table counts, each with its own chair rate.'),
2565:('25',True,'Separate bicycle/tricycle/unicycle batches and local wheel rates.'),
2668:('9600',False,'Source is 4 novels/month; target pages/year. Version41 lost the monthly time base and omitted 12 months/year.'),
2902:('60',True,'Two separate indoor/outdoor table batches, both 3 chairs/table.'),
3104:('480',True,'5 boxes times 8 bags/box times 12 balloons/bag.'),
3311:('400',True,'4 classes times 20 students/class times 5 sheets/student.'),
3844:('1200',False,'5 dozen boxes is 60 boxes. Version41 treated dozen as an ordinary adjective and lost factor12.'),
4176:('109',True,'5 bouquets at 5 roses each plus 7 table decorations at 12 roses each.')}

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'freeze.json').exists()
    oldaudit=read(ROOT/'experiments/round41/natural_relation_audit_packet.json');reviews=[]
    for r in oldaudit:
        a,good,note=REVIEW[r['train_index']];reviews.append(dict(train_index=r['train_index'],proof_supported=good,
            source_derived_answer=a,adjudication=note,provenance='assistant source review before official reference comparison; not independent human gold'))
    dump(ROOT/'experiments/round41/natural_relation_adjudications.json',reviews)
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    for r in reviews:assert parse_answer(refs[r['train_index']]['answer'])==r['source_derived_answer']
    dump(ROOT/'experiments/round41/result.json',dict(held_proofs=8,correct_complete_proofs=6,incorrect=2,precision=.75,
        decision='Rejected as proof authority before official validation. Implicit unit and time-scope facts missing.',test_opened=False))
    rows=read(ROOT/'experiments/round41/discovery.json')+oldaudit;supervision=[]
    for r in rows:
        i=r['train_index'];o=solve(r['question']);a=GOLD[i][0] if i in GOLD else REVIEW[i][0] if i in REVIEW else None
        assert o['answer']==a,(i,o,a)
        if a is None:continue
        for f in o['facts']:
            if f['evidence']:
                e=f['evidence'];assert r['question'][e['start']:e['end']]==e['text']
        supervision.append(dict(train_index=i,question=r['question'],labels=o,
            annotation='Assistant source adjudication; version41 discovery and audit are version42 discovery/regression data. Not fresh validation.'))
    lookup={r['train_index']:r['question'] for r in rows}
    tests=read(ROOT/'experiments/round41/counterfactual_tests.json')
    for t in tests:assert solve(t['question'])['answer']==t['expected']
    extra=[('remove_dozen',lookup[3844].replace('5 dozen boxes','5 boxes'),'100'),
        ('change_dozen_count',lookup[3844].replace('5 dozen','3 dozen'),'720'),
        ('same_time_base',lookup[2668].replace('in a year','in a month'),'800'),
        ('new_time_conversion','A printer prints 8 pages per minute. How many pages does it print in an hour?','480'),
        ('time_conversion_with_chain','A printer prints 8 sheets per minute. Each sheet has 2 pages. How many pages does it print in an hour?','960'),
        ('unknown_calendar_conversion','A printer prints 8 pages per day. How many pages does it print in a month?',None),
        ('workweek_not_calendar_week','Mira reads 8 pages per weekday. How many pages does she read in a week?',None),
        ('target_subset_unit','There are 4 boxes with 10 pens each. How many red pens are there?',None)]
    for name,q,a in extra:
        o=solve(q);assert o['answer']==a,(name,o,a);tests.append(dict(intervention=name,question=q,expected=a,passed=True))
    dump(OUT/'relation_supervision.json',supervision);dump(OUT/'counterfactual_tests.json',tests)
    dump(OUT/'freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),discovery_ids=list(lookup),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research/round42_relations.py',ROOT/'research/round42_study.py',OUT/'relation_supervision.json',OUT/'counterfactual_tests.json']},
        primary='One fixed directed relation graph and explicit unit/time conversion; no parameter sweep. Only source-provable target answers restrict original candidate final answers.',
        audit='All new accepted proofs in TRAIN folds1+2 excluding prior version41 discovery/audit; source review before reference comparison.',
        validation='One frozen variant versus round25 on all798. No test or external data. Previous validation from round40 did not supply parser changes.',test_opened=False))
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl')
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:ids=z['ids'];folds=z['folds']
    allrows=[];packet=[]
    for i,fold in zip(ids,folds):
        o=solve(bank[int(i)]['question']);r=dict(train_index=int(i),fold=int(fold),question=bank[int(i)]['question'],extraction=o);allrows.append(r)
        if fold!=0 and int(i) not in lookup and o['answer'] is not None:packet.append(r)
    dump(OUT/'train_extractions.json',allrows);dump(OUT/'natural_relation_audit_packet.json',packet)
    print(dict(supervised_questions=len(supervision),supervised_facts=sum(len(r['labels']['facts']) for r in supervision),
        counterfactuals=len(tests),all_proved=sum(r['extraction']['answer'] is not None for r in allrows),fresh_held_proved=len(packet)))
    for r in packet:print(r)

if __name__=='__main__':main()
