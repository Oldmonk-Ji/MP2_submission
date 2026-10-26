"""Annotate unseen natural candidates after freeze, before viewing predictions."""
from data_access import ROOT
from round15_annotations import build
from round15_learn import Model,evaluate,rule
import json,hashlib,time
OUT=ROOT/'experiments/round15'
AUDIT={
'2516:-1':'mary/N/- mary/I/-',
'2516:0':'mary/N/- mary/E/-',
'2516:2':'mary/N/- mary/N/- mary/N/- mary/E/-',
'315:-1':'james/N/- NIL/N/unit NIL/D/-',
'315:0':'james/N/- NIL/N/unit james/N/- james/E/-',
'315:1':'james/N/- NIL/N/unit james/N/- james/N/-',
'4384:-1':'cody/N/- cody/N/-',
'4384:0':'cody/N/- NIL/D/-',
'4384:1':'cody/N/- NIL/D/-',
'682:-1':'NIL/I/treadmill NIL/N/plates',
'682:0':'andre/N/- NIL/N/plates NIL/N/plates INVALID',
'682:1':'andre/N/- NIL/N/plates INVALID',
'3481:-1':'NIL/N/dress vanessa/I/- vanessa/N/week vanessa/N/weekend',
'3481:0':'vanessa/I/- vanessa/N/week vanessa/N/weekend NIL/D/-',
'3481:1':'vanessa/I/- vanessa/N/week vanessa/N/weekend vanessa/N/week',
'4163:-1':'NIL/N/kW coco/P/-',
'4163:0':'coco/N/- coco/P/-',
'4163:1':'coco/N/kWh coco/N/- coco/P/-',
'3419:-1':'NIL/I/gallon_of_whole_milk NIL/N/gallon_of_whole_milk NIL/X/- NIL/F/-',
'3419:0':'NIL/I/gallon_of_whole_milk NIL/N/gallon_of_whole_milk NIL/X/- NIL/F/-',
'3419:1':'NIL/I/gallon_of_whole_milk NIL/N/gallon_of_whole_milk NIL/X/- NIL/E/-',
'2335:-1':'minnie/I/- NIL/D/-',
'2335:0':'NIL/D/- NIL/D/-',
'2335:2':'minnie/I/- NIL/D/-',
'1705:-1':'mildred/N/- candice/N/- NIL/I/-',
'1705:0':'mildred/N/- candice/N/- NIL/I/-',
'1705:1':'mildred/N/- candice/N/- NIL/I/-',
}


def main():
    frozen=json.loads((OUT/'frozen.json').read_text())
    for name,h in frozen['sources'].items():assert hashlib.sha256((ROOT/'research'/name).read_bytes()).hexdigest()==h
    for name,h in frozen['assets'].items():assert hashlib.sha256((OUT/name/'model.joblib').read_bytes()).hexdigest()==h
    assert hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest()==frozen['plan_sha256']
    assert hashlib.sha256((OUT/'fit_annotations.json').read_bytes()).hexdigest()==frozen['labels_sha256']
    build('audit',AUDIT)
    seal=dict(labels_sha256=hashlib.sha256((OUT/'audit_annotations.json').read_bytes()).hexdigest(),model_freeze_sha256=hashlib.sha256((OUT/'frozen.json').read_bytes()).hexdigest(),source='same-assistant review before predictions; not independently human verified')
    (OUT/'audit_annotation_seal.json').write_text(json.dumps(seal,indent=2))
    rows=json.loads((OUT/'audit_annotations.json').read_text(encoding='utf-8'));candidate=[r for r in rows if r['kind']=='candidate'];result={}
    systems={name:Model(OUT/name).predict for name in frozen['assets']};systems['literal_rule']=rule
    for name,predict in systems.items():
        started=time.perf_counter();m,detail=evaluate(candidate,predict);m['seconds']=time.perf_counter()-started
        m['by_category']={cat:evaluate([r for r in candidate if r['category']==cat],predict)[0] for cat in ('coref','price','time')}
        result[name]=m;(OUT/f'audit_{name}.json').write_text(json.dumps(dict(metrics=m,details=detail),ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,json.dumps(m),flush=True)
    m=result['natural_candidates'];gate=json.loads((OUT/'plan.json').read_text())['audit_gate']
    passed=m['anchors']>=gate['min_candidate_anchors'] and (m['emitted_precision'] or 0)>=gate['min_joint_precision'] and m['correct_coverage']>=gate['min_correct_coverage'] and all(v['joint_accuracy']>=gate['min_each_category_joint'] for v in m['by_category'].values())
    decision=dict(systems=result,extraction_gate_passed=passed,validation_allowed=False,test_access=False,
                  deployment='diagnostic_only; retain round4 selector',
                  reason='Natural candidate coverage/precision gates not passed.' if not passed else 'Extraction screening passed; still requires fair outer train-side comparison and independent semantic review before validation.',
                  synthetic_examples_in_audit=0,final_answer_labels_used=0,
                  limitations=['Only predeclared sampled anchors; no whole-trace recall estimate.','Single assistant annotation.','Nine train parents; low sample size.','Stage is one coarse anchor field, not both endpoints of a general relation.'])
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2));print('GATE',passed,flush=True)


if __name__=='__main__':main()
