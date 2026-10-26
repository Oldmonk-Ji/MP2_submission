from data_access import ROOT
import json,hashlib
from round7_core import quantities,qfeatures,entities
from round7_learn import Model,metric,OUT

def main():
    plan=json.loads((OUT/'plan.json').read_text());tr=set(plan['existing_train_questions'])|{i for g in plan['groups'].values() for i in g['new_train']};audit={i for g in plan['groups'].values() for i in g['sealed_audit']};assert not tr&audit
    frozen=json.loads((OUT/'frozen.json').read_text())
    assert hashlib.sha256((OUT/'training_annotations.json').read_bytes()).hexdigest()==frozen['training_labels_sha256']
    assert hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest()==frozen['plan_sha256']
    for n,h in frozen['sources'].items():assert hashlib.sha256((ROOT/'research'/n).read_bytes()).hexdigest()==h
    for n,h in frozen['assets'].items():assert hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h
    q='Alice buys 2 books for $3 each. She has $10.';qs=quantities(q);assert [x['value'] for x in qs]==[2,3,10]
    assert len(quantities('Ann has 2 books and 2 pens.'))==2
    q2=q.replace('Alice','Diana');assert [qfeatures(q,qs,i) for i in range(3)]==[qfeatures(q2,quantities(q2),i) for i in range(3)]
    assert [x['value'] for x in quantities('Half of $3.50 is 7/4.')]==[.5,3.5,1.75]
    m=Model(OUT/'learned');m2=Model(OUT/'learned');assert m.predict(q)==m2.predict(q);assert m.predict('No numerical information.')==[]
    assert len(m.predict(q))==len(qs)
    # New names/amounts are code fixtures, not training examples.
    for p in m.predict(q):assert p['owner'] in entities(q);assert p['scope'] in (-1,0,1,2)
    record=dict(passed=True,checks=['question-disjoint sealed audit','training annotations and plan hashes','frozen code and models','occurrence preservation','fraction parsing','equal-length entity rename masks lexical identity','reload identical predictions','empty quantity input','predictions use input-derived candidates'],test_access=False,validation_access=False)
    (OUT/'checks.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
if __name__=='__main__':main()
