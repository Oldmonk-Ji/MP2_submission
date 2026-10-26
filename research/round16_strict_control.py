"""Repair evidence ablation without changing the frozen primary experiment.

The inherited link_features emits in_question even with context=False.
Remove the source question before BOTH fitting and predicting. This is a
post-audit engineering correction, not a newly independent audit result.
"""
from data_access import ROOT
from round16_rates import OUT,train,RateModel,evaluate
import json,joblib,hashlib,time

def stripped(r):
    return dict(r,question='')

class StrictControl:
    def __init__(self,path):self.model=RateModel(path)
    def predict(self,r,a):return self.model.predict(stripped(r),a)

def main():
    path=OUT/'strict_without_question.joblib'
    assert not path.exists(),'Preserve the correction record.'
    rows=json.loads((ROOT/'experiments/round15/fit_annotations.json').read_text(encoding='utf-8'))
    started=time.perf_counter();obj=train([stripped(r) for r in rows],False,True);joblib.dump(obj,path)
    result=dict(reason='Inherited link_features unconditionally exposed in_question; original without_question controls are not clean evidence ablations.',
                status='post-audit code correction; primary/frozen results unchanged',training=obj['training'],
                training_seconds=time.perf_counter()-started,asset_bytes=path.stat().st_size,
                asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
                original_ablation_valid=False,validation_allowed=False,test_access=False,metrics={})
    model=StrictControl(path)
    for name,p in [('development',ROOT/'experiments/round15/audit_annotations.json'),('opened_audit',OUT/'audit_annotations.json')]:
        data=json.loads(p.read_text(encoding='utf-8'));m,d=evaluate(data,model.predict);result['metrics'][name]=m
        (OUT/f'{name}_strict_control_details.json').write_text(json.dumps(d,indent=2))
    # Explicit intervention: arbitrary source questions cannot alter outputs.
    for r in rows:
        if r['kind']=='candidate':
            for a in r['anchors']:
                assert model.predict(r,a)==model.predict(dict(r,question='UNRELATED 999 per mystery'),a)
    result['source_question_intervention_check']='passed'
    (OUT/'strict_control_correction.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
