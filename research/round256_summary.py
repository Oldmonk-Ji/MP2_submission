"""Audit all text-complement experiments; keep TRAIN/VAL and TEST separate."""
from round22_common import ROOT,read,dump,sha

def main():
    source_checks=0
    for stage in (248,249,250,252,254,255):
        plan=read(ROOT/f'experiments/round{stage}/plan.json')
        for name,expected in plan['source_hashes'].items():
            assert sha(ROOT/'research'/name)==expected,(stage,name)
            source_checks+=1
    for stage,name in ((247,'round247_complement_audit.py'),(251,'round251_policy_transfer.py')):
        assert read(ROOT/f'experiments/round{stage}/plan.json')['source_hash']==sha(ROOT/'research'/name)
        source_checks+=1
    # Correct an originally unscoped metadata flag. Inner design excludes fold2;
    # the final OOF branch deliberately fits experts on all TRAIN partitions.
    provenance=read(ROOT/'experiments/round248/result.json')
    provenance.update(held_labels_used_for_fitting=True,inner_design_uses_fold2_labels=False,
        final_oof_uses_fold2_labels=True,
        metadata_note='Original unscoped false referred to inner-design fitting only. Corrected after provenance review; training code, models and predictions are unchanged.')
    dump(ROOT/'experiments/round248/result.json',provenance)
    audit=read(ROOT/'experiments/round247/result.json')
    aggregate=dict(train_held={k:v for k,v in audit['train_held'].items() if k!='train_cases'},
        validation_aggregate_only=audit['validation_aggregate_only'],test_opened=False)
    dump(ROOT/'experiments/round247/aggregate.json',aggregate)
    records=[]
    for stage in (249,250,252,255):
        result=read(ROOT/f'experiments/round{stage}/result.json');assert not result['test_opened'] and result['selected'] is None
        for r in result['records']:
            records.append(dict(round=stage,arm=r['arm'],held=r['held'],validation=r.get('validation')))
    r=read(ROOT/'experiments/round251/result.json');assert not r['test_opened'] and not r['accepted']
    records.append(dict(round=251,arm='fixed_inner_train_policy',held=r['held'],validation=r['validation']))
    evaluated=[r for r in records if r['validation'] is not None]
    besth=max(records,key=lambda r:(r['held']['correct'],-r['held']['harms']))
    bestv=max(evaluated,key=lambda r:(r['validation']['correct'],r['held']['correct']))
    summary=dict(baseline=dict(round=235,validation=763,held=1479),audit=aggregate,
        additional_text_fits=4,configurations=len(records),validation_evaluations=len(evaluated),
        records=records,best_held=besth,best_validation=bestv,selected_submission_round=235,
        accepted_new=False,test_opened=False,source_hashes_verified=source_checks,
        provenance='Inner-policy training excludes fold2 labels. Final OOF policy training intentionally uses all TRAIN; held statistics always use the held-excluded policy.',
        conclusion='Complementarity improves held to1484, but every new VAL result is below763. Do not request/evaluate TEST.')
    out=ROOT/'experiments/round256';out.mkdir(exist_ok=True)
    dump(out/'result.json',summary);print(summary)

if __name__=='__main__':main()
