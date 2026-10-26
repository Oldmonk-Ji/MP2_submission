"""Integrity, no-evidence intervention, serialization and bounded diagnostics."""
from data_access import ROOT,forbid_test
from round16_rates import OUT,RateModel
from round16_run import contrasts
from round16_strict_control import StrictControl
import json,hashlib,time

def main():
    plan=json.loads((OUT/'plan.json').read_text());freeze=json.loads((OUT/'frozen.json').read_text())
    for name,h in freeze['sources'].items():assert hashlib.sha256((ROOT/'research'/name).read_bytes()).hexdigest()==h
    for name,data in freeze['assets'].items():assert hashlib.sha256((OUT/(name+'.joblib')).read_bytes()).hexdigest()==data['sha256']
    seal=json.loads((OUT/'audit_annotation_seal.json').read_text())
    # The packet/label seals hash canonical LF text before Windows write_text
    # newline conversion. Model/source seals hash physical file bytes instead.
    assert hashlib.sha256((OUT/'audit_annotations.json').read_text(encoding='utf-8').encode()).hexdigest()==seal['sha256']
    assert hashlib.sha256((OUT/'audit_packet.json').read_text(encoding='utf-8').encode()).hexdigest()==plan['audit_packet_sha256']
    assert not set(plan['fresh_audit_parents'])&set(plan['excluded_parents'])
    rows=json.loads((OUT/'audit_annotations.json').read_text(encoding='utf-8'))
    model=RateModel(OUT/'local_pointer.joblib');reload=RateModel(OUT/'local_pointer.joblib')
    strict=StrictControl(OUT/'strict_without_question.joblib');tested=0
    for r in rows:
        assert hashlib.sha256(r['text'].encode()).hexdigest()==r['sha256']
        for a in r['anchors']:
            assert r['text'][a['start']:a['end']]==a['raw']
            assert r['text'][a['context_start']:a['context_end']]==a['context']
        if r['kind']=='candidate' and r['anchors'] and tested<3:
            a=r['anchors'][0];p=model.predict(r,a)
            assert p==reload.predict(r,a)==model.predict(dict(r,id='unseen',train_index=-123,candidate_index=99),a)
            assert strict.predict(r,a)==strict.predict(dict(r,question='irrelevant source 999 each'),a)
            tested+=1
    for name,h in json.loads((ROOT/'experiments/round4/asset_manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/'experiments/round4/selected_bundle'/name).read_bytes()).hexdigest()==h['sha256']
    assert contrasts([],model)==([],dict(answer_conflict_pairs=0,proposed_records=0,proposed_pairs=0,proposed_questions=0,verified_support_records=0,note='Proposal coverage is not verified alignment coverage; repeated values may be unresolved.'))
    # Warm diagnostic cost, measured separately from official end-to-end time.
    started=time.perf_counter();count=0
    for r in rows:
        if r['kind']=='candidate':
            for a in r['anchors']:model.predict(r,a);count+=1
    seconds=time.perf_counter()-started
    try:forbid_test('open',('gsm8k_test.jsonl',))
    except PermissionError:pass
    else:raise AssertionError('test guard missing')
    result=dict(status='passed',frozen_sources_assets=True,baseline_unchanged=True,
                audit_physical_sha256=hashlib.sha256((OUT/'audit_annotations.json').read_bytes()).hexdigest(),
                raw_span_offsets=True,train_parent_disjointness=True,reload_unseen_ids=True,
                strict_control_source_intervention=True,test_guard=True,
                warm_diagnostic_timing=dict(questions=8,candidate_texts=32,anchors=count,seconds=seconds,
                    official_timing=False,note='Small sample, no baseline or cold-start/scorer. No eligibility claim.'))
    (OUT/'checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
