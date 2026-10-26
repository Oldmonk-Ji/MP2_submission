from data_access import ROOT,forbid_test
from round17_model import OUT,Model
from round17_labels import norm
import json,re,hashlib

def main():
    plan=json.loads((OUT/'plan.json').read_text(encoding='utf-8'));freeze=json.loads((OUT/'frozen.json').read_text())
    for name,h in freeze['source_sha256'].items():assert hashlib.sha256((ROOT/'research'/name).read_bytes()).hexdigest()==h
    fit=json.loads((OUT/'fit_annotations.json').read_text(encoding='utf-8'));audit=json.loads((OUT/'audit_annotations.json').read_text(encoding='utf-8'))
    assert not {r['train_index'] for r in fit}&{r['train_index'] for r in audit}
    assert not {v['family'] for v in plan['groups'].values() if v['split']=='fit'}&{v['family'] for v in plan['groups'].values() if v['split']=='audit'}
    for r in fit+audit:
        assert hashlib.sha256(r['text'].encode()).hexdigest()==r['sha256']
        for a,g in zip(r['anchors'],r['labels']):
            assert r['text'][a['start']:a['end']]==a['raw']
            for e in g['object_lexical_evidence']:
                s=r['text'] if e['source']=='claim' else r['question']
                assert s[e['start']:e['end']]==e['text'] and norm(e['text'])==g['object']
    model=Model(OUT/'local_model.joblib');reload=Model(OUT/'local_model.joblib')
    for r in fit[:6]:
        for a in r['anchors']:
            p=model.predict(r,a)
            assert p==reload.predict(r,a)==model.predict(dict(r,id='unseen',train_index=-1,candidate_index=99),a)
            # Numeric equality is NOT an input to object/role/predicate models.
            assert p==model.predict(r,dict(a,value=987654321.0))
    pack=json.loads((ROOT/'review_app/questions.json').read_text(encoding='utf-8'))
    assert len(pack['questions'])==3
    for q in pack['questions']:
        assert plan['groups'][str(q['train_index'])]['split']=='fit'
        assert not {'answer','prediction','gold','program_implied_choice'}&set(q)
    html=(ROOT/'review_app/复核工作台.html').read_text(encoding='utf-8')
    assert json.loads(re.search(r'<script id="initial-pack" type="application/json">(.*?)</script>',html,re.S).group(1))==pack
    old=json.loads((ROOT/'review_app/questions_第四批.json').read_text(encoding='utf-8'));assert old['id']=='mp2-open-train-review-04' and len(old['questions'])==7
    for name,d in json.loads((ROOT/'experiments/round4/asset_manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/'experiments/round4/selected_bundle'/name).read_bytes()).hexdigest()==d['sha256']
    try:forbid_test('open',('gsm8k_test.jsonl',))
    except PermissionError:pass
    else:raise AssertionError('test protection missing')
    result=dict(status='passed',parent_and_template_isolation=True,source_and_object_spans=True,
                freeze_source_hashes=True,reload_and_unseen_ids=True,value_equality_not_model_input=True,
                public_fit_only_no_predictions=True,embedded_pack_matches=True,previous_pack_preserved=True,
                baseline_assets_unchanged=True,test_guard=True)
    (OUT/'checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
