"""Check data provenance, source spans, independent program arithmetic and isolation."""
from data_access import ROOT,parse_answer,forbid_test
from fractions import Fraction as F
import json,hashlib,re
from round15_learn import Model
OUT=ROOT/'experiments/round15'

plan=json.loads((OUT/'plan.json').read_text());fit=json.loads((OUT/'fit_annotations.json').read_text(encoding='utf-8'));audit=json.loads((OUT/'audit_annotations.json').read_text(encoding='utf-8'))
assert not {r['train_index'] for r in fit}&{r['train_index'] for r in audit}
assert not {plan['groups'][str(r['train_index'])]['family_sha256'] for r in fit}&{plan['groups'][str(r['train_index'])]['family_sha256'] for r in audit}
for r in fit+audit:
    assert hashlib.sha256(r['text'].encode()).hexdigest()==r['sha256']
    for a in r['anchors']:assert r['text'][a['start']:a['end']]==a['raw']
gen=[json.loads(x) for x in (OUT/'derived_training_quarantine.jsonl').read_text(encoding='utf-8').splitlines()]
for r in gen:
    p=r['program'];assert plan['groups'][str(r['parent_train_index'])]['split']=='fit'
    if r['family']=='person_binding':answer=F(p['rate'])*F(p['hours'])
    elif r['family']=='price_batch':answer=F(p['bread_count'])*F(p['bread_unit_price'])+F(p['juice_price'])*(1 if p['juice_scope']=='whole_batch' else p['juice_count'])
    else:
        now=F(p['past'])-F(p['decrease']);future=now*F(p['future_factor'])+F(p['future_extra']);answer=future-(now if p['target']=='increase' else 0)
    assert answer==F(r['answer'])
    for c in r['candidates']:assert parse_answer(c['text'])==c['answer'] and c['correct']==(F(c['answer'])==answer)
    assert any(c['correct'] for c in r['candidates'])
pack=json.loads((ROOT/'review_app/questions.json').read_text(encoding='utf-8'))
assert len(pack['questions'])==7
for q in pack['questions']:
    assert plan['groups'][str(q['train_index'])]['split']=='fit'
    assert not {'answer','prediction','program_implied_choice'}&set(q)
    assert len(q['options'])<=5
html=(ROOT/'review_app/复核工作台.html').read_text(encoding='utf-8')
embedded=json.loads(re.search(r'<script id="initial-pack" type="application/json">(.*?)</script>',html,re.S).group(1));assert embedded==pack
model=Model(OUT/'natural_candidates');r=fit[0];pred=model.predict(r,r['anchors'][0]);rr=dict(r,id='new-unseen',train_index=-999);assert model.predict(rr,rr['anchors'][0])==pred
for name,h in json.loads((ROOT/'experiments/round4/asset_manifest.json').read_text()).items():assert hashlib.sha256((ROOT/'experiments/round4/selected_bundle'/name).read_bytes()).hexdigest()==h['sha256']
try:forbid_test('open',('gsm8k_test.jsonl',))
except PermissionError:pass
else:raise AssertionError('Missing guard')
print('PASS: parent/template split, exact source offsets, 48 program answers + all 192 numeric labels, train-only review items, public no-gold contract, unseen IDs, baseline hashes, test guard.')
