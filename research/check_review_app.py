"""Check review-pack isolation and returned-answer validation using temporary QA data."""
from data_access import ROOT
import json,hashlib,tempfile,sys,contextlib,io
from pathlib import Path
import import_review_answers as importer

packfile=ROOT/'review_app/questions_第一批.json'
if not packfile.exists():packfile=ROOT/'review_app/questions.json'
pack=json.loads(packfile.read_text(encoding='utf-8'))
manifest=ROOT/'experiments/human_review/review_pack01_manifest.json'
plan=json.loads((ROOT/'experiments/round7/plan.json').read_text())
sealed={i for g in plan['groups'].values() for i in g['sealed_audit']}
assert pack['split']=='train' and len(pack['questions'])==7
for q in pack['questions']:
    assert q['train_index'] not in sealed
    assert set(q)=={'id','train_index','question_sha256','original','translation','title','category','prompt','options'}
    assert hashlib.sha256(q['original'].encode()).hexdigest()==q['question_sha256']
    assert all(set(o)=={'id','text'} for o in q['options'])
triage=json.loads((ROOT/'experiments/human_review/triage01_private.json').read_text(encoding='utf-8'))
assert len({r['train_index'] for r in triage['dispositions']})==56
q=pack['questions'][0]
data=dict(schema='mp2-review-answers/v1',pack_id=pack['id'],pack_version=1,split='train',answers=[dict(id=q['id'],question_sha256=q['question_sha256'],status='answered',choice='U',note='QA fixture only')])
with tempfile.TemporaryDirectory(prefix='mp2-review-check-') as directory:
    root=Path(directory);(root/'experiments/human_review').mkdir(parents=True)
    importer.ROOT=root
    fixture=root/'answers.json'
    def run(value):
        fixture.write_text(json.dumps(value),encoding='utf-8')
        sys.argv=['import_review_answers.py',str(fixture),'--manifest',str(manifest)]
        with contextlib.redirect_stdout(io.StringIO()):importer.main()
    run(data)
    output=list((root/'experiments/human_review/returned').glob('*.json'))
    assert len(output)==1 and 'not automatically promoted' in json.loads(output[0].read_text())['status']
    for bad in [dict(data,split='test'),dict(data,pack_version=2),dict(data,answers=[dict(data['answers'][0],question_sha256='0'*64)]),dict(data,answers=[dict(data['answers'][0],choice='INVALID')]),dict(data,answers=data['answers']*2)]:
        try:run(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid return accepted')
print('PASS: pack isolation, source hashes, triage coverage, valid return, and five invalid-return cases. No real human response written.')
