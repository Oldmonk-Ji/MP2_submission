"""Pre-model sampling correction; retain first packet and partition unchanged."""
from data_access import ROOT
from round15_prepare import anchors
import json,hashlib
OUT=ROOT/'experiments/round15'
assert not (OUT/'frozen.json').exists()
plan_path=OUT/'plan.json';plan=json.loads(plan_path.read_text())
assert 'sampling_revision' not in plan
(OUT/'plan_initial.json').write_bytes(plan_path.read_bytes())
for split in ('fit','audit'):
    path=OUT/f'{split}_annotation_packet.json';raw=path.read_bytes();(OUT/f'{split}_packet_initial.json').write_bytes(raw)
    rows=json.loads(raw)
    for r in rows:r['anchors']=anchors(r['text'])
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    if split=='fit':
        for r in rows:print(json.dumps(dict(id=r['id'],anchors=[dict(raw=a['raw'],context=a['context']) for a in r['anchors']]),ensure_ascii=True))
plan['anchor_scope']='First and last eligible verbal quantity mentions, plus up to two additional currency mentions; at most four per selected text. Step headings are retained as negative extraction examples when detected.'
plan['label_fields']=['valid_quantity','owner','stage','price_scope']
plan['sampling_revision']=dict(reason='Initial fit inspection showed missing middle-of-sentence prices and heading numbers; broadened annotation before any model fitting or audit inspection.',initial_plan_sha256=hashlib.sha256((OUT/'plan_initial.json').read_bytes()).hexdigest())
plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
