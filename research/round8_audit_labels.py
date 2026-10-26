"""Fresh question-only audit labels, authored after model freeze, before predictions."""
from data_access import ROOT
from round7_annotations import manual_rows
import json,hashlib
OUT=ROOT/'experiments/round8'
MANUAL={
1781:'H/P/NIL T/P/jocelyn R/N/jocelyn',
4440:'R/N/thor R/N/captain_america D/N/ironman A/N/thor',
33:"A/P/speaker H/F/NIL A/F/speaker's_dog",
2643:'D/N/benjie A/N/benjie H/F/NIL',
4456:'C/N/NIL C/N/NIL D/N/ryan R/N/jason C/N/zachary P/N/zachary/4 C/N/NIL',
2339:'C/F/ted C/F/ted P/N/ted/0 P/N/ted/1 C/F/ted C/F/ted',
1247:'D/P/reggie C/P/reggie P/P/reggie/1',
1722:'P/F/aaron/1 H/F/NIL O/N/NIL R/F/aaron',
988:'D/N/tommy C/N/jimmy D/F/jimmy',
2923:'H/N/derek H/N/derek C/F/derek',
2089:'C/N/steve C/N/steve',
973:'C/I/neil R/N/neil',
}
assert (OUT/'frozen.json').exists()
rows=manual_rows(MANUAL)
plan=json.loads((OUT/'plan.json').read_text())
assert set(MANUAL)=={i for g in plan['groups'].values() for i in g['sealed_audit']}
for r in rows:
    r['ambiguity']=int(r['index']==4440)
    r['annotation_note']='Question-only, before viewing model predictions or reference answers. Stage follows literal past/future wording. First explicit matching-unit count used for repeated price denominators.'
    if r['index']==4440:r['annotation_note']='times older ambiguous between ratio and increment; role R remains identifiable. Multiword entity Captain America is annotated faithfully even if candidate generator cannot represent it.'
    if r['index']==33:r['annotation_note']='First-person narrator and dog are distinct owners; preserve unsupported gold owners rather than replacing them with NIL to inflate accuracy.'
path=OUT/'audit_annotations.json';path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'audit_seal.json').write_text(json.dumps(dict(labels_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_freeze_sha256=hashlib.sha256((OUT/'frozen.json').read_bytes()).hexdigest(),questions=len(rows),mentions=sum(len(r['labels']) for r in rows),source='same-assistant question-only annotation, not independently human-verified'),indent=2))
print('Sealed',len(rows),'fresh audit questions')
