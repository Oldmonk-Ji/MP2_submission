"""Manual sealed-audit labels; authored after model freeze, before predictions.
This file is analysis-only and must never be imported by training or inference.
"""
from data_access import ROOT
import json,hashlib
from round7_annotations import manual_rows,OUT

AUDIT={
3647:'C/N/cynthia A/N/NIL D/N/matthew D/N/matthew',
4185:'C/N/kerry C/N/NIL P/N/kerry/-1 T/N/kerry',
4528:'B/I/daisy B/N/daisy D/N/daisy D/N/daisy',
4436:'H/F/NIL R/F/peter R/N/peter A/N/peter\'s_mother',
548:'D/F/NIL R/F/NIL R/F/NIL R/F/NIL R/F/NIL H/F/NIL',
2126:'R/N/sarah D/N/sarah D/N/mark R/N/billy A/F/ana H/F/NIL',
2514:'R/N/speaker A/N/billy',
1086:'C/N/nalani C/N/nalani R/N/nalani P/N/nalani/-1',
2844:'C/F/jerry P/N/jerry/0 C/F/jerry P/N/jerry/2 C/F/jerry P/N/jerry/4 B/I/jerry',
614:'B/I/james D/N/james P/N/james/-1',
1335:'C/F/calum C/F/calum P/N/calum/1 B/I/calum',
1998:'P/N/lucas/-1 C/N/lucas C/N/lucas P/N/lucas/4 H/N/NIL H/N/NIL',
462:'C/N/luigi T/N/luigi C/N/luigi',
3486:'B/I/NIL P/N/NIL/2 H/N/NIL H/N/NIL H/N/NIL D/N/NIL',
1361:'O/I/marites P/I/marites/-1 O/F/marites O/F/marites O/F/marites R/N/marites O/F/marites D/N/marites O/F/marites O/F/marites',
3651:'T/N/emily C/N/emily T/N/emily',
3248:'H/N/NIL C/N/jackson C/N/jackson',
1991:'C/N/chad C/N/chad C/N/NIL C/N/NIL C/N/chad',
2495:'C/N/danny C/N/danny C/N/danny R/N/NIL',
3898:'C/N/NIL P/N/NIL/0 C/N/NIL P/N/NIL/2',
4515:'C/N/karen C/N/karen C/N/karen',
925:'H/N/greta\'s_turtle D/N/george\'s_turtle R/N/gloria\'s_turtle',
3087:'C/N/james T/N/james C/N/james R/N/james',
1980:'O/I/tree O/I/boy R/N/tree O/F/boy',
}
if __name__=='__main__':
    frozen=json.loads((OUT/'frozen.json').read_text())
    for name,h in frozen['sources'].items():assert hashlib.sha256((ROOT/'research'/name).read_bytes()).hexdigest()==h
    rows=manual_rows(AUDIT);plan=json.loads((OUT/'plan.json').read_text());assert set(AUDIT)=={i for g in plan['groups'].values() for i in g['sealed_audit']}
    seal=json.loads((OUT/'audit_seal.json').read_text())
    for r in rows:assert hashlib.sha256(r['question'].encode()).hexdigest()==seal['question_hashes'][str(r['index'])]
    (OUT/'audit_annotations.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8');print(dict(questions=len(rows),mentions=sum(len(r['labels']) for r in rows)))
