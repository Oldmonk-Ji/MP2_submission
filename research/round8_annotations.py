"""Expanded train-only bindings; preserve disputed labels with explicit masks."""
from data_access import ROOT
from round7_annotations import manual_rows
import json
OUT=ROOT/'experiments/round8'
MANUAL={
1584:'D/N/arthur D/N/tom A/N/bob A/N/susan C/N/NIL',
721:'R/N/sam R/N/kendra A/N/kendra H/F/NIL',
3158:'D/N/ryanne A/N/NIL',
3226:'A/N/NIL R/N/dante D/N/maria',
1561:'C/N/NIL R/N/jim C/N/jim C/N/jim H/F/NIL',
4049:'R/F/christina A/F/NIL H/F/NIL R/F/oscar H/F/NIL',
165:'A/N/dog_1 D/N/dog_2 D/N/dog_3 R/N/dog_4 D/N/dog_5',
3489:'D/N/jessica H/F/NIL A/F/claire',
1549:'C/N/peter C/N/NIL P/N/peter/-1',
1402:'T/I/robi R/I/rudy R/E/NIL',
1864:'C/N/NIL C/N/NIL P/N/NIL/1 T/F/NIL',
3251:'C/N/yanna P/N/yanna/0 C/N/yanna P/N/yanna/2 T/N/yanna',
3136:'C/N/travis P/N/travis/0 C/N/travis H/N/NIL',
2549:'C/N/jesiah P/N/jesiah/-1 C/N/jesiah P/N/jesiah/2',
3909:'R/N/NIL R/N/NIL T/I/NIL',
2252:'T/F/trey H/F/NIL P/F/trey/-1',
2300:'C/N/cheryl H/N/NIL',
4310:'C/N/joel C/N/joel C/N/joel C/N/joel R/N/joel C/E/joel',
270:'C/I/haruto D/N/haruto R/N/haruto',
2399:'H/N/NIL H/N/NIL H/N/NIL H/N/NIL H/N/NIL',
2782:'D/N/lizzy D/N/lizzy D/N/lizzy D/N/lizzy',
1820:'T/P/larry D/P/larry B/N/larry',
2760:'C/I/adrianna D/N/adrianna C/N/NIL',
4493:'B/I/lydia R/F/lydia',
}

def main():
    old=json.loads((ROOT/'experiments/round7/training_annotations.json').read_text(encoding='utf-8'))
    rows=old+manual_rows(MANUAL)
    plan=json.loads((OUT/'plan.json').read_text())
    assert set(MANUAL)=={i for g in plan['groups'].values() for i in g['train']}
    assert not {r['index'] for r in rows}&{i for g in plan['groups'].values() for i in g['sealed_audit']}
    # Only affected mentions are masked. Explicit facts in those questions survive.
    disputed={1922:[0],3711:[3],3790:[1],1585:[1]}
    for r in rows:
        r['original_mask']=r['mask'].copy()
        r['disputed_mentions']=disputed.get(r['index'],[])
        for i in r['disputed_mentions']:r['mask'][i]=False
        r['annotation_note']='Source occurrence labels; NIL for unnamed/group owners; rate scope is an explicit count of matching units, not a volume denominator.'
        if r['index']==3251:
            r['mask'][4]=False
            r['annotation_note']='one hundred dollar bill is 100 but frozen quantity parser emits one=1; do not train that corrupted mention. Parser limitation retained in diagnostics.'
        if r['index']==4310:
            r['annotation_note']='Owner denotes current collector Joel, not original donor. Target ownership is ambiguous and sent for human review; target is outside reduced mention schema.'
        if r['index']==2549:
            r['annotation_note']='Explicit milk rates and expenses are clear; gross versus net income target is sent for human review.'
    (OUT/'training_annotations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    # These are screening judgments, not universally true linguistic labels.
    ambiguous={1922,3711,3790,1585,2549,4310}
    screen=[dict(index=r['index'],question=r['question'],ambiguity=int(r['index'] in ambiguous),source='human disagreement / assistant target ambiguity' if r['index'] in ambiguous else 'assistant assessed usable interpretation; not independent human gold') for r in rows]
    (OUT/'ambiguity_training.json').write_text(json.dumps(screen,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(questions=len(rows),new_questions=len(MANUAL),supervised_mentions=sum(sum(r['mask']) for r in rows),disputed_masked=sum(len(r['disputed_mentions']) for r in rows))))
if __name__=='__main__':main()
