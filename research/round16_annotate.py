"""Single-assistant monetary-rate annotation, written before audit predictions.

Exact surface denominator for a monetary per-item/per-time amount, including
per-item discounts. '-' is a non-rate amount, '?' unresolved, '!' not a valid
numeric mention. Physical rates (hours/week) are outside this schema.
"""
from data_access import ROOT
import json,hashlib
OUT=ROOT/'experiments/round16'
LABELS={
'231:-1':'hour hour hour -',
'231:0':'hour - hour -',
'231:1':'', '231:2':'', '231:3':'',
'2005:-1':'month !',
'2005:0':'- !','2005:1':'- -','2005:2':'- !','2005:3':'- !',
'3915:-1':'- ? erasers',
'3915:0':'- -','3915:1':'-','3915:2':'-','3915:3':'- ? - !',
'1553:-1':'Uno_Giant_Family_Card pieces',
'1553:0':'- Uno_Giant_Family_Card card !',
'1553:1':'- Uno_Giant_Family_Card card -',
'1553:2':'Uno_Giant_Family_Card piece -',
'1553:3':'- Uno_Giant_Family_Card card',
'2016:-1':'- pound -',
'2016:0':'- pound - -',
'2016:1':'pound pound - -',
'2016:2':'- pound - -',
'2016:3':'- pound pound -',
'895:-1':'- hour -',
'895:0':'- hour -',
'895:1':'- - -',
'895:2':'- hour -',
'895:3':'hour - - !',
'4609:-1':'- -',
'4609:0':'- -','4609:1':'- - - -','4609:2':'-','4609:3':'- - - !',
'3543:-1':'- - !',
'3543:0':'- -','3543:1':'- - female_mother',
'3543:2':'- - - !','3543:3':'- - -',
}

def main():
    assert (OUT/'frozen.json').exists()
    assert not (OUT/'audit_annotation_seal.json').exists(),'Do not overwrite a sealed annotation.'
    rows=json.loads((OUT/'audit_packet.json').read_text(encoding='utf-8'))
    assert set(LABELS)=={r['id'] for r in rows}
    for r in rows:
        labels=LABELS[r['id']].split();assert len(labels)==len(r['anchors']),r['id']
        r['labels']=[]
        for a,value in zip(r['anchors'],labels):
            value=value.replace('_',' ')
            if value not in ('!','?','-'):assert value.lower() in (r['text']+' '+r['question']).lower(),(r['id'],value)
            r['labels'].append(dict(valid=value!='!',price_scope=None if value in ('!','?') else value))
        r['annotation_source']='same-assistant text review before predictions; no references; not human gold'
    content=json.dumps(rows,ensure_ascii=False,indent=2)
    (OUT/'audit_annotations.json').write_text(content,encoding='utf-8')
    (OUT/'audit_annotation_seal.json').write_text(json.dumps(dict(sha256=hashlib.sha256(content.encode()).hexdigest(),
        frozen_sha256=hashlib.sha256((OUT/'frozen.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        uncertainties=['3915: $5 each could apply to notebooks only or both sharpeners and notebooks. Mask both source and matching candidate claim.'],
        schema='monetary rate denominator, including unit discounts; nonmonetary ratios outside scope'),indent=2))
    print('Sealed',len(rows),'text annotations before prediction.')

if __name__=='__main__':main()
