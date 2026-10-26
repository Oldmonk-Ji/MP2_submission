"""Natural-text annotations made before fitting/predictions; no answer labels.

Owner is the named quantity holder/actor (NIL for unnamed/groups); stage is
N unspecified/current, P explicitly past, F explicitly future, I/E story
initial/final, X cross-stage difference, D time-duration itself, O math constant.
Scope is an exact surface noun phrase for a per-unit price, '-' when not a rate,
'?' when the field genuinely cannot be resolved. Invalid split fractions and
step numbers are negative extraction examples. This is assistant-authored gold,
not independent human gold; unknowns never silently become supervised labels.
"""
from data_access import ROOT
import json,hashlib
OUT=ROOT/'experiments/round15'

# Each record annotates all frozen sampled anchors in source order.
# owner / stage / price-scope. Spaces in scope and owner use underscores.
FIT={
'1060:-1':'jerry/N/- jerry/N/-',
'1060:0':'jerry/I/- jerry/E/-',
'1060:1':'jerry/N/- jerry/N/-',
'4426:-1':'charley/N/- charley/N/-',
'4426:0':'charley/N/- INVALID',
'4426:1':'charley/N/- charley/N/-',
'4124:-1':'viggo/N/- viggo/N/- ?/N/- ?/N/-',
'4124:0':'viggo/N/- viggo/N/- viggo/N/- viggo/N/-',
'4124:1':'viggo/N/- viggo/N/- viggo/N/- viggo/N/-',
'3993:-1':'greta/N/- greta/N/hour lisa/N/hour greta/N/-',
'3993:0':'greta/N/hour lisa/F/-',
'3993:1':'greta/N/- greta/N/hour lisa/F/-',
'3962:-1':'reynald/N/- reynald/N/-',
'3962:0':'reynald/N/- reynald/N/-',
'3962:1':'reynald/N/- reynald/N/-',
'999:-1':'NIL/N/- NIL/N/Student_tickets NIL/N/adult_tickets',
'999:0':'NIL/N/- NIL/N/- NIL/N/-',
'999:2':'INVALID INVALID',
'1292:-1':'NIL/N/- NIL/N/meal elisa/I/-',
'1292:0':'NIL/N/- NIL/N/meals NIL/N/meal NIL/N/-',
'1292:1':'NIL/N/- NIL/N/meals elisa/E/- NIL/N/-',
'1721:-1':'nicky/I/- nicky/I/cards nicky/E/card',
'1721:0':'nicky/I/- nicky/I/cards nicky/E/card nicky/E/-',
'1721:1':'nicky/I/- nicky/I/cards nicky/E/card nicky/E/-',
'2553:-1':'wyatt/I/- NIL/N/loaf_of_bread NIL/N/carton_of_orange_juice',
'2553:2':'wyatt/N/- NIL/N/loaves_of_bread NIL/N/loaves_of_bread wyatt/N/-',
'2553:3':'wyatt/N/- NIL/N/loaves_of_bread NIL/N/loaves_of_bread wyatt/N/-',
'4044:-1':'NIL/N/- mila/N/hour agnes/N/hour agnes/N/-',
'4044:0':'agnes/N/- INVALID',
'4044:1':'agnes/N/- agnes/N/hour mila/N/hour INVALID',
'1691:-1':'evan/X/- NIL/D/-',
'1691:0':'evan/X/- evan/F/-',
'1691:1':'evan/X/- INVALID',
'2831:-1':'NIL/N/- NIL/D/-',
'2831:0':'NIL/P/- NIL/D/-',
'2831:1':'NIL/D/- NIL/F/-',
'1791:-1':'kate/I/- NIL/E/-',
'1791:0':'kate/I/- NIL/E/-',
'1791:2':'kate/N/- NIL/O/-',
'777:-1':'NIL/N/- NIL/N/-',
'777:1':'NIL/D/- NIL/D/-',
'777:2':'NIL/D/- NIL/N/- NIL/N/- NIL/D/-',
'1627:-1':'NIL/N/- NIL/N/orchids NIL/N/? NIL/N/-',
'1627:0':'NIL/N/- NIL/N/orchids NIL/N/orchids NIL/N/-',
'1627:2':'NIL/N/- NIL/N/potted_Chinese_money_plants NIL/N/- NIL/N/-',
}


def build(split,mapping):
    rows=json.loads((OUT/f'{split}_annotation_packet.json').read_text(encoding='utf-8'))
    assert set(mapping)=={r['id'] for r in rows}
    for r in rows:
        entries=mapping[r['id']].split();assert len(entries)==len(r['anchors']),(r['id'],len(entries),len(r['anchors']))
        labels=[]
        for a,entry in zip(r['anchors'],entries):
            assert r['text'][a['start']:a['end']]==a['raw']
            if entry=='INVALID':label=dict(valid=False,owner=None,stage=None,price_scope=None)
            else:
                owner,stage,scope=entry.split('/');scope=scope.replace('_',' ')
                label=dict(valid=True,owner=None if owner=='?' else owner.replace('_',' '),stage=None if stage=='?' else stage,price_scope=None if scope=='?' else scope)
                if scope not in ('-','?'):
                    assert scope.lower() in (r['text']+' '+r['question']).lower(),(r['id'],scope)
            labels.append(label)
        r['labels']=labels;r['annotation_source']='assistant text-only review, before model predictions; no reference answers'
    (OUT/f'{split}_annotations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(split=split,parents=len({r['train_index'] for r in rows}),texts=len(rows),anchors=sum(len(r['anchors']) for r in rows),invalid=sum(not x['valid'] for r in rows for x in r['labels']),masked_fields=sum(v is None for r in rows for x in r['labels'] if x['valid'] for v in (x['owner'],x['stage'],x['price_scope'])))))


if __name__=='__main__':build('fit',FIT)
