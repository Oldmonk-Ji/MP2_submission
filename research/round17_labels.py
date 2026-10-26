"""Source-span annotations for object, predicate and numeric role.

Objects are lexical heads, NOT fully resolved batches/times. Multipliers retain
their predicate (quantity vs price vs money), but RHS binding is unverified.
Unknowns are masked; no candidate correctness or reference answers are read.
"""
from data_access import ROOT
from pathlib import Path
import json,re,hashlib
OUT=ROOT/'experiments/round17'
ROLES={'C':('count','quantity'),'P':('unit_price','unit_price'),
       'U':('unit_price','unit_profit'),'S':('unit_price','unit_saving'),'MQ':('multiplier','quantity'),
       'MP':('multiplier','unit_price'),'MA':('multiplier','amount'),
       'MU':('multiplier','unit_profit'),'T':('total','amount'),
       'G':('total','profit'),'O':('other','other')}

def norm(s):
    s=s.lower().replace('’',"'").removesuffix("'s")
    if len(s)>4 and s.endswith('ies'):s=s[:-3]+'y'
    elif len(s)>3 and s.endswith('s') and not s.endswith(('ss','us')):s=s[:-1]
    # Documented spelling variants in FIT parent 2303, no audit-derived alias.
    if s=='lilly':s='lily'
    return s

FIT={
'330:-1':'manuscript:C page:P manuscript:P page:C manuscript:C',
'330:1':'manuscript:C page:C page:P manuscript:P ! manuscript:C ! !',
'330:3':'manuscript:C manuscript:C page:C manuscript:C page:C page:C manuscript:P manuscript:T',
'3019:-1':'month:O vacation:T paycheck:O',
'3019:0':'vacation:T month:O paycheck:O month:S paycheck:O paycheck:S',
'3019:1':'vacation:T month:O paycheck:O ! month:O paycheck:C vacation:T !',
'2303:-1':'rose:C lily:MQ rose:P lily:MP',
'2303:0':'rose:C lily:P rose:P lily:P rose:C lily:P lily:T !',
'2303:1':'! ! rose:C lily:C rose:P lily:P ! !',
'4040:-1':'natasha:MA carla:MA natasha:T ?:C good:MA',
'4040:0':'natasha:MA carla:MA natasha:MA natasha:T good:T good:T good:T !',
'4040:1':'natasha:MA natasha:MA natasha:T natasha:MA carla:MA carla:MA good:T good:G',
'201:-1':'fabric:P pattern:T thread:C thread:P ?:T',
'201:0':'thread:P fabric:P ?:T',
'201:1':'fabric:P thread:P thread:C ! fabric:P fabric:P ! !',
'3059:-1':'bed:O bed:P specialist:P specialist:C specialist:O ambulance:T',
'3059:0':'bed:P bed:O bed:T specialist:P specialist:O specialist:C specialist:T',
'3059:1':'bed:P bed:O bed:T specialist:P specialist:O specialist:O specialist:T',
'3886:-1':'sock:C sock:P sock:MU sock:C sock:U sock:C',
'3886:0':'sock:C sock:P sock:P sock:T sock:? sock:U sock:C sock:G',
'3886:1':'sock:C sock:? sock:T sock:U sock:P sock:T sock:C sock:G',
'1673:-1':'car:C car:O car:O car:O car:O car:C gas:P',
'1673:0':'car:C ! car:O ! gas:P gas:T car:C gas:T',
'1673:1':'car:C car:O car:O gas:P car:O car:C gas:T gas:T',
}

def write_labels(split,mapping):
    packet=OUT/f'{split}_packet.json';rows=json.loads(packet.read_text(encoding='utf-8'))
    assert set(mapping)=={r['id'] for r in rows}
    for r in rows:
        entries=mapping[r['id']].split();assert len(entries)==len(r['anchors']),(r['id'],len(entries),len(r['anchors']))
        labels=[]
        for a,e in zip(r['anchors'],entries):
            if e=='!':g=dict(valid=False,object=None,role=None,predicate=None)
            else:
                obj,code=e.split(':');role,predicate=ROLES[code] if code!='?' else (None,None)
                g=dict(valid=True,object=obj if obj!='?' else None,role=role,predicate=predicate)
            # Locate a lexical object evidence span without fabricating a new
            # explanation. This does not certify the inferred coreference.
            ev=[]
            if g['object']:
                for source,text in [('claim',r['text']),('question',r['question'])]:
                    hits=[m for m in re.finditer(r"[A-Za-z]+(?:['’]s)?",text) if norm(m.group())==g['object']]
                    if hits:
                        m=min(hits,key=lambda m:abs(m.start()-a['start'])) if source=='claim' else hits[0]
                        ev.append(dict(source=source,start=m.start(),end=m.end(),text=m.group()))
                assert ev,(r['id'],a['raw'],g['object'])
            g['object_lexical_evidence']=ev;g['identity_qualifiers_verified']=False
            labels.append(g)
        r['labels']=labels;r['annotation_source']='same assistant text review before predictions; lexical object heads, not independent human gold'
    path=OUT/f'{split}_annotations.json';assert not path.exists()
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    summary=dict(parents=len({r['train_index'] for r in rows}),texts=len(rows),anchors=sum(len(r['anchors']) for r in rows),
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(),packet_sha256=hashlib.sha256(packet.read_bytes()).hexdigest())
    (OUT/f'{split}_label_seal.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':write_labels('fit',FIT)
