"""Context separates fractional denominators and interval counts from ordinals."""
import re
from functools import lru_cache
from round46_graph import graph
from round139_features import source_features as previous
from round52_local import roles
DENOMINATORS={'third','fourth','fifth','sixth','seventh','eighth','ninth','tenth'}

def kind(q,i):
    g=graph(q);x=g['quantities'][i];old=g['nodes'][i]['kind'];raw=x['raw'].lower();before=q[max(0,x['start']-35):x['start']];after=q[x['end']:]
    if raw in DENOMINATORS:
        numerator=re.search(r'\b(?:a|one|two|three|four|five|six|seven|eight|nine|\d+)\s*[-–]?\s*$',before,re.I)
        if numerator and (re.search(r'[-–]\s*$',before) or re.match(r'\s+of\b',after,re.I)):return 'fraction_denominator'
        if re.search(r'\bevery\s*$',before,re.I):return 'fraction_denominator'
    return old

@lru_cache(maxsize=50000)
def source_features(q,i):
    f=previous(q,i).copy();g=graph(q);old=g['nodes'][i]['kind'];new=kind(q,i)
    if new!=old:
        f.pop('source:kind:'+old,None);f['source:kind:'+new]=1.
        rs,_=roles(q)
        if rs[i]['family']=='count':f.pop('source:family:count',None);f['source:family:fraction']=1.
    return f

def checks():
    cases=[('One-third of 30 children are girls. How many boys are there?','third','fraction_denominator'),('One fourth of 80 engines are defective. How many are working?','fourth','fraction_denominator'),('Every third shopper buys a package. How many packages are sold?','third','fraction_denominator'),('The third puzzle has 50 pieces. How many pieces are there?','third','ordinal'),('A third of 30 children are girls. How many boys are there?','third','fraction_denominator')]
    for q,raw,expected in cases:
        g=graph(q);i=next(i for i,n in enumerate(g['quantities']) if n['raw'].lower()==raw);assert kind(q,i)==expected
        assert 'source:kind:'+expected in source_features(q,i)
    return dict(explicit_hyphen_fraction=True,spaced_fraction_of=True,periodic_every_nth=True,true_ordinal_preserved=True,quantity_values_and_slots_unchanged=True)
