"""Let target meaning condition each source's local role, at fixed hash size."""
from functools import lru_cache
from round139_features import source_features as previous,matrix
from round162_decisions import packet as previous_packet
from round29_programs import quantities

@lru_cache(maxsize=50000)
def source_features(q,i):
    f=previous(q,i).copy();local=[(k,v) for k,v in f.items() if k.startswith(('source:','L:','R:','L2:','R2:'))];target=[(k,v) for k,v in f.items() if k.startswith('target:')]
    for a,x in local:
        for b,y in target:f['target_source:'+a+'|'+b]=x*y
    return f

def packet(row,programs):
    f,p,u,c=previous_packet(row,programs)
    return f,p,[source_features(row['question'],i) for i in range(len(quantities(row['question'])))],c

def checks():
    q='Paul has 20 books and gives 5 books to Mary. How many books does Paul have left?'
    a=source_features(q,0);b=source_features(q.replace('Paul','George').replace('Mary','Lucy'),0);assert a==b
    assert any(k.startswith('target_source:source:family:') for k in a)
    base=previous(q,0);assert all(a[k]==v for k,v in base.items());assert matrix([a]).shape==(1,16384)
    q2=q.replace('How many books does Paul have left?','How many books did Mary receive?');assert source_features(q2,0)!=a
    return dict(base_features_preserved=True,target_local_interactions_present=True,name_invariance=True,hash_dimension_unchanged=16384)
