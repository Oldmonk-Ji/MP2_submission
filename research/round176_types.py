"""Correct fraction/interval typing in both source and conditional-scope inputs."""
from round175_roles import kind,source_features,checks as type_checks
from round139_features import pair_features as previous_pair
from round162_decisions import packet as previous_packet
from round46_graph import graph
from round52_local import roles
from round29_programs import quantities

def revise_pair(q,i,features):
    f=features.copy();old=graph(q)['nodes'][i]['kind'];new=kind(q,i)
    if old==new:return f
    literal=next(k for k in f if k.startswith('literal:') and k.count(':')==1)
    f.pop('source:kind:'+old,None);f['source:kind:'+new]=1.;f.pop(literal+'|kind:'+old,None);f[literal+'|kind:'+new]=1.
    rs,_=roles(q)
    if rs[i]['family']=='count':
        f.pop('source:family:count',None);f['source:family:fraction']=1.;f.pop(literal+'|family:count',None);f[literal+'|family:fraction']=1.
    return f

def pair_features(q,i,c):return revise_pair(q,i,previous_pair(q,i,c))

def packet(row,programs):
    f,p,u,c=previous_packet(row,programs);q=row['question'];n=len(quantities(q));nc=len(c)
    ff=[revise_pair(q,j//nc,features) for j,features in enumerate(f)] if nc else []
    return ff,p,[source_features(q,i) for i in range(n)],c

def checks():
    report=type_checks();q='One-third of 30 children are girls. How many boys are there?';i=next(i for i,n in enumerate(quantities(q)) if n['raw']=='third');f=pair_features(q,i,'1')
    assert 'literal:1|kind:fraction_denominator' in f and 'literal:1|kind:ordinal' not in f;assert 'source:family:fraction' in f
    q='The third puzzle has 50 pieces. How many pieces are there?';assert pair_features(q,0,'1')==previous_pair(q,0,'1')
    report.update(scope_kind_cross_corrected=True,true_ordinal_features_unchanged=True);return report
