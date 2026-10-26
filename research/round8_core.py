"""Learned linking features for clause subjects, pronouns, and temporal scopes.
No hard-coded training IDs, answers, or entity names.
"""
import re
import round7_core as old
quantities=old.quantities

def entities(q):
    result=old.entities(q)
    for m in re.finditer(r'\b([1-5])(?:st|nd|rd|th) fastest dog\b',q,re.I):
        result.setdefault('dog '+m[1],[]).append(m.start())
    return result

def segment(q,pos):
    # Decimal points and Mr./Mrs. are not sentence boundaries.
    boundaries=[0]+[m.end() for m in re.finditer(r'(?<!Mr)(?<!Mrs)[.!?](?=\s+[A-Z]|$)',q)]
    start=max(x for x in boundaries if x<=pos)
    end=min([x for x in boundaries if x>pos]+[len(q)])
    return start,end

def qfeatures(q,qs,i,context=True):
    f=old.qfeatures(q,qs,i,context);pos=qs[i]['start'];a,b=segment(q,pos);s=q[a:b].lower();left=q[a:pos].lower()
    f['sentence_initial']=a==0
    for cue in ['ago','will','now','currently','was','had','next','last','each','per','remaining','more','than','twice','half']:
        f['sentence:'+cue]=bool(re.search(r'\b'+cue+r'\b',s))
        f['preceding:'+cue]=bool(re.search(r'\b'+cue+r'\b',left))
    f['future_in_years']=bool(re.search(r'\bin\b.{0,30}\byears?\b',s))
    return f

def owner_features(q,qs,i,entity,context=True):
    base_entity=entity if entity in old.entities(q) else 'NIL'
    f=old.owner_features(q,qs,i,base_entity,context)
    f.update(qfeatures(q,qs,i,context));es=entities(q);pos=qs[i]['start'];a,b=segment(q,pos)
    positions=es[entity];f['nil']=entity=='NIL';f['ordinal_owner']=entity.startswith('dog ')
    names=[(p,n) for n,ps in es.items() if n!='NIL' and "'s" not in n for p in ps if p>=0]
    within=sorted((p,n) for p,n in names if a<=p<b)
    before=sorted((p,n) for p,n in names if p<a)
    prefix=q[a:pos].lower()
    f['sentence_first_owner']=bool(within) and within[0][1]==entity
    f['previous_sentence_last_owner']=bool(before) and before[-1][1]==entity
    f['document_first_owner']=bool(names) and min(names)[1]==entity
    f['pronoun_sentence']=bool(re.match(r'\s*(he|she|his|her|they|their)\b',prefix))
    f['pronoun_x_document_first']=f['pronoun_sentence'] and f['document_first_owner']
    f['pronoun_x_previous_owner']=f['pronoun_sentence'] and f['previous_sentence_last_owner']
    f['same_sentence_v2']=any(a<=p<b for p in positions)
    if entity!='NIL':
        p=min(positions,key=lambda p:abs(p-pos));between=q[min(p,pos):max(p,pos)].lower()
        f['comparison_intervening']=bool(re.search(r'\b(than|as)\b',between))
        f['owner_before_quantity']=p<pos
        f['owner_after_quantity']=p>pos
    return f

def scope_features(q,qs,i,j,context=True):
    f=old.scope_features(q,qs,i,j,context)
    if j>=0:
        a,b=segment(q,qs[i]['start']);f['same_sentence_v2']=a<=qs[j]['start']<b
        f['count_unit_next']=' '.join(old.lex(q[qs[j]['end']:])[:2])
        f['rate_unit_next']=' '.join(old.lex(q[qs[i]['end']:])[:3])
    return f
