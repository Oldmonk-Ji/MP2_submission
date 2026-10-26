"""Target-relative named-entity relations; proximity is soft evidence, not ownership gold."""
import re,math
from collections import Counter,defaultdict
from round122_features import question_tokens as identity_tokens,EXCLUDE
from round103_features import local_tags,matrix
from round101_templates import canonical
from round29_programs import quantities

def question_tokens(q):
    out=identity_tokens(q);names={m.group().lower() for m in re.finditer(r'\b[A-Z][a-z]+\b',q)}-EXCLUDE
    target=re.split(r'[.?!]',q.rstrip('? .'))[-1];requested=names&set(re.findall(r'[a-z]+',target.lower()))
    if not names or not requested:return out
    def words(s):return [('target_entity' if w in requested else 'other_entity') if w in names else w for w in re.findall(r'[a-z]+',s.lower())]
    marked={'target_entity','other_entity'};counts=Counter();ws=words(q)
    for width in (2,3):
        for k in range(len(ws)-width+1):
            part=ws[k:k+width]
            if marked.intersection(part):counts['relative'+str(width)+':'+':'.join(part)]+=1
    for i,n in enumerate(quantities(q)):
        left=q[:n['start']];right=q[n['end']:]
        for side,part in [('L',words(left)[-3:]),('R',words(right)[:3])]:
            for w in part:
                if w in marked:counts[f'N{i}:{side}:{w}']+=1
        # Record only the nearest explicitly named actor in the sentence.
        # No pronoun replacement and no assertion that this actor owns the quantity.
        sentence=re.split(r'[.!?;]',left)[-1];actors=[w for w in words(sentence) if w in marked]
        if actors:counts[f'N{i}:last_explicit_actor:'+actors[-1]]+=1
    out.update({k:1.+math.log(v) for k,v in counts.items()});return out

def row_features(row,packet,arm='local'):
    assert arm=='local';words=question_tokens(row['question']);result=[]
    for roots in packet:
        programs={}
        for p in roots:
            try:normal=canonical(p);programs[repr(normal)]=normal
            except (ValueError,ZeroDivisionError):continue
        f=defaultdict(float)
        for p in programs.values():
            for t,v in local_tags(p).items():
                f[t]+=v/max(1,len(programs))
                for word,w in words.items():f[t+'|'+word]+=v*w*.2/max(1,len(programs))
        f['supported']=float(bool(programs));result.append(dict(f))
    return result

def build(bank,packets,arm='local'):return matrix([d for row,packet in zip(bank,packets) for d in row_features(row,packet,arm)])

def checks():
    q='Robert had 25 balls. Tim gave him half of his 40 balls. How many balls does Robert have now?'
    a=question_tokens(q);b=question_tokens(q.replace('does Robert','does Tim'))
    assert 'N0:last_explicit_actor:target_entity' in a and 'N0:last_explicit_actor:other_entity' in b
    assert 'relative2:other_entity:gave' in a and 'relative2:target_entity:gave' in b
    assert a==question_tokens(q.replace('Robert','Zachary').replace('Tim','Adam'))
    plain='There are 10 boxes with 5 balls in each box. How many balls are there?'
    assert question_tokens(plain)==identity_tokens(plain)
    return dict(target_binding_distinguished=True,giver_target_relation_distinguished=True,renaming_invariant=True,no_named_target_leaves_representation_unchanged=True,no_pronoun_resolution=True)
