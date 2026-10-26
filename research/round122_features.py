"""Preserve within-question entity identity while anonymizing actual names."""
import re,math
from collections import Counter,defaultdict
from round103_features import STOP,local_tags,matrix
from round101_templates import canonical
from round29_programs import quantities
EXCLUDE={'a','an','the','if','when','after','before','each','every','how','what','there','in','on','at'}

def question_tokens(q):
    names={}
    for m in re.finditer(r'\b[A-Z][a-z]+\b',q):
        key=m.group().lower()
        if key not in EXCLUDE and key not in names:names[key]='person'+str(len(names))
    def words(s):return [names.get(t,t) for t in re.findall(r'[a-z]+',s.lower())]
    tokens=Counter('q:'+w for w in words(q) if w not in STOP)
    target=re.split(r'[.?!]',q.rstrip('? .'))[-1];tail=words(target)
    tokens.update('t:'+w for w in tail if w not in STOP);tokens.update('t2:'+a+'_'+b for a,b in zip(tail,tail[1:]))
    for i,x in enumerate(quantities(q)):
        for side,ws in [('L',words(q[:x['start']])[-3:]),('R',words(q[x['end']:])[:3])]:tokens.update(f'N{i}:{side}:{w}' for w in ws if w not in STOP)
    return {k:1.+math.log(v) for k,v in tokens.items()}

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
    from round103_features import question_tokens as old
    q='Robert had 25 balls. Tim gave him half of his 40 balls. How many balls does Robert have now?'
    other=q.replace('does Robert','does Tim');renamed=q.replace('Robert','Zachary').replace('Tim','Adam')
    assert old(q)==old(other);assert question_tokens(q)!=question_tokens(other)
    assert question_tokens(q)==question_tokens(renamed)
    return dict(old_representation_target_identity_collision=True,new_representation_distinguishes_target=True,arbitrary_name_renaming_invariant=True,diagnostic_source='TRAIN-fit question3100; counterfactual used only for a feature check, never as added training data')
