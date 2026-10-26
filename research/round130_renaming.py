"""Equivalent source-variable relabeling; language, values and answer labels unchanged."""
import re
from collections import defaultdict
from round103_features import question_tokens,local_tags,matrix,row_features as original
from round101_templates import canonical
from round29_programs import quantities

def remap(p,permutation):
    if p[0]=='N':return ('N',permutation[p[1]])
    if p[0]=='L':return tuple(p)
    return (p[0],*(remap(c,permutation) for c in p[1:]))

def features(row,packet,permutation):
    def key(k):
        m=re.match(r'^N(\d+):',k)
        return 'N'+str(permutation[int(m[1])])+k[m.end()-1:] if m else k
    words={key(k):v for k,v in question_tokens(row['question']).items()};out=[]
    for roots in packet:
        programs={}
        for p in roots:
            try:normal=canonical(remap(p,permutation));programs[repr(normal)]=normal
            except (ValueError,ZeroDivisionError):continue
        f=defaultdict(float)
        for p in programs.values():
            for t,v in local_tags(p).items():
                f[t]+=v/max(1,len(programs))
                for word,w in words.items():f[t+'|'+word]+=v*w*.2/max(1,len(programs))
        f['supported']=float(bool(programs));out.append(dict(f))
    return out

def build(bank,packets,reverse=True):
    rows=[]
    for row,packet in zip(bank,packets):
        n=len(quantities(row['question']));permutation=list(reversed(range(n))) if reverse else list(range(n))
        rows.extend(features(row,packet,permutation))
    return matrix(rows)

def checks():
    from round51_programs import value
    q='Robert has 25 balls and receives half of 40 balls.';row=dict(question=q);p=('ADD',('N',0),('MUL',('N',1),('N',2)));packet=[[p],[('SUB',('N',0),('N',2))],[],[p]]
    qs=quantities(q);n=len(qs);order=list(reversed(range(n)));mapped=[None]*n
    for i,j in enumerate(order):mapped[j]=qs[i]
    assert value(p,qs)==value(remap(p,order),mapped)
    assert features(row,packet,list(range(n)))==original(row,packet,'local')
    assert remap(remap(p,order),order)==p
    return dict(identity_mapping_reproduces_original_features=True,execution_value_invariant=True,reverse_mapping_involution=True)
