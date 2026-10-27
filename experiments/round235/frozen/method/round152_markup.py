"""Recover strong-emphasis arithmetic before multiplication normalization."""
import re
from round51_programs import candidate_roots,normalized,value
from round44_programs import canonical,tree
from features import parse_answer
BOLD=re.compile(r'(?<![\w*])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![\w*])',re.S)

def clean(text):
    return BOLD.sub(r'\1',text)

def additions(q,text,g):
    if '**' not in text:return []
    changed=clean(text)
    if changed==text or parse_answer(changed)!=parse_answer(text):return []
    if normalized(changed)==normalized(text):return []
    return candidate_roots(q,changed,g)

def merged(q,text,g,old):
    roots={repr(canonical(tree(p))):tree(p) for p in old}
    for p,kind in additions(q,text,g):roots.setdefault(repr(canonical(tree(p))),tree(p))
    return list(roots.values())

def checks():
    from round46_graph import graph
    q='Ann buys 4 boxes of 2 pencils and loses 1 pencil. How many remain?';g=graph(q);text='She has **2 * 4 - 1 = 7 pencils**.\n#### 7'
    assert clean(text)=='She has 2 * 4 - 1 = 7 pencils.\n#### 7'
    assert clean('2**3 = 8 and 3 ** 2 = 9')=='2**3 = 8 and 3 ** 2 = 9'
    assert clean('**2**3 = 8**')=='2**3 = 8'
    roots=additions(q,text,g);assert roots and all(value(p,g['quantities'])==7 for p,k in roots)
    import json
    original=[p for p,k in roots];assert merged(q,text,g,original)==merged(q,text,g,json.loads(json.dumps(original)))
    assert len(merged(q,text,g,original))==len({repr(canonical(tree(p))) for p in original})
    assert parse_answer(text)==parse_answer(clean(text));return dict(bold_multiplication_recovered=True,powers_preserved=True,final_answer_unchanged=True,recovered_roots_execute_to_candidate_answer=True)
