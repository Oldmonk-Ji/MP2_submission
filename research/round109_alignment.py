"""TRAIN-only reference-equivalent root alignment, with exact rational identities."""
from functools import lru_cache
from fractions import Fraction as F
from collections import defaultdict
import numpy as np
from round101_templates import canonical
from round84_algebra import expression
from round44_programs import tree
from round51_programs import value
from round29_programs import quantities
from features import parse_answer

@lru_cache(maxsize=65536)
def mathematical_key(p):
    import sympy as sp
    return str(sp.cancel(expression(p,sp)))

def annotate(bank,ids,packets,ptr,sources,excluded,keep,progress=None):
    index={int(k):i for i,k in enumerate(ids)};reference=defaultdict(set);source_count=0
    for r in sources:
        idx=r['train_index'];i=index[idx]
        if idx in excluded or not keep[i]:continue
        try:reference[idx].add(mathematical_key(tree(r['program'])));source_count+=1
        except (ValueError,ZeroDivisionError):continue
    gold=np.zeros(int(ptr[-1]),bool);active=np.zeros(len(bank)*4,bool);parents=set();candidates=0;support=0;gold_roots=0
    for i in np.flatnonzero(keep):
        idx=int(ids[i])
        if idx not in reference:continue
        for j,roots in enumerate(packets[i]):
            # A reference-equivalent program necessarily has the correct value;
            # final candidate labels are not required to pick its operator graph.
            programs={}
            for p in roots:
                try:programs[repr(canonical(p))]=p
                except (ValueError,ZeroDivisionError):continue
            c=4*i+j;assert len(programs)==ptr[c+1]-ptr[c]
            for k,p in enumerate(programs.values()):
                if mathematical_key(tree(p)) in reference[idx]:gold[ptr[c]+k]=True
            g=int(gold[ptr[c]:ptr[c+1]].sum());support+=g>0;gold_roots+=g
            if 0<g<len(programs):active[c]=True;parents.add(idx);candidates+=1
        if progress is not None and i%500==0:progress(i)
    return gold,active,dict(reference_parents=len(reference),reference_programs=source_count,aligned_ambiguous_parents=len(parents),aligned_ambiguous_candidates=candidates,candidates_with_equivalent_reference=support,reference_equivalent_roots=gold_roots,alignment_parent_ids=sorted(parents))

def checks():
    a=('MUL',('N',0),('ADD',('N',1),('N',2)));b=('ADD',('MUL',('N',0),('N',1)),('MUL',('N',0),('N',2)))
    assert mathematical_key(a)==mathematical_key(b)
    assert mathematical_key(('SUB',('N',0),('N',1)))!=mathematical_key(('SUB',('N',1),('N',0)))
    return dict(distributive_equivalence=True,operand_direction_distinguished=True)
