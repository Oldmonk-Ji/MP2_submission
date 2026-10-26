"""TRAIN-fit audit of representation consistency, independent of accuracy claims."""
from round22_common import ROOT,load,read,dump,sha
from round147_algebra import normalize,polynomial,checks,relation_map
from round138_literals import normalize as previous
from round101_templates import execute
from round51_programs import value
from round29_programs import quantities
from round109_alignment import mathematical_key
from round44_programs import tree
from collections import Counter,defaultdict
from fractions import Fraction as F
import numpy as np,time
OUT=ROOT/'experiments/round147'
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(checks=checks(),method='Exact bounded Laurent expansion, with old exact normalization fallback. No question numeric substitutions. Fit-only audit; no new hyperparameter search.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round147*.py')},test_opened=False))
    b,a,y,ids=load('train');ix={int(idx):i for i,idx in enumerate(ids)};refs=defaultdict(list);stats=Counter();rng=np.random.default_rng(147147)
    for r in read(ROOT/'experiments/round115/fit_sources.json'):
        assert r['fold']!=2;p=r['program'];idx=r['train_index'];refs[idx].append(p);n=len(quantities(b[ix[idx]]['question']));normal=normalize(p)
        try:polynomial(p);stats['expanded_reference_programs']+=1
        except ValueError:stats['fallback_reference_programs']+=1
        for _ in range(3):
            vals=[F(int(z)) for z in rng.integers(1,30,size=n)]
            try:assert execute(normal,vals)==value(p,[dict(value=str(v)) for v in vals]);stats['exact_execution_checks']+=1
            except ZeroDivisionError:stats['undefined_point_skips']+=1
    examples=[]
    for r in read(ROOT/'experiments/round143/fit_examples.json'):
        if r['kind']!='symbolically_equivalent_root_exists':continue
        p=r['matching_program'];key=mathematical_key(tree(p));same=[q for q in refs[r['train_index']] if mathematical_key(tree(q))==key]
        assert same;normal=normalize(p);matched=any(normal==normalize(q) for q in same);stats['known_equivalent_parents']+=1;stats['equivalent_forms_unified']+=matched
        examples.append(dict(train_index=r['train_index'],unified=matched))
    for r in read(ROOT/'experiments/round137/fit_examples.json'):
        p=normalize(r['correct']['program']);q=normalize(r['incorrect']['program']);stats['blindspot_pairs']+=1;stats['blindspot_literal_maps_distinguished']+=relation_map(p)!=relation_map(q)
    dump(OUT/'fit_examples.json',examples);dump(OUT/'fit_audit.json',dict(stats=stats,fit_only=True,seconds=time.perf_counter()-tick,test_opened=False));print(stats,flush=True)
