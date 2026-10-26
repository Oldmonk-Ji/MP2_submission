"""Separate missing graphs from false gaps caused by equivalent formula representations."""
from round22_common import ROOT,load,read,dump,sha
from round138_literals import normalize,relation_map,ABSENT
from round109_alignment import mathematical_key
from round44_programs import tree
from collections import defaultdict,Counter
import numpy as np,time
OUT=ROOT/'experiments/round143'
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();b,a,y,ids=load('train');packets=read(ROOT/'experiments/round102/train_programs.json');table=defaultdict(dict);refs=defaultdict(list)
    for r in read(ROOT/'experiments/round138/fit_labels.json'):table[r['train_index']][(r['quantity'],r['constant'])]=r['label']
    for r in read(ROOT/'experiments/round115/fit_sources.json'):refs[r['train_index']].append(r['program']);assert r['fold']!=2
    stats=Counter();examples=[]
    for i,idx in enumerate(ids):
        idx=int(idx);ann=table.get(idx)
        if not ann or y[i].min()>0:continue
        correct=[];compatible=False
        for j in np.flatnonzero(y[i]):
            for p in packets[i][j]:
                try:r,cs=relation_map(normalize(p))
                except (ValueError,ZeroDivisionError):continue
                compatible|=all(r.get(key,ABSENT)==label for key,label in ann.items());correct.append((int(j),p))
        if compatible:continue
        gold={mathematical_key(tree(p)) for p in refs[idx]};matches=[(j,p) for j,p in correct if mathematical_key(tree(p)) in gold];stats['literal_oracle_no_correct_root']+=1
        key='symbolically_equivalent_root_exists' if matches else 'no_independent_source_equivalent_root';stats[key]+=1
        examples.append(dict(train_index=idx,kind=key,question=b[i]['question'],matching_candidate=matches[0][0] if matches else None,matching_program=matches[0][1] if matches else None,reference_program=refs[idx][0]))
    assert stats['literal_oracle_no_correct_root']==read(ROOT/'experiments/round142/fit_audit.json')['stats']['all_no_correct_root']
    dump(OUT/'fit_examples.json',examples);dump(OUT/'fit_audit.json',dict(stats=stats,fit_only=True,test_opened=False,seconds=time.perf_counter()-tick,source_sha256=sha(__file__)));print(stats,flush=True)
