"""TRAIN-fit oracle audit: can the current candidate graphs express literal supervision?"""
from round22_common import ROOT,load,read,dump,sha
from round138_literals import normalize,relation_map,ABSENT,candidate_constants
from collections import defaultdict,Counter
import numpy as np
OUT=ROOT/'experiments/round142'
if __name__=='__main__':
    OUT.mkdir();b,a,y,ids=load('train');packets=read(ROOT/'experiments/round102/train_programs.json');table=defaultdict(dict)
    for r in read(ROOT/'experiments/round138/fit_labels.json'):table[r['train_index']][(r['quantity'],r['constant'])]=r['label']
    stats=Counter();examples=[]
    for i,idx in enumerate(ids):
        idx=int(idx);ann=table.get(idx)
        if not ann or y[i].min()>0:continue
        flags=[];maps=[]
        for roots in packets[i]:
            roots_ok=[]
            for p in roots:
                try:r,cs=relation_map(normalize(p))
                except (ValueError,ZeroDivisionError):continue
                roots_ok.append(all(r.get(key,ABSENT)==label for key,label in ann.items()))
            flags.append(any(roots_ok))
        pc=any(flag and y[i,j]>0 for j,flag in enumerate(flags));nc=any(flag and y[i,j]==0 for j,flag in enumerate(flags));has_present=any(v!=ABSENT for v in ann.values())
        for prefix in ('all', 'present' if has_present else 'absent_only'):
            stats[prefix+'_parents']+=1;stats[prefix+'_correct_root_covered']+=pc;stats[prefix+'_incorrect_root_covered']+=nc;stats[prefix+'_separable']+=pc and not nc;stats[prefix+'_no_correct_root']+=not pc;stats[prefix+'_both']+=pc and nc
        if has_present and not pc:examples.append(dict(train_index=idx,question=b[i]['question'],correct_candidates=np.flatnonzero(y[i]).tolist(),constraints=[dict(quantity=k[0],constant=k[1],label=v) for k,v in ann.items()]))
    dump(OUT/'fit_uncovered_examples.json',examples);dump(OUT/'fit_audit.json',dict(stats=stats,fit_only=True,test_opened=False,source_sha256=sha(__file__)));print(stats,flush=True)
