"""TRAIN-fit audit of what perfect source-pair operation labels can distinguish."""
from round22_common import ROOT,load,read,dump,sha
from round82_relations import relations
from round44_programs import locations
from collections import defaultdict,Counter
import numpy as np
OUT=ROOT/'experiments/round137'
if __name__=='__main__':
    OUT.mkdir();b,a,y,ids=load('train');packets=read(ROOT/'experiments/round102/train_programs.json');labels=read(ROOT/'experiments/round120/labels.json');table=defaultdict(dict)
    for r in labels:
        if r['fold']!=2:table[r['train_index']][(r['left'],r['right'])]=r['label']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    stats=Counter();examples=[]
    for i in np.flatnonzero((f!=2)&(y.min(1)==0)):
        idx=int(ids[i]);ann=table.get(idx,{});stats['mixed_fit_parents']+=1;positive={};negative={};covered=[False]*4
        for j,roots in enumerate(packets[i]):
            for p in roots:
                rel=relations(p)
                if not rel:continue
                if ann and all(rel.get(pair)==label for pair,label in ann.items()):covered[j]=True
                signature=tuple(sorted(rel.items()));target=positive if y[i,j]>0 else negative
                target.setdefault(signature,dict(candidate=j,program=p))
        common=positive.keys()&negative.keys()
        if common:
            stats['same_full_source_pair_signature_correct_and_incorrect']+=1
            sig=sorted(common,key=repr)[0];pos=positive[sig];neg=negative[sig]
            examples.append(dict(train_index=idx,question=b[i]['question'],signature=sig,correct=pos,incorrect=neg,correct_literals=[n[1] for _,n in locations(pos['program']) if n[0]=='L'],incorrect_literals=[n[1] for _,n in locations(neg['program']) if n[0]=='L']))
        if ann:
            stats['with_consensus_labels']+=1;pc=any(c and y[i,j]>0 for j,c in enumerate(covered));nc=any(c and y[i,j]==0 for j,c in enumerate(covered))
            stats['known_labels_support_correct_root']+=pc;stats['known_labels_support_incorrect_root']+=nc
            stats['known_labels_separate_correct_from_all_incorrect']+=pc and not nc;stats['known_labels_allow_both']+=pc and nc;stats['known_labels_no_correct_root_coverage']+=not pc
    dump(OUT/'fit_examples.json',examples);dump(OUT/'fit_audit.json',dict(stats=stats,fit_only=True,source_sha256=sha(__file__),test_opened=False));print(stats,flush=True)
    for r in examples[:5]:print(r,flush=True)
