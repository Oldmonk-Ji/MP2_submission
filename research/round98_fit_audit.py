from round22_common import ROOT,load,read,dump,sha
from round98_influence import labels,checks
from round29_programs import quantities
from collections import Counter
import numpy as np

if __name__=='__main__':
    out=ROOT/'experiments/round98';out.mkdir();b,a,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)}
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:folds=z['folds']
    excluded={r['train_index'] for r in read(ROOT/'experiments/round56/held_source_review_before_reference.json')};sources=[dict(r,kind='strict') for r in read(ROOT/'experiments/round63/accepted_reference_programs.json')]+[dict(r,kind='reviewed') for r in read(ROOT/'experiments/round51/reviewed_programs.json')];counts=Counter();examples=[];parents=set()
    for r in sources:
        idx=r['train_index'];i=index[idx]
        if idx in excluded or folds[i]==2:continue
        q=b[i]['question'];ls=labels(q,r['program'],r['kind']=='reviewed');parents.add(idx);counts['programs']+=1;counts['quantity_labels']+=len(ls);counts['previously_excluded_labels']+=sum(x['old_excluded'] for x in ls);counts['duplicate_value_labels']+=sum(x['duplicate_value'] for x in ls)
        for x in ls:counts['class_'+str(x['label'])]+=1
        if any(x['duplicate_value'] for x in ls) and len(examples)<10:examples.append(dict(train_index=idx,question=q,source_kind=r['kind'],program=r['program'],labels=ls))
    report=dict(counts=dict(counts),parents=len(parents),examples=examples,checks=checks(),train_fit_only=True,test_opened=False)
    dump(out/'fit_audit.json',report);dump(out/'rule_freeze.json',dict(source_sha256=sha(ROOT/'research/round98_influence.py'),audit_sha256=sha(out/'fit_audit.json'),test_opened=False));print(dict(counts),len(parents),flush=True)
