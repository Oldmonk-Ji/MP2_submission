"""Frozen local-relation ablation on TRAIN diagnostics and complete validation."""
from data_access import ROOT
from round22_common import read,dump
from round48_local import focus,csr,node_scores,infer
from round44_programs import edits,valid
import round46_evaluate
import numpy as np
from collections import Counter,defaultdict

OUT=ROOT/'experiments/round48'

def held_relations():
    rows=read(OUT/'held_source_review_before_reference.json');pairs=[]
    for r in rows:
        counts=Counter();seen=set()
        for kind,p,d in edits(r['program'],r['quantities']):
            key=repr(p);a=valid(p,r['quantities'])
            if a is None or a==r['source_answer'] or key in seen or counts[kind]>=4:continue
            counts[kind]+=1;seen.add(key);pairs.append(dict(train_index=r['train_index'],question=r['question'],positive=r['program'],negative=p,decision=d,kind=kind))
    fp=[focus(p) for p in pairs];xp=csr([r[0] for r in fp]);xn=csr([r[1] for r in fp]);records=[]
    for arm in ('sft','exact','grpo'):
        margins=[]
        for seed in (48048,48049):
            with np.load(OUT/'diagnostic'/f'{arm}_s{seed}'/'model.npz') as w:margins.append(node_scores(xp,w)-node_scores(xn,w))
        m=np.mean(margins,axis=0);byparent=defaultdict(list);bykind=defaultdict(list)
        for p,v in zip(pairs,m):byparent[p['train_index']].append(bool(v>0));bykind[p['kind']].append(bool(v>0))
        rec=dict(arm=arm,n=len(pairs),correct=int((m>0).sum()),parents=len(byparent),parent_mean_accuracy=float(np.mean([np.mean(v) for v in byparent.values()])),by_kind={k:dict(n=len(v),correct=sum(v)) for k,v in bykind.items()})
        records.append(rec);print('LOCAL_HELD_RELATIONS',rec,flush=True)
    dump(OUT/'held_relation_results.json',dict(records=records,test_opened=False,held_labels_never_in_process_gradients=True))

if __name__=='__main__':
    held_relations();round46_evaluate.OUT=OUT;round46_evaluate.infer=infer;round46_evaluate.main()
