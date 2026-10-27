"""TRAIN-held complement audit; VAL aggregate ceilings only; no TEST."""
from round22_common import ROOT,load,dump,sha
from round224_target_conclusion import strict_choice
from common import parse_answer
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
from collections import Counter

OUT=ROOT/'experiments/round247'

def audit(bank,y,s,strict,neural,details=False):
    old=strict_choice(bank,s,strict);row=np.arange(len(y));good=y[row,old]>0
    net=neural.argmax(1);ngood=y[row,net]>0
    raw=s.argmax(1);second=np.argsort(-s,axis=1)[:,:2]
    help_indices=np.flatnonzero(~good&ngood)
    potential=[]
    for i in np.flatnonzero(~good):
        correct=np.flatnonzero(y[i]>0)
        supported=correct[neural[i,correct]>neural[i,old[i]]]
        if len(supported):potential.append(int(i))
    result=dict(n=len(y),baseline_correct=int(good.sum()),neural_correct=int(ngood.sum()),
        neural_argmax_repairs=int((~good&ngood).sum()),neural_argmax_harms=int((good&~ngood).sum()),
        any_neural_supported_correct_alternative=len(potential),
        neural_argmax_repairs_in_baseline_top2=int(sum(net[i] in second[i] for i in help_indices)),
        neural_argmax_repairs_with_strict_agreement=int(sum(net[i]==strict[i].argmax() for i in help_indices)),
        neural_argmax_repairs_after_gate_flip=int(sum(raw[i]!=old[i] for i in help_indices)))
    if details:
        cases=[]
        for i in help_indices:
            cases.append(dict(train_held_index=int(i),question=bank[i]['question'],old=int(old[i]),neural=int(net[i]),
                correct=np.flatnonzero(y[i]).tolist(),parsed=[parse_answer(c['text']) for c in bank[i]['candidates']],
                base_margin=float(s[i,old[i]]-s[i,net[i]]),neural_margin=float(neural[i,net[i]]-neural[i,old[i]]),
                strict_margin=float(strict[i,net[i]]-strict[i,old[i]]),old_was_raw_argmax=bool(raw[i]==old[i]),
                conclusions=[c['text'][-350:] for c in bank[i]['candidates']]))
        result['train_cases']=cases
    return result

def main():
    threadpool_limits(4);round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hash=sha(__file__),method='Audit utility of new text scorer, including gate/strict masking and answer-group effects.',test_opened=False))
    bank,_,y,_=load('train')
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        held=z['folds']==2;hs=z['held_s1'];strict=z['strict_nested'][held];vs=z['val_s1'];vstrict=z['strict_val']
    q=np.load(ROOT/'experiments/round234/context_ground_held.npz')['scores']
    hs=hs+.25*(q-q.mean(1,keepdims=True));bh=[b for b,k in zip(bank,held) if k]
    n=np.load(ROOT/'experiments/round241/held_scores.npz')['scores']
    train=audit(bh,y[held],hs,strict,n,True)
    val,_,vy,_=load('validation');q=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
    vs=vs+.25*(q-q.mean(1,keepdims=True));vn=np.load(ROOT/'experiments/round246/source241_validation_scores.npz')['scores']
    validation=audit(val,vy,vs,vstrict,vn,False)
    dump(OUT/'result.json',dict(train_held=train,validation_aggregate_only=validation,test_opened=False))
    print('TRAIN', {k:v for k,v in train.items() if k!='train_cases'},flush=True)
    print('VALIDATION_CEILING',validation,flush=True)

if __name__=='__main__':main()
