"""Separate conserved-right-hand-side algebra conflicts from symbol reuse."""
from round91_consistency import constraints,features
from fractions import Fraction as F
from collections import defaultdict
import numpy as np

def conserved(text):
    groups=defaultdict(list)
    for r in constraints(text):
        a,b=F(r['coefficient']),F(r['constant'])
        if not a or not b:continue
        if a<0:a,b=-a,-b
        groups[(r['variable'],b)].append(a)
    pairs=bad=groups_bad=0;res=[]
    for values in groups.values():
        problem=False
        for i,a in enumerate(values):
            for b in values[:i]:
                delta=float(abs(a-b)/max(abs(a),abs(b)));wrong=delta>.001
                pairs+=1;bad+=wrong;problem|=wrong;res.append(delta)
        groups_bad+=problem
    return np.asarray([np.log1p(pairs),bad/max(1,pairs),float(bad>0),max(res,default=0),float(np.mean(res)) if res else 0,groups_bad/max(1,len(groups))],np.float32)

def build(bank,arm='conserved'):
    fn=features if arm=='unscoped' else conserved
    return np.asarray([[fn(c['text']) for c in r['candidates']] for r in bank],np.float32)

if __name__=='__main__':
    from round22_common import ROOT,load,dump,sha
    out=ROOT/'experiments/round92';out.mkdir();b,a,y,ids=load('train')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:fit=z['folds']!=2
    rows=[r for r,k in zip(b,fit) if k];x=build(rows);flag=x[:,:,2]>0
    report=dict(fit_questions=len(rows),conserved_conflict_candidates=int(flag.sum()),correct=int((flag*y[fit]).sum()),wrong=int((flag*(1-y[fit])).sum()),train_fit_only=True,test_opened=False)
    assert conserved('x+3x+15x=380\n20x=380\nx=19')[2]==1
    assert conserved('4x=20\nx=5\n4x=40\nx=10')[2]==0
    assert conserved('T - .25*T=72\n.25*T=72')[2]==1
    assert conserved('v=250\nv=125')[2]==0
    dump(out/'fit_audit.json',report);dump(out/'rule_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research/round91_consistency.py',ROOT/'research/round92_consistency.py']},fit_audit_sha256=sha(out/'fit_audit.json'),test_opened=False));print(report,flush=True)
