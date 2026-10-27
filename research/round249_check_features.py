"""Check new arbiter feature invariances on TRAIN-held inputs only."""
from round22_common import ROOT,load,dump
from round249_features import build,choose
from round224_target_conclusion import strict_choice
import numpy as np

def main():
    bank,_,_,_=load('train')
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        held=z['folds']==2;s=z['held_s1'][:3];strict=z['strict_nested'][held][:3]
    rows=[b for b,k in zip(bank,held) if k][:3]
    q=np.load(ROOT/'experiments/round234/context_ground_held.npz')['scores'][:3]
    s=s+.25*(q-q.mean(1,keepdims=True));n=np.load(ROOT/'experiments/round241/held_scores.npz')['scores'][:3]
    old=strict_choice(rows,s,strict);perm=np.array([2,0,3,1]);inverse=np.argsort(perm)
    changed=[dict(question=r['question'],candidates=[r['candidates'][j] for j in perm]) for r in rows]
    for mode in ('supported','joint','top2'):
        x,eligible,names=build(rows,s,strict,n,old,mode)
        y,ye,_=build(changed,s[:,perm],strict[:,perm],n[:,perm],inverse[old],mode)
        assert np.isfinite(x).all()
        assert np.allclose(y,x[:,perm],atol=2e-5),mode
        assert np.array_equal(ye,eligible[:,perm])
        xx,ee,_=build(rows[:1],s[:1],strict[:1],n[:1],old[:1],mode)
        assert np.allclose(xx,x[:1],atol=2e-5) and np.array_equal(ee,eligible[:1])
    assert choose([],None,None,None,None,None,'supported').shape==(0,)
    result=dict(features=len(names),finite=True,candidate_permutation=True,batch_invariance=True,empty=True,test_opened=False)
    dump(ROOT/'experiments/round247/feature_checks.json',result);print(result)

if __name__=='__main__':main()
