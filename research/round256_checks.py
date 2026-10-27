"""TRAIN-only invariance checks for the learned complementary evidence."""
from round22_common import ROOT,load,dump
from round224_target_conclusion import strict_choice
from round253_attention import scores_and_features
import round250_features as scale_features
import round252_features as monotone_features
import round255_features as alignment_features
import numpy as np

def main():
    bank,_,_,_=load('train')
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        held=z['folds']==2;s=z['held_s1'][:3];strict=z['strict_nested'][held][:3]
    bank=[b for b,k in zip(bank,held) if k][:3]
    q=np.load(ROOT/'experiments/round234/context_ground_held.npz')['scores'][:3]
    s=s+.25*(q-q.mean(1,keepdims=True));old=strict_choice(bank,s,strict)
    path=ROOT/'experiments/round241/held';text,attention=scores_and_features(bank,path)
    perm=np.array([2,0,3,1]);inverse=np.argsort(perm)
    pb=[dict(question=r['question'],candidates=[r['candidates'][j] for j in perm]) for r in bank]
    pt,pa=scores_and_features(pb,path)
    assert np.allclose(pt,text[:,perm],atol=2e-4)
    assert np.allclose(pa,attention[:,perm],atol=2e-5)
    bt,ba=scores_and_features(bank[:1],path)
    assert np.allclose(bt,text[:1],atol=2e-4) and np.allclose(ba,attention[:1],atol=2e-5)
    cases=[(scale_features,['scale_free','text_anchored']),
           (monotone_features,['monotone_linear','monotone_text_gate']),
           (alignment_features,['alignment_utility','alignment_text_anchored'])]
    for module,arms in cases:
        for arm in arms:
            extra=[attention] if module is alignment_features else []
            x,e,_=module.build(bank,s,strict,text,*extra,old,arm)
            pextra=[pa] if module is alignment_features else []
            px,pe,_=module.build(pb,s[:,perm],strict[:,perm],pt,*pextra,inverse[old],arm)
            assert np.isfinite(x).all() and np.allclose(px,x[:,perm],atol=2e-4),arm
            assert np.array_equal(pe,e[:,perm])
            sx,se,_=module.build(bank,s*2+3,strict*4-2,text*.5+7,*extra,old,arm)
            assert np.allclose(sx,x,atol=3e-4) and np.array_equal(se,e),arm
    empty=scores_and_features([],path)
    assert empty[0].shape==(0,4) and empty[1].shape==(0,4,20)
    out=ROOT/'experiments/round256';out.mkdir(exist_ok=True)
    result=dict(candidate_permutation=True,batch_invariance=True,affine_score_invariance=True,
        finite=True,empty=True,test_opened=False)
    dump(out/'checks.json',result);print(result)

if __name__=='__main__':main()
