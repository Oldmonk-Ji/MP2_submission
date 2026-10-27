"""OOF attention grounding features; all representations come from TRAIN models."""
from round22_common import ROOT,load,dump,sha
from round253_attention import scores_and_features,NAMES
from threadpoolctl import threadpool_limits
import numpy as np

OUT=ROOT/'experiments/round254'

def main():
    threadpool_limits(4);OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hashes={n:sha(ROOT/'research'/n) for n in ('round253_attention.py','round254_attention_cache.py')},
        method='Extract learned question/candidate attention grounding, with fixed semantic axes rather than unaligned hidden units.',
        source='Same cross-fitted networks248 and original held/full241; no new network fitting.',test_opened=False))
    bank,_,_,ids=load('train')
    with np.load(ROOT/'experiments/round248/scores.npz') as z:
        folds=z['folds'];assert np.array_equal(z['ids'],ids)
        old={k:z[k] for k in ('design_text','oof_text','held_text')}
    design=np.zeros((len(bank),4,len(NAMES)),np.float32);oof=np.zeros_like(design)
    maxdiff=0.
    for mode in ('design','oof'):
        for fold in (0,1):
            mask=folds==fold;rows=[b for b,k in zip(bank,mask) if k]
            score,features=scores_and_features(rows,ROOT/f'experiments/round248/{mode}_fold{fold}/text')
            difference=float(np.abs(score-old[mode+'_text'][mask]).max());maxdiff=max(maxdiff,difference)
            assert difference<2e-4
            (design if mode=='design' else oof)[mask]=features
    held=folds==2;score,hf=scores_and_features([b for b,k in zip(bank,held) if k],ROOT/'experiments/round241/held')
    assert np.allclose(score,old['held_text'],atol=2e-4);oof[held]=hf
    val,_,_,_=load('validation');score,vf=scores_and_features(val,ROOT/'experiments/round244/source241/full')
    expected=np.load(ROOT/'experiments/round246/source241_validation_scores.npz')['scores']
    assert np.allclose(score,expected,atol=2e-4)
    np.savez_compressed(OUT/'cache.npz',design=design,oof=oof,held=hf,validation=vf,ids=ids)
    dump(OUT/'result.json',dict(attention_features=NAMES,raw_logits_unchanged=True,max_difference=maxdiff,test_opened=False))
    print('ATTENTION_READY',dict(features=len(NAMES),raw_logits_unchanged=True,test_opened=False),flush=True)

if __name__=='__main__':main()
