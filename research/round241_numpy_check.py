"""Verify exported inference on TRAIN held only; never open TEST."""
from round22_common import ROOT,load,dump
from round241_numpy import scores
from round241_text import encode
from threadpoolctl import threadpool_limits
import numpy as np
import json
import time
import argparse

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,default=241);args=parser.parse_args()
    stage=ROOT/f'experiments/round{args.round}'
    threadpool_limits(4)
    bank,_,_,_=load('train')
    held=np.load(ROOT/'experiments/round216/cache.npz')['folds']==2
    bank=[b for b,k in zip(bank,held) if k]
    folder=stage/'held'
    tick=time.perf_counter();actual=scores(bank,folder);elapsed=time.perf_counter()-tick
    expected=np.load(stage/'held_scores.npz')['scores']
    difference=float(np.abs(actual-expected).max())
    assert np.allclose(actual,expected,atol=2e-4,rtol=2e-5),difference
    tiny=bank[:3];permutation=[2,0,3,1]
    changed=[dict(question=r['question'],candidates=[r['candidates'][j] for j in permutation]) for r in tiny]
    original=scores(tiny,folder)
    assert np.allclose(scores(changed,folder),original[:,permutation],atol=2e-4)
    assert np.allclose(scores(tiny[:1],folder),original[:1],atol=2e-4)
    assert scores([],folder).shape==(0,4)
    tick=time.perf_counter();scores(bank[:467],folder);proxy=time.perf_counter()-tick
    size=sum(p.stat().st_size for p in folder.iterdir())
    result=dict(torch_numpy_max_absolute_difference=difference,held_inference_seconds=elapsed,
        training_prefix467_inference_seconds=proxy,asset_bytes=size,total_with_round235_bytes=16522247+size,
        candidate_permutation=True,batch_invariance=True,empty=True,inference_requires_torch=False,test_opened=False)
    assert result['total_with_round235_bytes']<=16*2**20
    dump(stage/'numpy_checks.json',result);print(result,flush=True)

if __name__=='__main__':main()
