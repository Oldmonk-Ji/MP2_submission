"""Freeze the inner-TRAIN utility policy; final experts still use all TRAIN."""
from round22_common import ROOT,load,read,dump,sha
from round249_features import choose
from round224_target_conclusion import strict_choice
from round225_train import record
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np

OUT=ROOT/'experiments/round251'

def main():
    threadpool_limits(4);round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    candidates=read(ROOT/'experiments/round249/result.json')['records']
    selected=max(candidates,key=lambda r:(r['held']['correct'],-r['held']['harms'],-r['held']['changed']))
    assert selected['held']['correct']>=1477
    policy=ROOT/f"experiments/round249/{selected['arm']}_held.npz"
    dump(OUT/'plan.json',dict(source_hash=sha(__file__),
        hypothesis='Refitting the calibration policy on a different OOF/training-size geometry may erase transferable utility; test the fixed inner-TRAIN policy with the all-TRAIN experts.',
        policy_selection='Highest TRAIN-held utility, then fewer harms/changes, from249. One policy only; no VAL-dependent policy choice.',
        policy_training_rows='folds0/1 only, with held-excluded cross-fitted inputs; no final policy refit.',
        final_expert_training='Existing round235 experts and round241 text expert fit all4743 eligible TRAIN. No question permanently excluded from expert fitting.',
        selected_arm=selected['arm'],policy_sha256=sha(policy),selection='VAL>763 and held>=1477.',test_opened=False))
    with np.load(policy) as z:w={k:z[k] for k in z.files}
    bank,_,y,_=load('train')
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        held=z['folds']==2;strict=z['strict_nested'][held];vstrict=z['strict_val'];vs=z['val_s1']
    with np.load(ROOT/'experiments/round248/scores.npz') as z:hs=z['held_base'];hn=z['held_text']
    bh=[b for b,k in zip(bank,held) if k];oldh=strict_choice(bh,hs,strict)
    hp=choose(bh,hs,strict,hn,oldh,w,selected['mode'])
    hr=record(y[held],hp,oldh);assert hr['correct']==selected['held']['correct']
    val,_,vy,_=load('validation');q=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
    vs=vs+.25*(q-q.mean(1,keepdims=True));vn=np.load(ROOT/'experiments/round246/source241_validation_scores.npz')['scores']
    oldv=strict_choice(val,vs,vstrict);vp=choose(val,vs,vstrict,vn,oldv,w,selected['mode'])
    result=dict(arm=selected['arm'],mode=selected['mode'],held=hr,validation=record(vy,vp,oldv),
        policy=policy.relative_to(ROOT).as_posix(),text_model='experiments/round244/source241/full',
        policy_uses_all_train=False,experts_use_all_train=True,test_opened=False)
    result['accepted']=result['validation']['correct']>763 and hr['correct']>=1477
    template=read(ROOT/'experiments/round235/frozen/validation_predictions.json');template['selected']=vp.tolist();template.pop('prediction_seconds',None)
    dump(OUT/'validation_policy_predictions.json',template);dump(OUT/'result.json',result);print('RESULT',result,flush=True)

if __name__=='__main__':main()
