"""Freeze the best round234 development candidate without any TEST access."""
from round22_common import ROOT, read, dump, sha, load
from round224_target_conclusion import strict_choice
import shutil
import numpy as np

def main():
    selected=read(ROOT/'experiments/round234/result.json')['selected']
    assert selected and selected['validation']['correct']>762 and selected['held']['correct']>=1479
    out=ROOT/'experiments/round235'
    out.mkdir(exist_ok=False)
    shutil.copytree(ROOT/'experiments/round232/bundle',out/'bundle')
    arm=selected['arm'];alpha=selected['alpha']
    shutil.copy2(ROOT/f'experiments/round234/{arm}_full.npz',out/'bundle/quadratic.npz')
    cfg=read(out/'bundle/config.json');cfg.update(quadratic_arm=arm,quadratic_alpha=alpha)
    dump(out/'bundle/config.json',cfg)
    size=sum(p.stat().st_size for p in (out/'bundle').rglob('*') if p.is_file())
    assert size<=16*2**20
    val,_,vy,_=load('validation')
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        base,strict=z['val_s1'],z['strict_val']
    scores=np.load(ROOT/f'experiments/round234/{arm}_validation.npz')['scores']
    choice=strict_choice(val,base+alpha*(scores-scores.mean(1,keepdims=True)),strict)
    assert int(vy[np.arange(len(vy)),choice].sum())==selected['validation']['correct']
    name='semantic_ground'
    pred=read(ROOT/'experiments/round232/frozen/validation_predictions.json')
    pred['selected']=choice.tolist();pred.pop('prediction_seconds',None)
    dump(out/f'validation_{name}_predictions.json',pred)
    dump(out/'final_feature_selection.json',dict(best_new=dict(arm=name,correct=selected['validation']['correct'],bundle='experiments/round235/bundle'),selected='round235',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(out/'freeze_plan.json',dict(selected=selected,asset_bytes=size,test_opened=False,test_authorized=False,
        source_hashes={n:sha(ROOT/'research'/n) for n in ('round234_semantic_verification.py','round235_quadratic_inference.py','round235_selector.py')},
        note='All new decisions use TRAIN/VAL only; previous aggregate test results are not an optimization criterion.'))
    print(dict(selected=selected,asset_bytes=size,test_opened=False),flush=True)

if __name__=='__main__':main()
