"""Freeze the final TRAIN/VAL-selected text blend before one authorized TEST."""
from round22_common import ROOT,read,load,dump,sha
from round224_target_conclusion import strict_choice
import numpy as np
import shutil

OUT=ROOT/'experiments/round257'

def main():
    candidate=read(ROOT/'experiments/round246/result.json')['advanced'][0]
    assert candidate['source']==241 and candidate['mode']=='raw' and candidate['alpha']==.01
    assert candidate['held']['correct']==1481 and candidate['validation']['correct']==763
    OUT.mkdir(exist_ok=False);bundle=OUT/'bundle'
    shutil.copytree(ROOT/'experiments/round235/bundle',bundle)
    shutil.copytree(ROOT/'experiments/round244/source241/full',bundle/'text')
    cfg=read(bundle/'config.json');cfg['text_alpha']=.01;dump(bundle/'config.json',cfg)
    size=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert size<=16*2**20
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        c={k:z[k] for k in ('folds','held_s1','val_s1','strict_nested','strict_val')}
    bank,_,y,_=load('train');held=c['folds']==2;bh=[b for b,k in zip(bank,held) if k]
    q=np.load(ROOT/'experiments/round234/context_ground_held.npz')['scores']
    n=np.load(ROOT/'experiments/round248/scores.npz')['held_text']
    hs=c['held_s1']+.25*(q-q.mean(1,keepdims=True))+.01*(n-n.mean(1,keepdims=True))
    hc=strict_choice(bh,hs,c['strict_nested'][held]);assert int(y[held][np.arange(held.sum()),hc].sum())==1481
    val,_,vy,_=load('validation');q=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
    n=np.load(ROOT/'experiments/round246/source241_validation_scores.npz')['scores']
    vs=c['val_s1']+.25*(q-q.mean(1,keepdims=True))+.01*(n-n.mean(1,keepdims=True))
    choice=strict_choice(val,vs,c['strict_val']);assert int(vy[np.arange(len(vy)),choice].sum())==763
    pred=read(ROOT/'experiments/round235/frozen/validation_predictions.json');pred['selected']=choice.tolist();pred.pop('prediction_seconds',None)
    dump(OUT/'validation_text_raw001_predictions.json',pred)
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm='text_raw001',correct=763,bundle='experiments/round257/bundle'),
        selected='round257',test_target_fixed_before_test=True,test_used_for_selection=False,
        selection_basis='Existing TRAIN/VAL candidates: highest VAL, then higher held. Choose763/1481 before final TEST; no post-TEST reselection.'))
    dump(OUT/'freeze_plan.json',dict(method='round235 plus raw-centered scratch text expert coefficient0.01 before unchanged gate/strict rule.',
        validation=763,held=1481,asset_bytes=size,final_optimization_stop=True,
        latest_user_authorization='在测试集上试一下 完成这组之后不改了',
        authorized_single_final_trial_despite_no_val_gain_over235=True,
        test_authorized=True,test_opened=False,test_used_for_tuning_or_selection=False,
        after_test='Stop model optimization and preserve frozen method, irrespective of test outcome.',
        source_hashes={n:sha(ROOT/'research'/n) for n in ('round257_selector.py','round241_numpy.py','round241_text.py')},
        text_model_hashes={p.name:sha(p) for p in (bundle/'text').iterdir() if p.is_file()}))
    print(dict(validation=763,held=1481,asset_bytes=size,final_optimization_stop=True,test_opened=False),flush=True)

if __name__=='__main__':main()
