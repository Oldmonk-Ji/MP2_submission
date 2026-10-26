"""Resume after isolating CPU/CUDA fused-attention numerical discrepancy.

No tolerance relaxation, architecture change, or retraining of completed arms.
The fused evaluation path is disabled; the ordinary training path is unchanged.
"""
from round22_common import ROOT,load,read,dump,sha
from round76_train import fit
from round76_selector import score_tokens
from round22_selector import normalized
from round25_selector import scores as baseline
from round4_gates import apply_gate
import numpy as np,torch,time,shutil
OUT=ROOT/'experiments/round77'

if __name__=='__main__':
    assert not (OUT/'training_complete.json').exists();tick=time.perf_counter();torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False)
    checks=read(OUT/'numerics.json');assert checks['False']['max_error']<2e-4
    dump(OUT/'resume_plan.json',dict(reason='Fused eval attention CPU/CUDA discrepancy; disable fused eval path. Original 2e-4 agreement threshold retained.',maximum_observed_error=checks['True']['max_error'],unfused_error=checks['False']['max_error'],training_gradient_path_unchanged=True,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round77*.py')}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');tokens={s:read(ROOT/f'experiments/round76/{s}_tokens.json') for s in ('train','validation')};weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']]
    with np.load(OUT/'nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nbs=normalized(z['components'])@weights;f=z['folds']
    with np.load(ROOT/'experiments/round68/oof.npz') as z:bs=normalized(z['components'])@weights
    held=np.flatnonzero(f==2);hb=[b[i] for i in held];rr=score_tokens([s for i in held for s in tokens['train'][i]],OUT/'diagnostic/ordered').reshape(-1,4);c=apply_gate(hb,nbs[held]+rr,'parse_and_repair',4);hc=apply_gate(hb,nbs[held],'parse_and_repair',4)
    dump(OUT/'ordered_held.json',dict(correct=int(y[held][np.arange(len(held)),c].sum()),baseline_correct=int(y[held][np.arange(len(held)),hc].sum()),n=len(held),nested=True))
    pairs=[p for p in read(ROOT/'experiments/round67/constraint_pairs.json') if p['train_index'] in set(map(int,ids))]
    fit(tokens['train'],y,bs,ids,pairs,OUT/'full/ordered',True)
    vbs=baseline(v,va,ROOT/'experiments/round25/selected_bundle');bc=apply_gate(v,vbs,'parse_and_repair',4);bo=vy[np.arange(len(v)),bc];records=[]
    for arm in ('positionless','ordered'):
        r=score_tokens([s for q in tokens['validation'] for s in q],OUT/'full'/arm).reshape(-1,4);c=apply_gate(v,vbs+r,'parse_and_repair',4);ok=vy[np.arange(len(v)),c]
        rec=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()));bundle=OUT/f'bundle_{arm}'
        if arm=='positionless':assert read(OUT/f'validation_{arm}_predictions.json')['selected']==c.tolist()
        else:
            bundle.mkdir()
            for name in ('model.npz','config.json','vocabulary.json'):shutil.copy2(OUT/'full'/arm/name,bundle/name)
            shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred)
        rec['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert rec['asset_bytes']<=16*1024**2;records.append(rec);print('VAL',rec,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round77/bundle_{best['arm']}"),selected='round77' if best['correct']>753 else 'round75',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(resumed_seconds=time.perf_counter()-tick,training_seconds=sum(read(OUT/stage/arm/'training.json')['seconds'] for stage in ('diagnostic','full') for arm in ('positionless','ordered')),test_opened=False,cpu_cuda_agreement_without_relaxing_tolerance=True))
