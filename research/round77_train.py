"""Repeat ordered-token ablation with genuinely nested diagnostic conditioning."""
from round22_common import ROOT,load,read,dump,sha
from round76_train import fit
from round76_selector import score_tokens
from round77_nested import prepare,OUT
from round22_selector import normalized
from round25_selector import scores as baseline
from round4_gates import apply_gate
import numpy as np,torch,time,shutil

if __name__=='__main__':
    OUT.mkdir();torch.set_num_threads(4);tick=time.perf_counter()
    sources=list((ROOT/'research').glob('round76*.py'))+list((ROOT/'research').glob('round77*.py'))
    dump(ROOT/'experiments/round76/aborted.json',dict(reason='Stopped before completing first arm: diagnostic OOF conditioning was not nested. No validation arm or TEST evaluated.',superseded_by='round77',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources if p.name.startswith('round76')}))
    dump(OUT/'plan.json',dict(arms=['positionless','ordered'],epochs=20,seed=76076,reference_weight=.1,residual_penalty=.01,test_opened=False,nested_diagnostic=True,final_selection_threshold=753,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');tokens={s:read(ROOT/f'experiments/round76/{s}_tokens.json') for s in ('train','validation')}
    weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']]
    nested,f=prepare(b,a,y,ids);nbs=normalized(nested)@weights
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(ids,z['ids']);bs=normalized(z['components'])@weights
    pairs=read(ROOT/'experiments/round67/constraint_pairs.json');vbs=baseline(v,va,ROOT/'experiments/round25/selected_bundle');bc=apply_gate(v,vbs,'parse_and_repair',4);records=[]
    for ordered in (False,True):
        arm='ordered' if ordered else 'positionless'
        for stage in ('diagnostic','full'):
            ix=np.flatnonzero(f!=2) if stage=='diagnostic' else np.arange(len(b));keep=set(map(int,ids[ix]));pp=[p for p in pairs if p['train_index'] in keep]
            path=OUT/stage/arm;conditioning=nbs if stage=='diagnostic' else bs
            fit([tokens['train'][i] for i in ix],y[ix],conditioning[ix],ids[ix],pp,path,ordered)
            if stage=='diagnostic':
                held=np.flatnonzero(f==2);hb=[b[i] for i in held];rr=score_tokens([s for i in held for s in tokens['train'][i]],path).reshape(-1,4)
                c=apply_gate(hb,nbs[held]+rr,'parse_and_repair',4);hc=apply_gate(hb,nbs[held],'parse_and_repair',4)
                dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(len(held)),c].sum()),baseline_correct=int(y[held][np.arange(len(held)),hc].sum()),n=len(held),nested=True))
        r=score_tokens([s for q in tokens['validation'] for s in q],OUT/'full'/arm).reshape(-1,4);c=apply_gate(v,vbs+r,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
        rec=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()));bundle=OUT/f'bundle_{arm}';bundle.mkdir()
        for name in ('model.npz','config.json','vocabulary.json'):shutil.copy2(OUT/'full'/arm/name,bundle/name)
        shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');rec['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert rec['asset_bytes']<=16*1024**2
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);records.append(rec);print('VAL',rec,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round77/bundle_{best['arm']}"),selected='round77' if best['correct']>753 else 'round75',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
