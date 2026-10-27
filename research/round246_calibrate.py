"""TRAIN-only scale choice for new text networks; one VAL check per useful model."""
from round22_common import ROOT,load,read,dump,sha
from round224_target_conclusion import strict_choice
from round225_train import record
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
import time

OUT=ROOT/'experiments/round246'
SOURCES=[241,242,243,245]
GRID=[('raw',.005),('raw',.01),('raw',.02),('normalized',.025),('normalized',.05),('normalized',.1)]

def adjusted(s,mode):
    centered=s-s.mean(1,keepdims=True)
    return centered if mode=='raw' else centered/np.maximum(s.std(1,keepdims=True),1e-6)

def main():
    threadpool_limits(4);round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hash=sha(__file__),sources=SOURCES,grid=GRID,
        reason='Neural score units are arbitrary; calibrate weaker/normalized increments on TRAIN-held only. No old feature or gate changes.',
        advance='Choose ONE configuration per architecture by highest held, fewer harms, fewer changed choices; require strict held>1479 before an additional VAL evaluation.',
        accept='VAL>763, original user held floor1477, assets<=16MiB; no TEST access.',test_opened=False))
    bank,_,y,_=load('train')
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        c={k:z[k] for k in ('folds','held_s1','val_s1','strict_nested','strict_val')}
    held=c['folds']==2;bh=[b for b,k in zip(bank,held) if k]
    hs=np.load(ROOT/'experiments/round234/context_ground_held.npz')['scores']
    baseh=c['held_s1']+.25*(hs-hs.mean(1,keepdims=True));oldh=strict_choice(bh,baseh,c['strict_nested'][held])
    all_records=[];advanced=[];val=oldv=None;histories={};profiles=[]
    for source in SOURCES:
        stage=ROOT/f'experiments/round{source}'
        if source==245:
            from round245_numpy import scores
        else:
            from round241_numpy import scores
        tick=time.perf_counter();raw=scores(bh,stage/'held');elapsed=time.perf_counter()-tick
        expected=np.load(stage/'held_scores.npz')['scores']
        difference=float(np.abs(raw-expected).max())
        assert np.allclose(raw,expected,atol=2e-4,rtol=2e-5),(source,difference)
        tick=time.perf_counter();scores(bh[:467],stage/'held');prefix=time.perf_counter()-tick
        profiles.append(dict(source=source,torch_numpy_max_difference=difference,held_inference_seconds=elapsed,train_prefix467_seconds=prefix))
        records=[]
        for mode,alpha in GRID:
            choice=strict_choice(bh,baseh+alpha*adjusted(raw,mode),c['strict_nested'][held])
            rec=dict(source=source,mode=mode,alpha=alpha,held=record(y[held],choice,oldh))
            records.append(rec);all_records.append(rec);print('CALIBRATION',rec,flush=True)
        chosen=max(records,key=lambda r:(r['held']['correct'],-r['held']['harms'],-r['held']['changed'],-GRID.index((r['mode'],r['alpha']))))
        if chosen['held']['correct']<=1479:continue
        folder=stage/'full'
        if source==241 and not (folder/'model.npz').exists():folder=ROOT/'experiments/round244/source241/full'
        if not (folder/'model.npz').exists():
            folder=OUT/f'source{source}'/'full';folder.mkdir(parents=True,exist_ok=False)
            if source==245:
                from round245_text import fit_vocab,encode,save_vocab
                from round245_network import train
            else:
                from round241_text import fit_vocab,encode,save_vocab
                if source==241:
                    from round241_network import train
                elif source==242:
                    from round242_pretrain import train
                else:
                    from round243_residual import train
            vocab=fit_vocab(bank,np.arange(len(y)));save_vocab(folder/'vocab.json',vocab)
            _,histories[str(source)]=train(encode(bank,vocab),y,np.arange(len(y)),len(vocab),folder/'model.npz')
        if val is None:
            val,_,vy,_=load('validation')
            vs=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
            basev=c['val_s1']+.25*(vs-vs.mean(1,keepdims=True));oldv=strict_choice(val,basev,c['strict_val'])
        asset=sum(p.stat().st_size for p in folder.iterdir() if p.is_file());assert 16522247+asset<16*2**20
        vs=scores(val,folder);np.savez(OUT/f'source{source}_validation_scores.npz',scores=vs)
        choice=strict_choice(val,basev+chosen['alpha']*adjusted(vs,chosen['mode']),c['strict_val'])
        chosen.update(validation=record(vy,choice,oldv),model_directory=folder.relative_to(ROOT).as_posix(),text_asset_bytes=asset)
        advanced.append(chosen);print('VALIDATION',chosen,flush=True)
        dump(OUT/'progress.json',dict(records=all_records,advanced=advanced,profiles=profiles,histories=histories,test_opened=False))
    eligible=[r for r in advanced if r['validation']['correct']>763]
    selected=max(eligible,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['validation']['harms'])) if eligible else None
    result=dict(records=all_records,advanced=advanced,selected=selected,profiles=profiles,histories=histories,test_opened=False)
    dump(OUT/'result.json',result);print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
