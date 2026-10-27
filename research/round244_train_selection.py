"""Original user held floor1477 adjudication for independent text models."""
from round22_common import ROOT,load,read,dump,sha
from round241_text import fit_vocab,encode,save_vocab
from round241_network import train as plain_train,predict
from round241_numpy import scores as numpy_scores
from round224_target_conclusion import strict_choice
from round225_train import record
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
import argparse
import datetime

OUT=ROOT/'experiments/round244'

def prepare():
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hash=sha(__file__),sources=[241,242,243],held_floor=1477,validation_must_exceed=763,
        reason='Return to original explicit user held floor1477. Prior1479 was an extra conservative research heuristic. Amendment made using TRAIN-held evidence only, before any VAL evaluation of these text models.',
        selection='VAL>763 and held>=1477, then validation,held,smaller alpha,earlier source. Preserve complete records and report any held regression.',
        training='Same fixed raw-text recipes, all eligible TRAIN rows for final fit. Use exact NumPy inference for held/VAL decisions.',
        test_opened=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:prepare();return
    plan=read(OUT/'plan.json');assert plan['source_hash']==sha(__file__)
    assert not (OUT/'result.json').exists()
    threadpool_limits(4);round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    bank,_,y,_=load('train')
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        c={k:z[k] for k in ('folds','held_s1','val_s1','strict_nested','strict_val')}
    held=c['folds']==2;bh=[b for b,k in zip(bank,held) if k]
    hs=np.load(ROOT/'experiments/round234/context_ground_held.npz')['scores']
    baseh=c['held_s1']+.25*(hs-hs.mean(1,keepdims=True));oldh=strict_choice(bh,baseh,c['strict_nested'][held])
    assert int(y[held][np.arange(held.sum()),oldh].sum())==1479
    records=[];val=oldv=None;histories={}
    for source in plan['sources']:
        stage=ROOT/f'experiments/round{source}';previous=read(stage/'result.json')
        configs=[r for r in previous['records'] if r['held']['correct']>=plan['held_floor']]
        if not configs:continue
        raw=numpy_scores(bh,stage/'held');eligible=[]
        for config in configs:
            alpha=config['alpha'];chosen=strict_choice(bh,baseh+alpha*(raw-raw.mean(1,keepdims=True)),c['strict_nested'][held])
            rec=dict(source=source,alpha=alpha,held=record(y[held],chosen,oldh),prior_stricter_held_floor=1479)
            records.append(rec)
            assert rec['held']['correct']==config['held']['correct'],'Torch/NumPy selected-choice mismatch'
            if rec['held']['correct']>=plan['held_floor']:eligible.append(rec)
        if not eligible:continue
        folder=stage/'full'
        if not (folder/'model.npz').exists():
            folder=OUT/f'source{source}'/'full';folder.mkdir(parents=True,exist_ok=False)
            vocab=fit_vocab(bank,np.arange(len(y)));save_vocab(folder/'vocab.json',vocab)
            if source==241:trainer=plain_train
            elif source==242:
                from round242_pretrain import train as trainer
            else:
                from round243_residual import train as trainer
            _,histories[str(source)]=trainer(encode(bank,vocab),y,np.arange(len(y)),len(vocab),folder/'model.npz')
        if val is None:
            val,_,vy,_=load('validation')
            vs=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
            basev=c['val_s1']+.25*(vs-vs.mean(1,keepdims=True));oldv=strict_choice(val,basev,c['strict_val'])
            assert int(vy[np.arange(len(vy)),oldv].sum())==763
        asset=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        assert 16522247+asset<16*2**20
        vs=numpy_scores(val,folder)
        np.savez(OUT/f'source{source}_validation_scores.npz',scores=vs)
        for rec in eligible:
            chosen=strict_choice(val,basev+rec['alpha']*(vs-vs.mean(1,keepdims=True)),c['strict_val'])
            rec.update(validation=record(vy,chosen,oldv),model_directory=folder.relative_to(ROOT).as_posix(),text_asset_bytes=asset)
            print('VALIDATION',rec,flush=True)
        dump(OUT/'progress.json',dict(records=records,histories=histories,test_opened=False))
    accepted=[r for r in records if r.get('validation',{}).get('correct',0)>763]
    selected=max(accepted,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['alpha'],-r['source'])) if accepted else None
    result=dict(records=records,histories=histories,selected=selected,held_floor=1477,baseline_held=1479,test_opened=False)
    dump(OUT/'result.json',result);print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
