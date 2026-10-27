"""Independent raw-text model, fixed training, held gate, then VAL selection."""
from round22_common import ROOT,load,dump,sha
from round241_text import fit_vocab,encode,save_vocab
from round241_network import train,predict,SEED,EPOCHS
from round224_target_conclusion import strict_choice
from round225_train import record
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
import time

OUT=ROOT/'experiments/round241'
ALPHAS=[.05,.1,.2,.35]

def main():
    threadpool_limits(4)
    round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(method='Randomly initialized word24 embeddings, question-number binding, strided CNN, decomposable cross attention and group comparison; TRAIN-only matching auxiliary.',
        source_hashes={n:sha(ROOT/'research'/n) for n in ('round241_text.py','round241_network.py','round241_train.py')},
        seed=SEED,epochs=EPOCHS,alphas=ALPHAS,vocab_max=2048,question_length=80,candidate_length=192,
        external_data=False,pretrained_weights=False,gpu=False,threads=4,
        selection='VAL>763 and held>=1479; then validation,held,smaller alpha.',
        validation_role='Reused development set; no TEST access or test-driven choices.',test_opened=False))
    tick=time.perf_counter();bank,_,y,ids=load('train')
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        c={k:z[k] for k in ('folds','held_s1','val_s1','strict_nested','strict_val','ids')}
    assert np.array_equal(ids,c['ids'])
    held=c['folds']==2;rows=np.flatnonzero(~held);bh=[b for b,k in zip(bank,held) if k]
    hs=np.load(ROOT/'experiments/round234/context_ground_held.npz')['scores']
    baseh=c['held_s1']+.25*(hs-hs.mean(1,keepdims=True))
    oldh=strict_choice(bh,baseh,c['strict_nested'][held])
    assert int(y[held][np.arange(held.sum()),oldh].sum())==1479
    vocab=fit_vocab(bank,rows);data=encode(bank,vocab)
    folder=OUT/'held';folder.mkdir();save_vocab(folder/'vocab.json',vocab)
    print('PREPARED',dict(vocab=len(vocab),training=len(rows),seconds=time.perf_counter()-tick),flush=True)
    model,history=train(data,y,rows,len(vocab),folder/'model.npz')
    scores=predict(model,data,np.flatnonzero(held))
    # Architecture must be candidate-permutation and batch independent.
    tiny={k:v[:3].copy() for k,v in data.items()};perm=np.array([2,0,3,1])
    shuffled={k:(v[:,perm].copy() if k in ('c','cn') else v.copy()) for k,v in tiny.items()}
    original=predict(model,tiny)
    assert np.allclose(predict(model,shuffled),original[:,perm],atol=2e-4)
    assert np.allclose(predict(model,{k:v[:1] for k,v in tiny.items()}),original[:1],atol=2e-4)
    np.savez(OUT/'held_scores.npz',scores=scores)
    records=[];eligible=[]
    for alpha in ALPHAS:
        choice=strict_choice(bh,baseh+alpha*(scores-scores.mean(1,keepdims=True)),c['strict_nested'][held])
        rec=dict(alpha=alpha,held=record(y[held],choice,oldh));records.append(rec)
        if rec['held']['correct']>=1479:eligible.append(rec)
        print('HELD',rec,flush=True)
    result=dict(records=records,held_history=history,standalone_held=int(y[held][np.arange(held.sum()),scores.argmax(1)].sum()),
        asset_bytes=sum(p.stat().st_size for p in folder.iterdir()),contract_checks=True,test_opened=False,validation_evaluated=False)
    dump(OUT/'progress.json',result)
    if eligible:
        folder=OUT/'full';folder.mkdir();vocab=fit_vocab(bank,np.arange(len(y)));save_vocab(folder/'vocab.json',vocab)
        data=encode(bank,vocab);model,history=train(data,y,np.arange(len(y)),len(vocab),folder/'model.npz')
        val,_,vy,_=load('validation');vs=predict(model,encode(val,vocab))
        np.savez(OUT/'validation_scores.npz',scores=vs)
        oldvs=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
        basev=c['val_s1']+.25*(oldvs-oldvs.mean(1,keepdims=True));oldv=strict_choice(val,basev,c['strict_val'])
        for rec in eligible:
            choice=strict_choice(val,basev+rec['alpha']*(vs-vs.mean(1,keepdims=True)),c['strict_val'])
            rec['validation']=record(vy,choice,oldv);print('VALIDATION',rec,flush=True)
        result.update(full_history=history,validation_evaluated=True,asset_bytes=sum(p.stat().st_size for p in folder.iterdir()))
    accepted=[r for r in records if r.get('validation',{}).get('correct',0)>763]
    result.update(selected=max(accepted,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['alpha'])) if accepted else None,seconds=time.perf_counter()-tick)
    dump(OUT/'result.json',result);print('RESULT',result,flush=True)

if __name__=='__main__':main()
