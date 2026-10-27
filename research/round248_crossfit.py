"""Generate TRAIN-only cross-fitted text/base predictions for a utility arbiter."""
from round22_common import ROOT,load,dump,sha
from round241_text import fit_vocab,encode,save_vocab
from round241_network import train
from round241_numpy import scores as text_scores
from round227_quadratic import inputs,fit_space
from round234_semantic_verification import design,fit
from threadpoolctl import threadpool_limits
import numpy as np
import time

OUT=ROOT/'experiments/round248'

def main():
    threadpool_limits(4);OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hashes={n:sha(ROOT/'research'/n) for n in ('round248_crossfit.py','round241_text.py','round241_network.py','round241_numpy.py','round227_quadratic.py','round234_semantic_verification.py')},
        method='Two-fold OOF on folds0/1 for held-excluded arbiter fitting; three-fold OOF over all TRAIN for final arbiter fitting.',
        neural='Fixed original round241 recipe; same seed/epochs, no seed ensemble. Additional models generate training features only.',
        teacher='Cached fit-only/3-fold OOF old fusion plus separately TRAIN-fitted context-ground quadratic experts.',
        fitting='Every vocabulary/projection/normalizer fits the corresponding fitting TRAIN partition only.',
        test_opened=False))
    tick=time.perf_counter();bank,arrays,y,ids=load('train');e,g=inputs(bank,arrays)
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        folds=z['folds'];assert np.array_equal(z['ids'],ids) and np.array_equal(z['y'],y)
        held_s1=z['held_s1'];oof_s1=z['oof_s1']
    design_s1=np.load(ROOT/'experiments/round225/fit_only_s1.npz')['scores']
    held=folds==2;bh=[b for b,k in zip(bank,held) if k]
    held_q=np.load(ROOT/'experiments/round234/context_ground_held.npz')['scores']
    held_text=text_scores(bh,ROOT/'experiments/round241/held')
    design_text=np.zeros_like(y);oof_text=np.zeros_like(y);oof_text[held]=held_text
    design_q=np.zeros_like(y);oof_q=np.zeros_like(y);oof_q[held]=held_q
    histories={}
    for mode in ('design','oof'):
        for fold in (0,1):
            rows=np.flatnonzero(folds==1-fold if mode=='design' else folds!=fold)
            target=folds==fold
            folder=OUT/f'{mode}_fold{fold}';text=folder/'text';text.mkdir(parents=True)
            vocab=fit_vocab(bank,rows);save_vocab(text/'vocab.json',vocab)
            model,history=train(encode(bank,vocab),y,rows,len(vocab),text/'model.npz');del model
            ns=text_scores([b for b,k in zip(bank,target) if k],text)
            space=fit_space(e,g,rows);x=design(e,g,space,'context_ground');weight=fit(x,y,rows)
            qs=x[target]@weight;del x
            np.savez(folder/'quadratic.npz',**dict(space,weight=weight))
            (design_text if mode=='design' else oof_text)[target]=ns
            (design_q if mode=='design' else oof_q)[target]=qs
            histories[f'{mode}_{fold}']=history
            dump(OUT/'progress.json',dict(completed=list(histories),histories=histories,test_opened=False))
            print('CROSSFIT',mode,fold,'fit',len(rows),'target',int(target.sum()),'seconds',time.perf_counter()-tick,flush=True)
    design_base=design_s1+.25*(design_q-design_q.mean(1,keepdims=True))
    oof_base=oof_s1+.25*(oof_q-oof_q.mean(1,keepdims=True))
    held_base=held_s1+.25*(held_q-held_q.mean(1,keepdims=True))
    np.savez(OUT/'scores.npz',design_base=design_base,design_text=design_text,oof_base=oof_base,oof_text=oof_text,
             held_base=held_base,held_text=held_text,folds=folds,ids=ids)
    dump(OUT/'result.json',dict(histories=histories,seconds=time.perf_counter()-tick,held_labels_used_for_fitting=False,
        prediction_ensemble=False,test_opened=False))
    print('READY',dict(seconds=time.perf_counter()-tick,test_opened=False),flush=True)

if __name__=='__main__':main()
