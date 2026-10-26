from data_access import ROOT
from round22_common import load,read,dump,sha
from round22_selector import normalized
from round26_programs import transform
from round11_train import fit
from round25_selector import scores
from features import metadata
from scipy import sparse
import numpy as np
import datetime,functools,shutil,time
import round4_gates

OUT=ROOT/'experiments/round26'


def main():
    OUT.mkdir();plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method='Learned compatibility of question arithmetic hypotheses and proposed final numbers; candidate reasoning not used by residual',
        contexts=[False,True],regularization=[.01,.1,1.],alphas=[.25,.5,1.],max_quantities=8,max_hypotheses_per_answer=64,
        training_n=4743,validation_n=798,test_opened=False,all_final_experts_fit_4743=True,
        offset='honest grouped round22 TRAIN OOF; final deployment round25 chosen only by validation',
        limitations='Enumerated expression matching is weak evidence, not a proof; no forced answer changes, no per-question rules.')
    dump(OUT/'plan.json',plan);b,_,y,ids=load('train');v,va,vy,_=load('validation');dump(OUT/'training_ids.json',ids.tolist())
    with np.load(ROOT/'experiments/round22/oof.npz') as z:
        assert np.array_equal(z['ids'],ids)
        off=normalized(z['components'][:,:,:5])@np.asarray(read(ROOT/'experiments/round22/selected_bundle/readout.json')['base_weights'])
    base=scores(v,va,ROOT/'experiments/round25/selected_bundle')
    with np.load(ROOT/'experiments/round22/validation_comparison.npz') as z:old=z['incumbent_choice']
    oldok=vy[np.arange(len(v)),old].astype(bool);hard=metadata(v)['hard'];records=[];choices={};tick=time.perf_counter()
    original=round4_gates.trace;round4_gates.trace=functools.lru_cache(maxsize=8000)(original)
    for context in plan['contexts']:
        start=time.perf_counter();print('FEATURES',context,flush=True)
        x=transform(b,context);vx=transform(v,context);sparse.save_npz(OUT/f'train_{context}.npz',x)
        sparse.save_npz(OUT/f'validation_{context}.npz',vx);print('FEATURES DONE',time.perf_counter()-start,'nnz',x.nnz,flush=True)
        for reg in plan['regularization']:
            name=f'context{context}_r{reg}';weight,info=fit(x,y,off,np.arange(len(b)),reg)
            np.savez(OUT/f'{name}.npz',weight=weight);delta=np.asarray(vx@weight).reshape(-1,4)
            for alpha in plan['alphas']:
                key=f'{name}_a{alpha}';c=round4_gates.apply_gate(v,base+alpha*delta,'parse_and_repair',4)
                ok=vy[np.arange(len(v)),c].astype(bool);choices[key]=c
                rec=dict(name=key,model=name,context=context,alpha=alpha,reg=reg,correct=int(ok.sum()),n=len(v),
                    hard_correct=int(ok[hard].sum()),hard_n=int(hard.sum()),repaired=int((ok&~oldok).sum()),
                    damaged=int((~ok&oldok).sum()),net=int(ok.sum()-oldok.sum()),optimizer=info)
                records.append(rec);print('VAL',key,rec['correct'],flush=True)
            dump(OUT/'records.json',records)
    round4_gates.trace=original
    winner=max(records,key=lambda r:(r['correct'],-r['damaged'],-r['alpha']))
    dump(OUT/'selection.json',dict(selected=winner,records=records,seconds=time.perf_counter()-tick,test_opened=False))
    np.savez(OUT/'validation_choices.npz',**choices)
    bundle=OUT/'selected_bundle';bundle.mkdir()
    shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
    shutil.copyfile(OUT/f"{winner['model']}.npz",bundle/'residual.npz')
    dump(bundle/'config.json',dict(context=winner['context'],alpha=winner['alpha']))
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets);assert size<=16*1024**2
    dump(OUT/'asset_manifest.json',dict(bytes=size,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets}))
    print('SELECTED',winner,'bytes',size,flush=True)


if __name__=='__main__':main()
