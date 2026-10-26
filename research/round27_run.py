from data_access import ROOT
from round22_common import load,read,dump,sha
from round22_selector import normalized
from round24_run import fit
from round27_selector import compact,kernel
from round25_selector import scores
from features import metadata
import numpy as np
import torch
import datetime,functools,shutil,time
import round4_gates

OUT=ROOT/'experiments/round27'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'selection.json').exists();plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=27027,
        anchors=[256,2048],gamma=[.3,3.,10.],regularization=[.0001,.001,.01],alphas=[.25,.5,1.],
        training_n=4743,validation_n=798,test_opened=False,offset='round22 grouped train OOF',
        hypothesis='Larger multiscale nonlinear basis for official embeddings and arithmetic evidence, controlled against 256-anchor capacity',
        projection='Train-only PCA64/scalers from round24. Anchor sampling uses eligible TRAIN only, no labels.',
        baseline='round25 752/798; no test inputs used',selection='full validation, correct, fewer harms, smaller asset')
    if (OUT/'plan.json').exists():plan=read(OUT/'plan.json')
    else:dump(OUT/'plan.json',plan)
    torch.set_num_threads(4)
    b,a,y,ids=load('train');v,va,vy,_=load('validation');dump(OUT/'training_ids.json',ids.tolist())
    with np.load(ROOT/'experiments/round24/space.npz') as z:space={k:z[k] for k in ['pca_mean','pca','pca_scale','small_mean','small_scale']}
    print('FEATURES',flush=True);x=compact(b,a,space);vx=compact(v,va,space);rng=np.random.default_rng(plan['seed'])
    with np.load(ROOT/'experiments/round22/oof.npz') as z:
        assert np.array_equal(z['ids'],ids)
        off=normalized(z['components'][:,:,:5])@np.asarray(read(ROOT/'experiments/round22/selected_bundle/readout.json')['base_weights'])
    base=scores(v,va,ROOT/'experiments/round25/selected_bundle')
    with np.load(ROOT/'experiments/round22/validation_comparison.npz') as z:old=z['incumbent_choice']
    oldok=vy[np.arange(len(v)),old].astype(bool);hard=metadata(v)['hard'];records=[];choices={};tick=time.perf_counter()
    original=round4_gates.trace;round4_gates.trace=functools.lru_cache(maxsize=8000)(original)
    anchor_ids=rng.choice(len(x),2048,replace=False)
    for count in plan['anchors']:
        sp=dict(space,anchors=x[anchor_ids[:count]])
        sample=x[rng.choice(len(x),512,replace=False)];anchors=sp['anchors']
        distances=np.maximum((sample*sample).sum(1)[:,None]+(anchors*anchors).sum(1)[None,:]-2*sample@anchors.T,0)
        sp['distance_median']=np.asarray(np.median(distances),np.float32)
        k=kernel(x,sp);vk=kernel(vx,sp);mean=k.mean(0);scale=k.std(0).clip(.001)*np.float32(np.sqrt(k.shape[1]))
        k=((k-mean)/scale).reshape(len(b),4,-1);vk=((vk-mean)/scale).reshape(len(v),4,-1)
        for reg in plan['regularization']:
            name=f'anchors{count}_r{reg}';w,info=fit(k,y,off,reg)
            np.savez(OUT/f'{name}.npz',**sp,mean=mean,scale=scale,weight=w)
            delta=vk@w
            for alpha in plan['alphas']:
                key=f'{name}_a{alpha}';c=round4_gates.apply_gate(v,base+alpha*delta,'parse_and_repair',4)
                ok=vy[np.arange(len(v)),c].astype(bool);choices[key]=c
                r=dict(name=key,model=name,alpha=alpha,correct=int(ok.sum()),n=len(v),
                    hard_correct=int(ok[hard].sum()),hard_n=int(hard.sum()),repaired=int((ok&~oldok).sum()),
                    damaged=int((~ok&oldok).sum()),net=int(ok.sum()-oldok.sum()),kernel_bytes=(OUT/f'{name}.npz').stat().st_size,optimizer=info)
                records.append(r);print('VAL',key,r['correct'],flush=True)
            dump(OUT/'records.json',records)
    round4_gates.trace=original
    winner=max(records,key=lambda r:(r['correct'],-r['damaged'],-r['kernel_bytes'],-r['alpha']))
    dump(OUT/'selection.json',dict(selected=winner,records=records,seconds=time.perf_counter()-tick,test_opened=False))
    np.savez(OUT/'validation_choices.npz',**choices)
    bundle=OUT/'selected_bundle';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
    shutil.copyfile(OUT/f"{winner['model']}.npz",bundle/'kernel.npz');dump(bundle/'config.json',dict(alpha=winner['alpha']))
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets);assert size<=16*1024**2
    dump(OUT/'asset_manifest.json',dict(bytes=size,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets}))
    print('SELECTED',winner,'bytes',size,flush=True)


if __name__=='__main__':main()
