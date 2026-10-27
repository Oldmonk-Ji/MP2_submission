"""TRAIN/VAL-only answer-group verification interactions over round235."""
from round22_common import ROOT, load, dump, sha
from round234_semantic_verification import design as old_design, fit
from round224_target_conclusion import strict_choice
from round225_train import record
from features import metadata
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
import time

OUT=ROOT/'experiments/round236'
ARMS=['group_mean','group_extrema','evidence_curvature']
ALPHAS=[.1,.25,.5]

def design(e,g,w,answers,arm):
    x=old_design(e,g,w,'context_ground')
    q=e.mean(1)
    q=np.clip(((q-w['qmean'])@w['qpc'].T)/w['qscale'],-6,6)
    g=np.clip((g-w['gmean'])/w['gscale'],-6,6)
    if arm=='evidence_curvature':
        evidence=[np.square(g)/6]
    else:
        same=np.asarray([[[a is not None and a==b for b in row] for a in row] for row in answers],bool)
        same |= np.eye(4,dtype=bool)[None]
        other=~same
        own=np.einsum('nij,njk->nik',same.astype(np.float32),g)/same.sum(2,keepdims=True)
        alt=np.einsum('nij,njk->nik',other.astype(np.float32),g)/np.maximum(other.sum(2,keepdims=True),1)
        if arm=='group_mean': evidence=[own-alt]
        else:
            allg=g[:,None,:,:]
            high=np.max(np.where(same[:,:,:,None],allg,-np.inf),axis=2)
            low=np.min(np.where(same[:,:,:,None],allg,np.inf),axis=2)
            evidence=[high-alt,low-alt]
    blocks=[x]
    for h in evidence:
        blocks.append((q[:,None,:,None]*h[:,:,None,:]).reshape(len(e),4,-1)/np.sqrt(32*g.shape[-1]))
    return np.ascontiguousarray(np.concatenate(blocks,2),dtype=np.float32)

def main():
    threadpool_limits(4)
    round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hash=sha(__file__),arms=ARMS,alphas=ALPHAS,
        selection='VAL>763 and held>=1479, then VAL, held, smaller alpha.',
        training='Same TRAIN-fitted PCA, C1 pairwise loss and parent normalization; same gate/strict rule.',
        validation_role='Reused development set; no unbiased confidence claim.',test_opened=False))
    tick=time.perf_counter()
    from round227_quadratic import inputs
    bank,arrays,y,_=load('train');e,g=inputs(bank,arrays)
    answers=metadata(bank)['answers']
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        c={k:z[k] for k in ('folds','held_s1','val_s1','strict_nested','strict_val')}
    held=c['folds']==2;rows=np.flatnonzero(~held)
    bh=[b for b,k in zip(bank,held) if k]
    hw=dict(np.load(ROOT/'experiments/round227/held_model.npz'))
    hweight=np.load(ROOT/'experiments/round234/context_ground_held.npz')['weight']
    hs=old_design(e[held],g[held],hw,'context_ground')@hweight
    oldh=strict_choice(bh,c['held_s1']+.25*(hs-hs.mean(1,keepdims=True)),c['strict_nested'][held])
    assert int(y[held][np.arange(held.sum()),oldh].sum())==1479
    records=[];val=fw=oldv=None
    for arm in ARMS:
        x=design(e,g,hw,answers,arm)
        weight=fit(x,y,rows);hs=x[held]@weight
        np.savez(OUT/(arm+'_held.npz'),weight=weight,scores=hs)
        eligible=[]
        for alpha in ALPHAS:
            choice=strict_choice(bh,c['held_s1']+alpha*(hs-hs.mean(1,keepdims=True)),c['strict_nested'][held])
            rec=dict(arm=arm,alpha=alpha,held=record(y[held],choice,oldh));records.append(rec)
            if rec['held']['correct']>=1479:eligible.append(rec)
            print('HELD',rec,'seconds',time.perf_counter()-tick,flush=True)
        del x
        if not eligible:continue
        if val is None:
            fw=dict(np.load(ROOT/'experiments/round235/bundle/quadratic.npz'))
            val,va,vy,_=load('validation');ve,vg=inputs(val,va);vanswers=metadata(val)['answers']
            vs=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
            oldv=strict_choice(val,c['val_s1']+.25*(vs-vs.mean(1,keepdims=True)),c['strict_val'])
            assert int(vy[np.arange(len(vy)),oldv].sum())==763
        x=design(e,g,fw,answers,arm)
        weight=fit(x,y,np.arange(len(y)));del x
        vs=design(ve,vg,fw,vanswers,arm)@weight
        np.savez(OUT/(arm+'_full.npz'),**dict(fw,weight=weight))
        np.savez(OUT/(arm+'_validation.npz'),scores=vs)
        for rec in eligible:
            choice=strict_choice(val,c['val_s1']+rec['alpha']*(vs-vs.mean(1,keepdims=True)),c['strict_val'])
            rec['validation']=record(vy,choice,oldv)
            print('VALIDATION',rec,flush=True)
        dump(OUT/'progress.json',dict(records=records,test_opened=False))
    eligible=[r for r in records if r.get('validation',{}).get('correct',0)>763]
    selected=max(eligible,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['alpha'])) if eligible else None
    dump(OUT/'result.json',dict(records=records,selected=selected,seconds=time.perf_counter()-tick,test_opened=False))
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
