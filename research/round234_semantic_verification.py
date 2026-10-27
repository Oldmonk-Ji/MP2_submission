"""Conditional numeric evidence in the existing TRAIN-fitted quadratic space."""
from round22_common import ROOT, load, dump, sha
from round227_quadratic import inputs, transform
from round224_target_conclusion import strict_choice
from round225_train import record
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
import time

OUT=ROOT/'experiments/round234'
ARMS=['context_ground','residual_ground','both_ground']
ALPHAS=[.1,.25,.5]

def design(e,g,w,arm):
    x=transform(e,g,w)
    q=e.mean(1)
    r=np.clip(((e-q[:,None]-w['rmean'])@w['rpc'][:32].T)/w['rscale'][:32],-6,6)
    q=np.clip(((q-w['qmean'])@w['qpc'].T)/w['qscale'],-6,6)
    g=np.clip((g-w['gmean'])/w['gscale'],-6,6)
    blocks=[x]
    if arm in ('context_ground','both_ground'):
        blocks.append((q[:,None,:,None]*g[:,:,None,:]).reshape(len(e),4,-1)/np.sqrt(32*g.shape[-1]))
    if arm in ('residual_ground','both_ground'):
        blocks.append((r[:,:,:,None]*g[:,:,None,:]).reshape(len(e),4,-1)/np.sqrt(32*g.shape[-1]))
    return np.ascontiguousarray(np.concatenate(blocks,2),dtype=np.float32)

def fit(x,y,rows):
    d,weights=[],[]
    for i in rows:
        pos,neg=np.flatnonzero(y[i]==1),np.flatnonzero(y[i]==0)
        for a in pos:
            for b in neg:
                d.append(x[i,a]-x[i,b]);weights.append(1/(len(pos)*len(neg)))
    d=np.asarray(d,np.float32)
    model=LogisticRegression(C=1,fit_intercept=False,solver='liblinear',max_iter=500,tol=1e-5,random_state=227227)
    model.fit(np.concatenate([d,-d]),np.r_[np.ones(len(d)),np.zeros(len(d))],sample_weight=np.tile(np.asarray(weights)/2,2))
    return model.coef_[0].astype(np.float32)

def main():
    threadpool_limits(4)
    round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hash=sha(__file__),arms=ARMS,alphas=ALPHAS,
        method='Semantic context/residual crossed with deterministic numerical verification; same PCA spaces and fixed pairwise training.',
        selection='VAL>762 and held>=1479; choose validation then held then smaller alpha.',test_opened=False))
    tick=time.perf_counter()
    bank,arrays,y,_=load('train')
    e,g=inputs(bank,arrays)
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        c={k:z[k] for k in ('folds','held_s1','val_s1','strict_nested','strict_val')}
    held=c['folds']==2;tr=np.flatnonzero(~held)
    bh=[b for b,k in zip(bank,held) if k]
    hw=dict(np.load(ROOT/'experiments/round227/held_model.npz'))
    hs=np.load(ROOT/'experiments/round227/scores.npz')['held']
    oldh=strict_choice(bh,c['held_s1']+.25*(hs-hs.mean(1,keepdims=True)),c['strict_nested'][held])
    records=[];val=fw=oldv=None
    for arm in ARMS:
        x=design(e,g,hw,arm)
        weight=fit(x,y,tr);s=x[held]@weight
        np.savez(OUT/(arm+'_held.npz'),weight=weight,scores=s)
        eligible=[]
        for alpha in ALPHAS:
            chosen=strict_choice(bh,c['held_s1']+alpha*(s-s.mean(1,keepdims=True)),c['strict_nested'][held])
            rec=dict(arm=arm,alpha=alpha,held=record(y[held],chosen,oldh))
            records.append(rec)
            if rec['held']['correct']>=1479:eligible.append(rec)
            print('HELD',rec,'seconds',time.perf_counter()-tick,flush=True)
        del x
        if not eligible:continue
        if val is None:
            fw=dict(np.load(ROOT/'experiments/round228/full_227.npz'))
            val,va,vy,_=load('validation');ve,vg=inputs(val,va)
            vs=np.load(ROOT/'experiments/round228/validation_227.npz')['scores']
            oldv=strict_choice(val,c['val_s1']+.25*(vs-vs.mean(1,keepdims=True)),c['strict_val'])
        x=design(e,g,fw,arm)
        weight=fit(x,y,np.arange(len(y)));del x
        vs=design(ve,vg,fw,arm)@weight
        np.savez(OUT/(arm+'_full.npz'),**dict(fw,weight=weight))
        np.savez(OUT/(arm+'_validation.npz'),scores=vs)
        for rec in eligible:
            chosen=strict_choice(val,c['val_s1']+rec['alpha']*(vs-vs.mean(1,keepdims=True)),c['strict_val'])
            rec['validation']=record(vy,chosen,oldv)
            print('VALIDATION',rec,flush=True)
        dump(OUT/'progress.json',dict(records=records,test_opened=False))
    eligible=[r for r in records if r.get('validation',{}).get('correct',0)>762]
    selected=max(eligible,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['alpha'])) if eligible else None
    dump(OUT/'result.json',dict(records=records,selected=selected,seconds=time.perf_counter()-tick,test_opened=False))
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
