"""Direct lexical conditioning of numerical evidence; TRAIN/VAL only."""
from round22_common import ROOT,load,dump,sha
from round227_quadratic import inputs
from round234_semantic_verification import design
from round239_features import questions,fit_scale,build
from round224_target_conclusion import strict_choice
from round225_train import record
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
import time

OUT=ROOT/'experiments/round239'
ALPHAS=[.1,.25,.5]

def fit(text,g,y,rows):
    w=fit_scale(g,rows);x=build(text,g,w)
    a=[];b=[];sample=[]
    for i in rows:
        pos,neg=np.flatnonzero(y[i]==1),np.flatnonzero(y[i]==0)
        for j in pos:
            for k in neg:
                a.append(4*i+j);b.append(4*i+k);sample.append(1/(len(pos)*len(neg)))
    delta=x[a]-x[b]
    model=LogisticRegression(C=1,fit_intercept=False,solver='liblinear',max_iter=500,tol=1e-5,random_state=239239)
    model.fit(sparse.vstack([delta,-delta]),np.r_[np.ones(len(a)),np.zeros(len(a))],sample_weight=np.tile(np.asarray(sample)/2,2))
    w['weight']=model.coef_[0].astype(np.float32)
    return w,np.asarray(x@w['weight']).reshape(-1,4)

def main():
    threadpool_limits(4)
    round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hashes={n:sha(ROOT/'research'/n) for n in ('round239_features.py','round239_train.py')},alphas=ALPHAS,
        method='Hashed question unigrams/bigrams crossed directly with 32 TRAIN-selected varying numerical features;32768 weights;fixed C1 pairwise loss.',
        selection='VAL>763 and held>=1479, then validation,held,smaller alpha.',test_opened=False))
    tick=time.perf_counter();bank,arrays,y,_=load('train');e,g=inputs(bank,arrays);text=questions(bank)
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        c={k:z[k] for k in ('folds','held_s1','val_s1','strict_nested','strict_val')}
    held=c['folds']==2;bh=[b for b,k in zip(bank,held) if k]
    hw=dict(np.load(ROOT/'experiments/round227/held_model.npz'))
    hs=design(e[held],g[held],hw,'context_ground')@np.load(ROOT/'experiments/round234/context_ground_held.npz')['weight']
    baseh=c['held_s1']+.25*(hs-hs.mean(1,keepdims=True))
    oldh=strict_choice(bh,baseh,c['strict_nested'][held])
    w,s=fit(text,g,y,np.flatnonzero(~held));hs=s[held]
    np.savez(OUT/'held_model.npz',**w);np.savez(OUT/'held_scores.npz',scores=hs)
    records=[];eligible=[]
    for alpha in ALPHAS:
        choice=strict_choice(bh,baseh+alpha*(hs-hs.mean(1,keepdims=True)),c['strict_nested'][held])
        rec=dict(alpha=alpha,held=record(y[held],choice,oldh));records.append(rec)
        if rec['held']['correct']>=1479:eligible.append(rec)
        print('HELD',rec,'seconds',time.perf_counter()-tick,flush=True)
    if eligible:
        w,_=fit(text,g,y,np.arange(len(y)));val,va,vy,_=load('validation');_,vg=inputs(val,va)
        vs=np.asarray(build(questions(val),vg,w)@w['weight']).reshape(-1,4)
        np.savez(OUT/'full_model.npz',**w);np.savez(OUT/'validation_scores.npz',scores=vs)
        oldvs=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
        basev=c['val_s1']+.25*(oldvs-oldvs.mean(1,keepdims=True));oldv=strict_choice(val,basev,c['strict_val'])
        for rec in eligible:
            choice=strict_choice(val,basev+rec['alpha']*(vs-vs.mean(1,keepdims=True)),c['strict_val'])
            rec['validation']=record(vy,choice,oldv);print('VALIDATION',rec,flush=True)
    eligible=[r for r in records if r.get('validation',{}).get('correct',0)>763]
    selected=max(eligible,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['alpha'])) if eligible else None
    dump(OUT/'result.json',dict(records=records,selected=selected,seconds=time.perf_counter()-tick,test_opened=False))
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
