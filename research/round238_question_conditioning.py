"""Question-only lexical context x numeric verification, added to round235."""
from round22_common import ROOT, load, dump, sha
from round234_semantic_verification import design as old_design, fit
from round224_target_conclusion import strict_choice
from round225_train import record
from round227_quadratic import inputs
from round238_context_features import hashed, fit_context, context
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
import time

OUT=ROOT/'experiments/round238'
ALPHAS=[.1,.25,.5]

def design(e,g,text,w):
    base=old_design(e,g,w,'context_ground')
    h=context(text,w)
    g=np.clip((g-w['gmean'])/w['gscale'],-6,6)
    cross=(h[:,None,:,None]*g[:,:,None,:]).reshape(len(e),4,-1)/np.sqrt(16*g.shape[-1])
    return np.ascontiguousarray(np.concatenate([base,cross],2),dtype=np.float32)

def main():
    threadpool_limits(4)
    round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hashes={n:sha(ROOT/'research'/n) for n in ('round238_context_features.py','round238_question_conditioning.py')},alphas=ALPHAS,
        method='Question-only hashed4096 TF-IDF -> SVD16, FP16 projection, crossed with numeric/ground evidence.',
        training='IDF/SVD/normalization fit corresponding TRAIN rows; same pairwise C1 loss and gate.',
        selection='VAL>763 and held>=1479, then validation, held, smaller alpha.',test_opened=False))
    tick=time.perf_counter();bank,arrays,y,_=load('train');e,g=inputs(bank,arrays);text=hashed(bank)
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        c={k:z[k] for k in ('folds','held_s1','val_s1','strict_nested','strict_val')}
    held=c['folds']==2;rows=np.flatnonzero(~held);bh=[b for b,k in zip(bank,held) if k]
    hw=dict(np.load(ROOT/'experiments/round227/held_model.npz'))
    hs=old_design(e[held],g[held],hw,'context_ground')@np.load(ROOT/'experiments/round234/context_ground_held.npz')['weight']
    oldh=strict_choice(bh,c['held_s1']+.25*(hs-hs.mean(1,keepdims=True)),c['strict_nested'][held])
    w=fit_context(text,rows,hw);x=design(e,g,text,w);weight=fit(x,y,rows);hs=x[held]@weight;del x
    w['weight']=weight;np.savez(OUT/'held_model.npz',**w);np.savez(OUT/'held_scores.npz',scores=hs)
    records=[];eligible=[]
    for alpha in ALPHAS:
        choice=strict_choice(bh,c['held_s1']+alpha*(hs-hs.mean(1,keepdims=True)),c['strict_nested'][held])
        rec=dict(alpha=alpha,held=record(y[held],choice,oldh));records.append(rec)
        if rec['held']['correct']>=1479:eligible.append(rec)
        print('HELD',rec,'seconds',time.perf_counter()-tick,flush=True)
    if eligible:
        fw=dict(np.load(ROOT/'experiments/round235/bundle/quadratic.npz'));w=fit_context(text,np.arange(len(y)),fw)
        x=design(e,g,text,w);weight=fit(x,y,np.arange(len(y)));del x
        val,va,vy,_=load('validation');ve,vg=inputs(val,va)
        vs=design(ve,vg,hashed(val),w)@weight;w['weight']=weight
        np.savez(OUT/'full_model.npz',**w);np.savez(OUT/'validation_scores.npz',scores=vs)
        oldvs=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
        oldv=strict_choice(val,c['val_s1']+.25*(oldvs-oldvs.mean(1,keepdims=True)),c['strict_val'])
        for rec in eligible:
            choice=strict_choice(val,c['val_s1']+rec['alpha']*(vs-vs.mean(1,keepdims=True)),c['strict_val'])
            rec['validation']=record(vy,choice,oldv);print('VALIDATION',rec,flush=True)
    eligible=[r for r in records if r.get('validation',{}).get('correct',0)>763]
    selected=max(eligible,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['alpha'])) if eligible else None
    dump(OUT/'result.json',dict(records=records,selected=selected,seconds=time.perf_counter()-tick,test_opened=False))
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
