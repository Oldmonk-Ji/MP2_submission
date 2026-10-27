"""Nonlinear semantic regions condition numeric verification, TRAIN/VAL only."""
from round22_common import ROOT, load, dump, sha
from round234_semantic_verification import design as old_design, fit
from round224_target_conclusion import strict_choice
from round225_train import record
from round227_quadratic import inputs
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
import time

OUT=ROOT/'experiments/round237'
ARMS=['rectified_context','rbf_context']
ALPHAS=[.1,.25,.5]

def context(e,w):
    return np.clip(((e.mean(1)-w['qmean'])@w['qpc'].T)/w['qscale'],-6,6)

def fit_regions(e,w,rows,arm):
    w=dict(w);q=context(e,w)
    if arm=='rectified_context':
        h=np.maximum(q,0)
    else:
        km=MiniBatchKMeans(n_clusters=64,random_state=237237,n_init=3,batch_size=256,max_iter=100)
        km.fit(q[rows]);w['centers']=km.cluster_centers_.astype(np.float32)
        dist=cdist(q[rows],w['centers'],'sqeuclidean')
        w['kernel_width']=np.asarray(np.median(dist),np.float32)
        h=np.exp(-cdist(q,w['centers'],'sqeuclidean')/w['kernel_width'])
    w['region_mean']=h[rows].mean(0).astype(np.float32)
    w['region_scale']=h[rows].std(0).clip(.01).astype(np.float32)
    return w

def design(e,g,w,arm):
    x=old_design(e,g,w,'context_ground');q=context(e,w)
    h=np.maximum(q,0) if arm=='rectified_context' else np.exp(-cdist(q,w['centers'],'sqeuclidean')/w['kernel_width'])
    h=np.clip((h-w['region_mean'])/w['region_scale'],-6,6)
    g=np.clip((g-w['gmean'])/w['gscale'],-6,6)
    cross=(h[:,None,:,None]*g[:,:,None,:]).reshape(len(e),4,-1)/np.sqrt(h.shape[-1]*g.shape[-1])
    return np.ascontiguousarray(np.concatenate([x,cross],2),dtype=np.float32)

def main():
    threadpool_limits(4)
    round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hash=sha(__file__),arms=ARMS,alphas=ALPHAS,
        selection='VAL>763 and held>=1479; then validation, held, smaller alpha.',
        method='Numeric evidence conditioned on rectified or smooth semantic regions; fixed pairwise C1 and frozen gate.',
        region_fitting='Every center, bandwidth and normalization fitted on corresponding TRAIN partition only.',test_opened=False))
    tick=time.perf_counter();bank,arrays,y,_=load('train');e,g=inputs(bank,arrays)
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        c={k:z[k] for k in ('folds','held_s1','val_s1','strict_nested','strict_val')}
    held=c['folds']==2;rows=np.flatnonzero(~held);bh=[b for b,k in zip(bank,held) if k]
    hw=dict(np.load(ROOT/'experiments/round227/held_model.npz'))
    hs=old_design(e[held],g[held],hw,'context_ground')@np.load(ROOT/'experiments/round234/context_ground_held.npz')['weight']
    oldh=strict_choice(bh,c['held_s1']+.25*(hs-hs.mean(1,keepdims=True)),c['strict_nested'][held])
    records=[];val=fw=oldv=None
    for arm in ARMS:
        w=fit_regions(e,hw,rows,arm);x=design(e,g,w,arm);weight=fit(x,y,rows);hs=x[held]@weight;del x
        w['weight']=weight;np.savez(OUT/(arm+'_held.npz'),**w)
        np.savez(OUT/(arm+'_held_scores.npz'),scores=hs)
        eligible=[]
        for alpha in ALPHAS:
            choice=strict_choice(bh,c['held_s1']+alpha*(hs-hs.mean(1,keepdims=True)),c['strict_nested'][held])
            rec=dict(arm=arm,alpha=alpha,held=record(y[held],choice,oldh));records.append(rec)
            if rec['held']['correct']>=1479:eligible.append(rec)
            print('HELD',rec,'seconds',time.perf_counter()-tick,flush=True)
        if not eligible:continue
        if val is None:
            fw=dict(np.load(ROOT/'experiments/round235/bundle/quadratic.npz'))
            val,va,vy,_=load('validation');ve,vg=inputs(val,va)
            vs=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
            oldv=strict_choice(val,c['val_s1']+.25*(vs-vs.mean(1,keepdims=True)),c['strict_val'])
        w=fit_regions(e,fw,np.arange(len(y)),arm);x=design(e,g,w,arm);weight=fit(x,y,np.arange(len(y)));del x
        vs=design(ve,vg,w,arm)@weight;w['weight']=weight
        np.savez(OUT/(arm+'_full.npz'),**w);np.savez(OUT/(arm+'_validation.npz'),scores=vs)
        for rec in eligible:
            choice=strict_choice(val,c['val_s1']+rec['alpha']*(vs-vs.mean(1,keepdims=True)),c['strict_val'])
            rec['validation']=record(vy,choice,oldv);print('VALIDATION',rec,flush=True)
        dump(OUT/'progress.json',dict(records=records,test_opened=False))
    eligible=[r for r in records if r.get('validation',{}).get('correct',0)>763]
    selected=max(eligible,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['alpha'])) if eligible else None
    dump(OUT/'result.json',dict(records=records,selected=selected,seconds=time.perf_counter()-tick,test_opened=False))
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
