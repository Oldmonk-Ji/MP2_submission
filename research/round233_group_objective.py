"""TRAIN/VAL-only group objectives in the existing frozen quadratic space."""
from round22_common import ROOT, load, dump, sha
from round227_quadratic import inputs, transform
from round224_target_conclusion import strict_choice
from round225_train import record
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
import time

OUT = ROOT / 'experiments/round233'
VARIANTS = [('uniform_positive', False), ('any_positive', False), ('any_positive', True)]
ALPHAS = [.1, .25, .5, 1.]

def fit_group(x, y, offset, initial, objective):
    # Zero-positive groups have no selectable correct option and no ranking gradient.
    keep = (y.sum(1) > 0) & (y.sum(1) < 4)
    x, y, offset = x[keep].astype(np.float64), y[keep], offset[keep]
    target = y / y.sum(1, keepdims=True)
    def loss(w):
        s = x @ w + offset
        all_lse = logsumexp(s, axis=1)
        probability = softmax(s, axis=1)
        if objective == 'uniform_positive':
            value = (all_lse - (target*s).sum(1)).sum()
            residual = probability-target
        else:
            positive = np.where(y > 0, s, -np.inf)
            value = (all_lse-logsumexp(positive, axis=1)).sum()
            residual = probability-softmax(positive, axis=1)
        return value + .5*np.dot(w,w), np.einsum('nc,ncd->d', residual,x) + w
    result = minimize(loss, initial.astype(np.float64), method='L-BFGS-B', jac=True,
                      options=dict(maxiter=180, ftol=1e-8, gtol=1e-5, maxcor=12))
    return result.x.astype(np.float32), dict(loss=float(result.fun), iterations=int(result.nit), success=bool(result.success), message=str(result.message))

def main():
    threadpool_limits(4)
    round4_gates.trace = lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json', dict(source_hash=sha(__file__), variants=VARIANTS, alphas=ALPHAS,
         purpose='Change ranking objective, retain TRAIN-fitted spaces, dimension and inference architecture.',
         selection='Validation>762 and held>=1479; then validation, held, fewer changes.',
         validation_role='Reused development set', test_opened=False))
    tick = time.perf_counter()
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        c = {k:z[k] for k in ('folds','held_s1','val_s1','strict_nested','strict_val')}
    bank, arrays, y, _ = load('train')
    fit, held = c['folds'] != 2, c['folds'] == 2
    bh = [b for b,k in zip(bank,held) if k]
    e,g = inputs(bank,arrays)
    hw = dict(np.load(ROOT/'experiments/round227/held_model.npz'))
    x = transform(e,g,hw)
    hs = np.load(ROOT/'experiments/round227/scores.npz')['held']
    baseh = c['held_s1']+.25*(hs-hs.mean(1,keepdims=True))
    oldh = strict_choice(bh,baseh,c['strict_nested'][held])
    assert int(y[held][np.arange(held.sum()),oldh].sum()) == 1479
    # Existing cross-fitted fusion scores provide training offsets, without HELD labels.
    design = np.load(ROOT/'experiments/round225/fit_only_s1.npz')['scores']
    records=[]
    val=vy=vx=fw=oldv=None
    for objective, conditioned in VARIANTS:
        name=objective+('_residual' if conditioned else '_standalone')
        offsets = design[fit] if conditioned else np.zeros((fit.sum(),4))
        initial = .25*hw['weight'] if conditioned else hw['weight']
        weight,opt = fit_group(x[fit],y[fit],offsets,initial,objective)
        sh = x[held]@weight
        np.savez(OUT/(name+'_held.npz'),weight=weight,scores=sh)
        print('TRAINED',name,opt,'seconds',time.perf_counter()-tick,flush=True)
        eligible=[]
        for alpha in ALPHAS:
            choice = strict_choice(bh,c['held_s1']+alpha*(sh-sh.mean(1,keepdims=True)),c['strict_nested'][held])
            rec=dict(variant=name,alpha=alpha,optimizer=opt,held=record(y[held],choice,oldh))
            records.append(rec)
            if rec['held']['correct']>=1479: eligible.append(rec)
            print('HELD',rec,flush=True)
        if not eligible: continue
        if val is None:
            fw=dict(np.load(ROOT/'experiments/round228/full_227.npz'))
            xf=transform(e,g,fw)
            val,va,vy,_=load('validation')
            vx=transform(*inputs(val,va),fw)
            oldvs=np.load(ROOT/'experiments/round228/validation_227.npz')['scores']
            oldv=strict_choice(val,c['val_s1']+.25*(oldvs-oldvs.mean(1,keepdims=True)),c['strict_val'])
            assert int(vy[np.arange(len(vy)),oldv].sum())==762
        # Final model uses all eligible TRAIN questions. Offset is train OOF for all folds.
        if conditioned:
            offset=design.copy()
            offset[held]=c['held_s1']
        else: offset=np.zeros_like(y)
        initial=.25*fw['weight'] if conditioned else fw['weight']
        weight,optfull=fit_group(xf,y,offset,initial,objective)
        sw=vx@weight
        full=dict(fw,weight=weight)
        np.savez(OUT/(name+'_full.npz'),**full)
        np.savez(OUT/(name+'_validation.npz'),scores=sw)
        for rec in eligible:
            choice=strict_choice(val,c['val_s1']+rec['alpha']*(sw-sw.mean(1,keepdims=True)),c['strict_val'])
            rec['validation']=record(vy,choice,oldv)
            print('VALIDATION',rec,flush=True)
        dump(OUT/'progress.json',dict(records=records,test_opened=False))
    candidates=[r for r in records if r.get('validation',{}).get('correct',0)>762]
    chosen=max(candidates,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['validation']['changed'])) if candidates else None
    dump(OUT/'result.json',dict(records=records,selected=chosen,seconds=time.perf_counter()-tick,test_opened=False))
    print('SELECTED',chosen,flush=True)

if __name__=='__main__': main()
