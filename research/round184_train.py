"""Strict-trace latent selection without treating reference mismatch as error."""
from round22_common import ROOT,load,read,dump,sha
from round107_latent import subset,pairs,loss_gradient,predict,checks
from scipy import sparse
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round184'

def fit(x,ptr,y,ix,ids,stage):
    warm=ROOT/f'experiments/round181/{stage}.npz';allowed=ids[ix].tolist();assert read(warm.with_suffix('.json'))['training_ids']==allowed
    with np.load(warm) as z:w=z['weight'].astype(np.float64)
    xx,pp=subset(x,ptr,ix);pair=pairs(y[ix]);history=[];state={};tick=time.perf_counter()
    def fun(w):
        loss,g=loss_gradient(w,xx,pp,pair);state.update(loss=loss,gradient_max=float(np.abs(g).max()));return loss,g
    initial=fun(w)[0]
    def callback(w):
        history.append(dict(iteration=len(history)+1,**state))
        if len(history)%25==0:print('ITER',stage,len(history),state['loss'],flush=True)
    result=minimize(fun,w,method='L-BFGS-B',jac=True,callback=callback,options=dict(maxiter=200,ftol=1e-8,gtol=1e-5,maxls=30));assert np.isfinite(result.fun) and result.fun<=initial+1e-5
    weight=result.x.astype(np.float32);np.savez(OUT/f'{stage}.npz',weight=weight);report=dict(training_ids=allowed,initial_loss=initial,final_loss=float(result.fun),optimizer_success=bool(result.success),optimizer_message=str(result.message),iterations=int(result.nit),seconds=time.perf_counter()-tick,history=history,test_opened=False);dump(OUT/f'{stage}.json',report);print('FIT',stage,report['iterations'],report['seconds'],flush=True);return weight

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();assert read(ROOT/'experiments/round183/fit_stats.json')['fit_only'];dump(OUT/'plan.json',dict(checks=checks(),source_sha256=sha(__file__),method='Fixed strict179 graphs and same103 per-root features. Ablate mean versus count-normalized logsumexp using frozen181 weights, then train latent branch with same parent-balanced final-answer pair loss,C1,maxiter200. No reference-mismatch negatives: fit review found valid half/complement, unit1, redundant totals and incorrect reference numeric aliases.',arms=['pool_only','latent'],test_opened=False))
    b,a,y,ids=load('train');base=ROOT/'experiments/round183';x=sparse.load_npz(base/'train_root_features.npz');ptr=np.load(base/'train_root_ptr.npy');vx=sparse.load_npz(base/'validation_root_features.npz');vp=np.load(base/'validation_root_ptr.npy')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    outputs={arm:dict(oof=np.zeros((len(b),4)),nested=np.zeros((len(b),4))) for arm in ('pool_only','latent')}
    stages=[(f'oof{k}',np.flatnonzero(f!=k),np.flatnonzero(f==k),'oof') for k in range(3)]+[(f'nested{k}',np.flatnonzero(f==1-k),np.flatnonzero(f==k),'nested') for k in (0,1)]+[('full',np.arange(len(b)),None,'validation')]
    for stage,ix,te,key in stages:
        w=fit(x,ptr,y,ix,ids,stage);xx,pp=(vx,vp) if te is None else subset(x,ptr,te)
        with np.load(ROOT/f'experiments/round181/{stage}.npz') as z:old=z['weight']
        for arm,weight in [('pool_only',old),('latent',w)]:
            s=predict(xx,pp,weight)
            if te is None:outputs[arm][key]=s
            else:outputs[arm][key][te]=s
    reports=[]
    for arm,values in outputs.items():
        values['nested'][f==2]=values['oof'][f==2];np.savez(OUT/f'{arm}_scores.npz',**values);te=f==2;p=values['oof'][te].argmax(1);r=dict(arm=arm,correct=int(y[te][np.arange(te.sum()),p].sum()),n=int(te.sum()),parse_gate_applied=False);reports.append(r);print('STANDALONE',r,flush=True)
    dump(OUT/'held_standalone.json',reports);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
