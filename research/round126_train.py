from round22_common import ROOT,load,read,dump,sha
from round126_objective import loss_gradient,checks
from scipy import sparse
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round126'

def fit(x,y,ix,ids,stage):
    warm=ROOT/f'experiments/round103/local/{stage}.npz';assert read(warm.with_suffix('.json'))['training_ids']==ids[ix].tolist()
    with np.load(warm) as z:w=z['weight'].astype(np.float64)
    xx=x[(ix[:,None]*4+np.arange(4)).ravel()];yy=y[ix];history=[];state={};tick=time.perf_counter()
    def fun(w):
        loss,g=loss_gradient(w,xx,yy);state.update(loss=loss,gradient_max=float(abs(g).max()));return loss,g
    initial=fun(w)[0]
    def callback(w):
        history.append(dict(iteration=len(history)+1,**state))
        if len(history)%25==0:print('ITERATION',stage,len(history),state['loss'],flush=True)
    result=minimize(fun,w,method='L-BFGS-B',jac=True,callback=callback,options=dict(maxiter=200,ftol=1e-8,gtol=1e-5,maxls=30));assert np.isfinite(result.fun) and result.fun<=initial+1e-5
    weight=result.x.astype(np.float32);np.savez(OUT/f'{stage}.npz',weight=weight)
    report=dict(training_ids=ids[ix].tolist(),warm_start=str(warm.relative_to(ROOT)),initial_loss=initial,final_loss=float(result.fun),optimizer_success=bool(result.success),optimizer_message=str(result.message),iterations=int(result.nit),seconds=time.perf_counter()-tick,history=history,test_opened=False)
    dump(OUT/f'{stage}.json',report);print('FIT_COMPLETE',stage,{k:v for k,v in report.items() if k not in ('training_ids','history')},flush=True);return weight

if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();OUT.mkdir();b,a,y,ids=load('train')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    mask=(f!=2)&(y.min(1)==0)&(y.max(1)>0);audit=dict(fit_mixed=int(mask.sum()),fit_mixed_multiple_correct=int((mask&(y.sum(1)>1)).sum()))
    dump(OUT/'plan.json',dict(checks=checks(),fit_audit=audit,method='Keep round103 local structural mean features; replace all-positive-versus-all-negative pair loss with minus log marginal softmax probability of the entire correct set. Original per-parent loss factor2 and L2 coefficient1 preserved. Fixed warm start on same allowed parents; maximum200 iterations.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round126*.py')},test_opened=False))
    x=sparse.load_npz(ROOT/'experiments/round103/train_local.npz');vx=sparse.load_npz(ROOT/'experiments/round103/validation_local.npz');outputs=dict(oof=np.zeros((len(b),4)),nested=np.zeros((len(b),4)))
    stages=[(f'oof{k}',np.flatnonzero(f!=k),np.flatnonzero(f==k),'oof') for k in range(3)]+[(f'nested{k}',np.flatnonzero(f==1-k),np.flatnonzero(f==k),'nested') for k in (0,1)]+[('full',np.arange(len(b)),None,'validation')]
    for stage,ix,te,kind in stages:
        w=fit(x,y,ix,ids,stage);xx=vx if te is None else x[(te[:,None]*4+np.arange(4)).ravel()];s=np.asarray(xx@w).reshape(-1,4)
        if te is None:outputs[kind]=s
        else:outputs[kind][te]=s
    outputs['nested'][f==2]=outputs['oof'][f==2];np.savez(OUT/'marginal_scores.npz',**outputs)
    te=f==2;c=outputs['oof'][te].argmax(1);r=dict(correct=int(y[te][np.arange(te.sum()),c].sum()),n=int(te.sum()),parse_gate_applied=False);dump(OUT/'held_standalone.json',r);print(r,flush=True)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
