from round22_common import ROOT,load,read,dump,sha
from round107_latent import subset,pairs,loss_gradient,predict,checks
from scipy import sparse
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round107'

def fit(x,ptr,y,ix,ids,stage):
    allowed=ids[ix].tolist();warm=ROOT/f'experiments/round103/local/{stage}.npz';assert read(warm.with_suffix('.json'))['training_ids']==allowed
    with np.load(warm) as z:w=z['weight'].astype(np.float64)
    xx,pp=subset(x,ptr,ix);pair=pairs(y[ix]);history=[];state={};tick=time.perf_counter()
    def fun(w):
        loss,grad=loss_gradient(w,xx,pp,pair);state.update(loss=loss,gradient_max=float(np.abs(grad).max()));return loss,grad
    initial=fun(w)[0]
    def callback(w):
        history.append(dict(iteration=len(history)+1,**state))
        if len(history)%25==0:print('ITERATION',stage,len(history),state['loss'],flush=True)
    result=minimize(fun,w,method='L-BFGS-B',jac=True,callback=callback,options=dict(maxiter=200,ftol=1e-8,gtol=1e-5,maxls=30));assert np.isfinite(result.fun) and result.fun<=initial+1e-5
    path=OUT/'latent'/f'{stage}.npz';path.parent.mkdir(exist_ok=True);weight=result.x.astype(np.float32);np.savez(path,weight=weight)
    report=dict(training_ids=allowed,warm_start=str(warm.relative_to(ROOT)),initial_loss=initial,final_loss=float(result.fun),optimizer_success=bool(result.success),optimizer_message=str(result.message),iterations=int(result.nit),function_calls=int(result.nfev),seconds=time.perf_counter()-tick,C=1,history=history,test_opened=False)
    dump(path.with_suffix('.json'),report);print('FIT_COMPLETE',stage,{k:v for k,v in report.items() if k not in ('training_ids','history')},flush=True);return weight

if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();assert read(OUT/'preparation_complete.json')['test_opened']==False;checks();assert not (OUT/'training_plan.json').exists()
    for src,h in read(OUT/'plan.json')['source_hashes'].items():assert sha(ROOT/src)==h,src
    dump(OUT/'training_plan.json',dict(source_sha256=sha(__file__),objective='Same parent-balanced original candidate-pair loss and C=1 as round103; replace uniform root averaging by normalized logsumexp. Pool-only arm reuses frozen round103 weights.',max_iterations=200,warm_start_parent_isolation=True,test_opened=False))
    b,a,y,ids=load('train');x=sparse.load_npz(OUT/'train_root_features.npz');ptr=np.load(OUT/'train_root_ptr.npy');vx=sparse.load_npz(OUT/'validation_root_features.npz');vptr=np.load(OUT/'validation_root_ptr.npy')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    outputs={arm:dict(oof=np.zeros((len(b),4)),nested=np.zeros((len(b),4))) for arm in ('pool_only','latent')}
    stages=[(f'oof{k}',np.flatnonzero(f!=k),np.flatnonzero(f==k),'oof') for k in range(3)]+[(f'nested{k}',np.flatnonzero(f==1-k),np.flatnonzero(f==k),'nested') for k in (0,1)]+[('full',np.arange(len(b)),None,'validation')]
    for stage,ix,te,kind in stages:
        w=fit(x,ptr,y,ix,ids,stage)
        xx,pp=(vx,vptr) if te is None else subset(x,ptr,te)
        with np.load(ROOT/f'experiments/round103/local/{stage}.npz') as z:old=z['weight']
        for arm,weight in [('pool_only',old),('latent',w)]:
            s=predict(xx,pp,weight)
            if te is None:outputs[arm][kind]=s
            else:outputs[arm][kind][te]=s
    reports=[]
    for arm,values in outputs.items():
        values['nested'][f==2]=values['oof'][f==2];np.savez(OUT/f'{arm}_scores.npz',**values);te=f==2;choice=values['oof'][te].argmax(1);r=dict(arm=arm,correct=int(y[te][np.arange(te.sum()),choice].sum()),n=int(te.sum()),parse_gate_applied=False);reports.append(r);print('STANDALONE',r,flush=True)
    dump(OUT/'held_standalone.json',reports);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
