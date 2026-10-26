from round22_common import ROOT,load,read,dump,sha
from round107_latent import subset,predict
from round110_alignment import supervision,loss_gradient,checks
from scipy import sparse
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round117'

def fit(x,ptr,y,gold,active,ix,ids,stage):
    warm=ROOT/f'experiments/round107/latent/{stage}.npz';allowed=ids[ix].tolist()
    assert read(warm.with_suffix('.json'))['training_ids']==allowed
    with np.load(warm) as z:w=z['weight'].astype(np.float64)
    candidates=(ix[:,None]*4+np.arange(4)).ravel();rows=np.concatenate([np.arange(ptr[c],ptr[c+1]) for c in candidates])
    xx,pp=subset(x,ptr,ix);gg=gold[rows];aa=active[candidates];pair,coef=supervision(y[ix],pp,gg,aa)
    state={};history=[];tick=time.perf_counter()
    def fun(w):
        loss,grad=loss_gradient(w,xx,pp,gg,pair,coef);state.update(loss=loss,gradient_max=float(np.abs(grad).max()));return loss,grad
    initial=fun(w)[0]
    def callback(w):
        history.append(dict(iteration=len(history)+1,**state))
        if len(history)%25==0:print('ITERATION',stage,len(history),state['loss'],flush=True)
    result=minimize(fun,w,method='L-BFGS-B',jac=True,callback=callback,options=dict(maxiter=200,ftol=1e-8,gtol=1e-5,maxls=30))
    assert np.isfinite(result.fun) and result.fun<=initial+1e-5
    weight=result.x.astype(np.float32);np.savez(OUT/f'{stage}.npz',weight=weight)
    report=dict(training_ids=allowed,alignment_parent_ids=ids[ix][aa.reshape(-1,4).any(1)].tolist(),active_candidates=int(aa.sum()),warm_start=str(warm.relative_to(ROOT)),initial_loss=initial,final_loss=float(result.fun),optimizer_success=bool(result.success),optimizer_message=str(result.message),iterations=int(result.nit),seconds=time.perf_counter()-tick,history=history,test_opened=False)
    dump(OUT/f'{stage}.json',report);print('FIT_COMPLETE',stage,{k:v for k,v in report.items() if k not in ('training_ids','alignment_parent_ids','history')},flush=True)
    return weight

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter()
    assert read(ROOT/'experiments/round116/full_stats.json')['test_opened']==False
    for src,h in read(ROOT/'experiments/round116/plan.json')['source_hashes'].items():assert sha(ROOT/src)==h,src
    dump(OUT/'plan.json',dict(checks=checks(),objective='Half candidate ranking plus half reference-equivalent root alignment for aligned parents; other parents retain ranking. C=1 and fixed maximum 200 iterations.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round117*.py')},test_opened=False))
    b,a,y,ids=load('train');base=ROOT/'experiments/round107';x=sparse.load_npz(base/'train_root_features.npz');ptr=np.load(base/'train_root_ptr.npy');vx=sparse.load_npz(base/'validation_root_features.npz');vp=np.load(base/'validation_root_ptr.npy')
    gold=np.load(ROOT/'experiments/round116/full_gold_roots.npy');active=np.load(ROOT/'experiments/round116/full_active_candidates.npy')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    outputs=dict(oof=np.zeros((len(b),4)),nested=np.zeros((len(b),4)))
    stages=[(f'oof{k}',np.flatnonzero(f!=k),np.flatnonzero(f==k),'oof') for k in range(3)]+[(f'nested{k}',np.flatnonzero(f==1-k),np.flatnonzero(f==k),'nested') for k in (0,1)]+[('full',np.arange(len(b)),None,'validation')]
    for stage,ix,te,kind in stages:
        w=fit(x,ptr,y,gold,active,ix,ids,stage);xx,pp=(vx,vp) if te is None else subset(x,ptr,te);s=predict(xx,pp,w)
        if te is None:outputs[kind]=s
        else:outputs[kind][te]=s
    outputs['nested'][f==2]=outputs['oof'][f==2];np.savez(OUT/'aligned_scores.npz',**outputs)
    te=f==2;choice=outputs['oof'][te].argmax(1);report=dict(correct=int(y[te][np.arange(te.sum()),choice].sum()),n=int(te.sum()),parse_gate_applied=False)
    dump(OUT/'held_standalone.json',report);print('STANDALONE',report,flush=True)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
