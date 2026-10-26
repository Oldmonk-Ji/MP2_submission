"""Learn a small question-conditioned offset using train pairs and fixed base scores."""
from data_access import ROOT,load
import json,hashlib,time
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from round11_context import transform,DIM
from round4_gates import apply_gate
from round9_train import metric
from features import metadata
OUT=ROOT/'experiments/round11'

def fit(x,y,base,ids,reg):
    pos=[];neg=[];ww=[];off=[]
    for i in ids:
        pp=np.flatnonzero(y[i]);nn=np.flatnonzero(y[i]==0)
        for p in pp:
            for n in nn:
                pos.append(i*4+p);neg.append(i*4+n);ww.append(1/(len(pp)*len(nn)));off.append(base[i,p]-base[i,n])
    d=x[pos]-x[neg];ww=np.asarray(ww);ww/=ww.sum();off=np.asarray(off)
    def objective(w):
        margin=off+d@w;err=-1/(1+np.exp(np.clip(margin,-40,40)))
        return np.dot(ww,np.logaddexp(0,-margin))+.5*reg*np.dot(w,w),np.asarray(d.T@(ww*err)).ravel()+reg*w
    result=minimize(objective,np.zeros(x.shape[1]),method='L-BFGS-B',jac=True,options={'maxiter':120,'ftol':1e-8,'gtol':1e-6,'maxcor':6})
    return result.x.astype(np.float32),dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))

def main():
    OUT.mkdir(exist_ok=True);tick=time.perf_counter()
    plan=dict(context=[True,False],regularization=[.1,.01,.001],alpha=[.25,.5,1],dimensions=DIM,protocol='Train OOF offset logistic ranking; fixed 3804/951 question split; selected by internal accuracy. Reused base OOF is not a fully nested estimate.',gate='At least 2 more internal correct than baseline before validation',interpretation='Learned context/trace compatibility, not verified semantic constraints; no failed relation extractor used',test_access=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    b,_,y=load('train');base=np.load(ROOT/'experiments/round3/stack_scores.npz')['meta_oof'];meta=metadata(b);hard=meta['hard'];gate=json.loads((ROOT/'experiments/round4/selected_bundle/config.json').read_text());baseline=apply_gate(b,base,gate['gate_kind'],gate['margin_limit'])
    tr,hold=train_test_split(np.arange(len(y)),test_size=.2,random_state=20261014,stratify=y.sum(1));records=[];selected={}
    np.savez(OUT/'internal_split.npz',fit=tr,holdout=hold)
    for context in (True,False):
        cache=OUT/f'features_{context}.npz'
        if cache.exists():x=sparse.load_npz(cache)
        else:
            print('EXTRACT',context,flush=True);x=transform(b,context);sparse.save_npz(cache,x)
        for reg in plan['regularization']:
            w,info=fit(x,y,base,tr,reg);delta=np.asarray(x@w).reshape(-1,4)
            for alpha in plan['alpha']:
                c=apply_gate(b,base+alpha*delta,gate['gate_kind'],gate['margin_limit']);rec=dict(context=context,reg=reg,alpha=alpha,internal=metric(c[hold],y[hold],hard[hold],baseline[hold]),optimization=info);records.append(rec);print(json.dumps(rec),flush=True)
        best=max([r for r in records if r['context']==context],key=lambda r:r['internal']['correct']);selected[str(context)]=best
        w,info=fit(x,y,base,np.arange(len(y)),best['reg']);np.savez(OUT/f'context_{context}.npz',weight=w);best['full_fit']=info
    bc=int(y[hold,baseline[hold]].sum());result=dict(selected=selected,records=records,baseline_correct=bc,n=len(hold),internal_gate_passed=selected['True']['internal']['correct']>=bc+2,seconds=time.perf_counter()-tick)
    (OUT/'selection.json').write_text(json.dumps(result,indent=2))
    paths=[ROOT/'research/round11_context.py',ROOT/'research/round11_train.py',OUT/'plan.json',OUT/'selection.json',OUT/'context_True.npz',OUT/'context_False.npz']
    (OUT/'frozen.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2));print('BASE',bc,'GATE',result['internal_gate_passed'],flush=True)

if __name__=='__main__':main()
