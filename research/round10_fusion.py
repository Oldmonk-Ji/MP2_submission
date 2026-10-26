"""Train-only calibrated probability fusion over frozen existing experts.
No new neural model and no relation extractor. Never opens test data.
"""
from data_access import ROOT,load
import json,hashlib,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax,logsumexp
from sklearn.model_selection import train_test_split
from features import metadata
from round4_gates import apply_gate
from round9_group_features import normalize
OUT=ROOT/'experiments/round10'

def arrays(bank,raw,base,normalized):
    z=np.concatenate([raw,base[:,:,None]],2).astype(float)
    if normalized:z=np.stack([normalize(z[:,:,k]) for k in range(z.shape[2])],2)
    else:
        # Global expert units from train data are folded into learned temperatures;
        # no question-wise rescaling, so confidence magnitudes survive.
        pass
    meta=metadata(bank);eq=np.array([[[a is not None and a==b for b in row] for a in row] for row in meta['answers']],float)
    return z,eq,meta['valid']

def fuse(z,eq,valid,params,pooling,grouping):
    m=z.shape[2];temperatures=np.exp(params[:m]);w=softmax(params[m:2*m]);logp=z/temperatures
    logp=np.where(valid[:,:,None],logp,-1e4);logp-=logsumexp(logp,axis=1,keepdims=True)
    if pooling=='probability':p=np.exp(logp)@w
    else:
        v=logp@w;p=softmax(v,axis=1)
    mass=np.einsum('nij,nj->ni',eq,p);count=np.maximum(eq.sum(2),1)
    if grouping=='mean':mass/=count
    # Re-normalize over unique numeric answers; duplicates count only once.
    denom=(mass/count).sum(1,keepdims=True);pgroup=mass/np.maximum(denom,1e-12)
    return np.where(valid,np.log(np.maximum(pgroup,1e-12)),-1e4)

def fit(z,eq,valid,y,pooling,grouping):
    m=z.shape[2];scale=np.maximum(z.std((0,1)),.05);initial=np.r_[np.log(scale),np.zeros(m)]
    correct=y.astype(float)/np.maximum(eq.sum(2),1)
    def objective(p):
        s=fuse(z,eq,valid,p,pooling,grouping)
        likelihood=(np.exp(s)*correct).sum(1)
        return -np.log(np.maximum(likelihood,1e-12)).mean()+.0005*np.sum((p[:m]-initial[:m])**2)+.0005*np.sum(p[m:]**2)
    bounds=[(np.log(v)-3,np.log(v)+3) for v in scale]+[(-6,6)]*m
    result=minimize(objective,initial,method='L-BFGS-B',bounds=bounds,options={'maxiter':150,'ftol':1e-9})
    return result.x,dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),objective=float(result.fun))

def gated(bank,scores,base):
    # Transfer only the existing baseline gate when its original choice survives.
    # Fusion cannot change the gate's score scale or loosen its margin threshold.
    cfg=json.loads((ROOT/'experiments/round4/selected_bundle/config.json').read_text())
    old=base.argmax(1);gate=apply_gate(bank,base,cfg['gate_kind'],cfg['margin_limit']);new=scores.argmax(1)
    return np.where((old!=gate)&(new==old),gate,new)

def main():
    OUT.mkdir(exist_ok=True);tick=time.perf_counter()
    specs=[dict(normalized=n,pooling=p,grouping=g) for n in (False,True) for p in ('probability','logit') for g in ('sum','mean')]
    (OUT/'plan.json').write_text(json.dumps(dict(specs=specs,protocol='Train OOF expert scores only; same fixed question holdout, selected weights/temperatures learned on fit portion. Eight fusion forms fixed before validation. Existing OOF reuse is not fully nested.',selection='Best probability variant and best logit variant internally, require probability improvement >=2 questions for validation',test_access=False,validation='One frozen comparison only; do not tune weights on validation'),indent=2))
    b,_,y=load('train');cache=np.load(ROOT/'experiments/round3/stack_scores.npz');raw=cache['oof_members'];base=cache['meta_oof']
    fitids,hold=train_test_split(np.arange(len(y)),test_size=.2,random_state=20261014,stratify=y.sum(1));np.savez(OUT/'internal_split.npz',fit=fitids,holdout=hold)
    baseline=gated(b,base,base);bc=int(y[hold,baseline[hold]].sum());records=[]
    for spec in specs:
        z,eq,valid=arrays(b,raw,base,spec['normalized']);params,info=fit(z[fitids],eq[fitids],valid[fitids],y[fitids],spec['pooling'],spec['grouping'])
        s=fuse(z,eq,valid,params,spec['pooling'],spec['grouping']);c=gated(b,s,base);oldok=y[hold,baseline[hold]]>0;ok=y[hold,c[hold]]>0
        record=dict(**spec,correct=int(ok.sum()),n=len(hold),fixed=int((ok&~oldok).sum()),lost=int((~ok&oldok).sum()),optimization=info);records.append(record);print(json.dumps(record),flush=True)
    chosen={p:max([r for r in records if r['pooling']==p],key=lambda r:r['correct']) for p in ('probability','logit')}
    for name,config in chosen.items():
        z,eq,valid=arrays(b,raw,base,config['normalized']);params,info=fit(z,eq,valid,y,name,config['grouping']);np.savez(OUT/f'{name}.npz',params=params)
        config['full_fit']=info
    selection=dict(selected=chosen,records=records,baseline_correct=bc,n=len(hold),internal_gate_passed=chosen['probability']['correct']>=bc+2,seconds=time.perf_counter()-tick)
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    paths=[ROOT/'research/round10_fusion.py',OUT/'plan.json',OUT/'selection.json',OUT/'probability.npz',OUT/'logit.npz']
    (OUT/'frozen.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    print('BASE',bc,'GATE',selection['internal_gate_passed'],flush=True)

if __name__=='__main__':main()
