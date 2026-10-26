"""Small confidence-aware linear correction. Fits train OOF scores only."""
from data_access import ROOT,load
import json,hashlib,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from features import metadata
from round9_group_features import normalize
from round4_gates import apply_gate
OUT=ROOT/'experiments/round13'

def features(raw,scale,kind):
    center=raw-raw.mean(1,keepdims=True);z=np.stack([normalize(raw[:,:,k]) for k in range(raw.shape[2])],2);u=center/scale
    if kind=='norm_only':return z
    p=softmax(u,axis=1);confidence=np.max(p,axis=1,keepdims=True)
    return np.concatenate([z,np.clip(u,-8,8),p-p.mean(1,keepdims=True),z*confidence,z*np.minimum(u.std(1,keepdims=True),4)],2)

def fit(x,base,y,ids,reg):
    d=[];offset=[];weight=[]
    for i in ids:
        pos=np.flatnonzero(y[i]);neg=np.flatnonzero(y[i]==0)
        for p in pos:
            for n in neg:d.append(x[i,p]-x[i,n]);offset.append(base[i,p]-base[i,n]);weight.append(1/(len(pos)*len(neg)))
    d=np.asarray(d);offset=np.asarray(offset);weight=np.asarray(weight);weight/=weight.sum()
    def objective(w):
        margin=offset+d@w;error=-1/(1+np.exp(np.clip(margin,-40,40)))
        return float(weight@np.logaddexp(0,-margin)+.5*reg*(w@w)),d.T@(weight*error)+reg*w
    result=minimize(objective,np.zeros(x.shape[-1]),jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-10});assert result.success,result.message
    return result.x

def main():
    OUT.mkdir(exist_ok=True);tick=time.perf_counter()
    (OUT/'plan.json').write_text(json.dumps(dict(kinds=['confidence','norm_only'],regularization=[1,.1,.01,.001],gate='>=2 more internal holdout correct than baseline; one frozen validation comparison if passed',hypothesis='Raw expert confidence was absent from the normalized stack; add globally calibrated raw margin/confidence interactions while retaining signed corrections.',protocol='Same train-only 3804/951 OOF reuse screening. Not fully nested. No relation model, no extra neural member.',test_access=False),indent=2))
    b,_,y=load('train');old=np.load(ROOT/'experiments/round3/stack_scores.npz');raw=old['oof_members'];base=old['meta_oof'];meta=metadata(b);gate=json.loads((ROOT/'experiments/round4/selected_bundle/config.json').read_text())
    tr,ho=train_test_split(np.arange(len(y)),test_size=.2,random_state=20261014,stratify=y.sum(1));hb=[b[i] for i in ho];baseline=apply_gate(hb,base[ho],gate['gate_kind'],gate['margin_limit']);bc=int(y[ho,baseline].sum());scale=np.maximum(raw[tr].std((0,1)),.05);records=[];selected={}
    for kind in ('confidence','norm_only'):
        x=features(raw,scale,kind)
        for reg in (1.,.1,.01,.001):
            w=fit(x,base,y,tr,reg);scores=base[ho]+x[ho]@w;choice=apply_gate(hb,scores,gate['gate_kind'],gate['margin_limit']);ok=y[ho,choice]>0;prev=y[ho,baseline]>0
            rec=dict(kind=kind,reg=reg,correct=int(ok.sum()),n=len(ho),fixed=int((ok&~prev).sum()),lost=int((~ok&prev).sum()));records.append(rec);print(json.dumps(rec),flush=True)
        best=max([r for r in records if r['kind']==kind],key=lambda r:r['correct']);selected[kind]=best
        fullscale=np.maximum(raw.std((0,1)),.05);w=fit(features(raw,fullscale,kind),base,y,np.arange(len(y)),best['reg']);np.savez(OUT/f'{kind}.npz',weight=w,scale=fullscale)
    decision=dict(selected=selected,records=records,baseline_correct=bc,n=len(ho),internal_gate_passed=selected['confidence']['correct']>=bc+2,seconds=time.perf_counter()-tick)
    (OUT/'selection.json').write_text(json.dumps(decision,indent=2))
    paths=[ROOT/'research/round13_confidence.py',OUT/'selection.json',OUT/'plan.json',OUT/'confidence.npz',OUT/'norm_only.npz']
    (OUT/'frozen.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2));print('BASE',bc,'GATE',decision['internal_gate_passed'],flush=True)

if __name__=='__main__':main()
