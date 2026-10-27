"""Remove score-scale drift and old-expert-only overrides in the new arbiter."""
from round22_common import ROOT,load,read,dump,sha
from round250_features import build,choose
from round224_target_conclusion import strict_choice
from round225_train import record
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np

OUT=ROOT/'experiments/round250'
ARMS=['scale_free','text_anchored']

def fit(bank,y,s,strict,text,old,arm):
    x,eligible,names=build(bank,s,strict,text,old,arm)
    delta=y-y[np.arange(len(y)),old,None];mask=eligible&(delta!=0)
    weights=np.broadcast_to((1/np.maximum(mask.sum(1),1))[:,None],mask.shape)[mask]
    xx=x[mask];target=(delta[mask]>0).astype(np.float64);scale=xx.std(0).clip(.1)
    if arm=='scale_free':
        mean=xx.mean(0);z=np.clip((xx-mean)/scale,-6,6)
        model=LogisticRegression(C=1,solver='lbfgs',max_iter=1000)
        model.fit(z,target,sample_weight=weights)
        coef=model.coef_[0];intercept=model.intercept_[0]
    else:
        mean=np.zeros(xx.shape[1],np.float32);z=np.clip(xx/scale,-6,6).astype(np.float64)
        # At zero text evidence, retain the TRAIN benefit/harm prior. No free
        # old-expert-only offset can trigger a switch without textual support.
        prior=np.clip(np.sum(weights*target)/np.sum(weights),1e-5,1-1e-5)
        intercept=float(np.log(prior/(1-prior)))
        def loss(w):
            logits=z@w+intercept
            value=np.sum(weights*(np.logaddexp(0,logits)-target*logits))+.5*np.dot(w,w)
            grad=z.T@(weights*(expit(logits)-target))+w
            return value,grad
        result=minimize(loss,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-10))
        assert result.success,result.message
        coef=result.x
    w=dict(mean=mean.astype(np.float32),scale=scale.astype(np.float32),coef=coef.astype(np.float32),intercept=np.asarray(intercept,np.float32))
    return w,dict(features=names,benefit_pairs=int(target.sum()),harm_pairs=int((target==0).sum()),C=1,intercept=float(intercept))

def main():
    threadpool_limits(4);round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hashes={n:sha(ROOT/'research'/n) for n in ('round250_train.py','round250_features.py')},arms=ARMS,
        hypothesis='OOF-to-full scorer calibration drift or old-expert-only features cause new utility overrides to misfire. Enforce independent affine score invariance; optionally make all evidence proportional to text support.',
        structure='Top2 family fixed from highest TRAIN-held count in249 (not chosen from VAL). Binary benefit/harm, fixed C1 and zero-utility cutoff.',
        selection='VAL>763 and held>=1477, then validation,held,fewer harms.',test_opened=False))
    bank,_,y,_=load('train')
    with np.load(ROOT/'experiments/round216/cache.npz') as z:c={k:z[k] for k in ('folds','strict_nested','strict_oof','strict_val','val_s1')}
    with np.load(ROOT/'experiments/round248/scores.npz') as z:a={k:z[k] for k in z.files}
    held=c['folds']==2;tr=~held;bt=[b for b,k in zip(bank,tr) if k];bh=[b for b,k in zip(bank,held) if k]
    oldt=strict_choice(bt,a['design_base'][tr],c['strict_nested'][tr]);oldh=strict_choice(bh,a['held_base'],c['strict_nested'][held])
    oldoof=strict_choice(bank,a['oof_base'],c['strict_oof']);records=[];val=oldv=None
    for arm in ARMS:
        w,info=fit(bt,y[tr],a['design_base'][tr],c['strict_nested'][tr],a['design_text'][tr],oldt,arm)
        np.savez(OUT/f'{arm}_held.npz',**w)
        pred=choose(bh,a['held_base'],c['strict_nested'][held],a['held_text'],oldh,w,arm)
        rec=dict(arm=arm,training=info,held=record(y[held],pred,oldh));records.append(rec);print('HELD',rec,flush=True)
        if rec['held']['correct']<1477:continue
        fw,fi=fit(bank,y,a['oof_base'],c['strict_oof'],a['oof_text'],oldoof,arm);np.savez(OUT/f'{arm}_full.npz',**fw)
        if val is None:
            val,_,vy,_=load('validation');q=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
            vs=c['val_s1']+.25*(q-q.mean(1,keepdims=True));vn=np.load(ROOT/'experiments/round246/source241_validation_scores.npz')['scores']
            oldv=strict_choice(val,vs,c['strict_val'])
        vp=choose(val,vs,c['strict_val'],vn,oldv,fw,arm)
        rec.update(validation=record(vy,vp,oldv),full_training=fi);print('VALIDATION',rec,flush=True)
        template=read(ROOT/'experiments/round235/frozen/validation_predictions.json');template['selected']=vp.tolist();template.pop('prediction_seconds',None)
        dump(OUT/f'validation_{arm}_predictions.json',template)
    eligible=[r for r in records if r.get('validation',{}).get('correct',0)>763]
    selected=max(eligible,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['validation']['harms'])) if eligible else None
    dump(OUT/'result.json',dict(records=records,selected=selected,test_opened=False));print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
