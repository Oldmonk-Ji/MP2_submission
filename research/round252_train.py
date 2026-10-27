"""Constrain learned complementary evidence rather than arbitrary utility rules."""
from round22_common import ROOT,load,read,dump,sha
from round252_features import build,choose
from round224_target_conclusion import strict_choice
from round225_train import record
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np

OUT=ROOT/'experiments/round252'
ARMS=['monotone_linear','monotone_text_gate']

def fit(bank,y,s,strict,text,old,arm):
    x,eligible,names=build(bank,s,strict,text,old,arm)
    delta=y-y[np.arange(len(y)),old,None];mask=eligible&(delta!=0)
    weights=np.broadcast_to((1/np.maximum(mask.sum(1),1))[:,None],mask.shape)[mask]
    xx=x[mask];target=(delta[mask]>0).astype(np.float64);scale=xx.std(0).clip(.1)
    z=(xx/scale).astype(np.float64)
    prior=np.clip(np.sum(weights*target)/weights.sum(),1e-5,1-1e-5);offset=float(np.log(prior/(1-prior)))
    fixed=arm=='monotone_text_gate'
    def loss(w):
        beta=w if fixed else w[:-1];intercept=offset if fixed else w[-1]
        logits=z@beta+intercept;residual=weights*(expit(logits)-target)
        value=np.sum(weights*(np.logaddexp(0,logits)-target*logits))+.5*np.dot(beta,beta)
        grad=z.T@residual+beta
        return value,grad if fixed else np.r_[grad,residual.sum()]
    initial=np.zeros(z.shape[1]) if fixed else np.r_[np.zeros(z.shape[1]),offset]
    bounds=[(0,None)]*z.shape[1]+([] if fixed else [(None,None)])
    model=minimize(loss,initial,jac=True,bounds=bounds,method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-10))
    assert model.success,model.message
    coef=model.x if fixed else model.x[:-1];intercept=offset if fixed else model.x[-1]
    w=dict(scale=scale.astype(np.float32),coef=coef.astype(np.float32),intercept=np.asarray(intercept,np.float32))
    return w,dict(features=names,benefit_pairs=int(target.sum()),harm_pairs=int((target==0).sum()),
                  positive_coefficients=True,coefficients=coef.tolist(),intercept=float(intercept),iterations=int(model.nit))

def main():
    threadpool_limits(4);round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hashes={n:sha(ROOT/'research'/n) for n in ('round252_features.py','round252_train.py')},arms=ARMS,
        method='Four dimensionless evidence channels, nonnegative coefficients, fixed L2 penalty1. Text-gated arm fixes the zero-text prior to TRAIN benefit/harm odds.',
        hypotheses='Prevent spurious nonmonotonic associations and reliance on scorer absolute scale. Top2 eligibility remains fixed from TRAIN evidence.',
        fitting='Same held-excluded design OOF and final3fold OOF sources from248; no new data or threshold/regularization scan.',
        selection='VAL>763 and held>=1477.',test_opened=False))
    bank,_,y,_=load('train')
    with np.load(ROOT/'experiments/round216/cache.npz') as z:c={k:z[k] for k in ('folds','strict_nested','strict_oof','strict_val','val_s1')}
    with np.load(ROOT/'experiments/round248/scores.npz') as z:a={k:z[k] for k in z.files}
    held=c['folds']==2;tr=~held;bt=[b for b,k in zip(bank,tr) if k];bh=[b for b,k in zip(bank,held) if k]
    oldt=strict_choice(bt,a['design_base'][tr],c['strict_nested'][tr]);oldh=strict_choice(bh,a['held_base'],c['strict_nested'][held])
    oldoof=strict_choice(bank,a['oof_base'],c['strict_oof']);records=[];val=oldv=None
    for arm in ARMS:
        w,info=fit(bt,y[tr],a['design_base'][tr],c['strict_nested'][tr],a['design_text'][tr],oldt,arm)
        np.savez(OUT/f'{arm}_held.npz',**w)
        hp=choose(bh,a['held_base'],c['strict_nested'][held],a['held_text'],oldh,w,arm)
        rec=dict(arm=arm,training=info,held=record(y[held],hp,oldh));records.append(rec);print('HELD',rec,flush=True)
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
