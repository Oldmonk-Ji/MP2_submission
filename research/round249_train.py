"""Cross-fitted expected-utility arbiter for the new raw-text expert."""
from round22_common import ROOT,load,read,dump,sha
from round224_target_conclusion import strict_choice
from round225_train import record
from round249_features import build,choose
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np

OUT=ROOT/'experiments/round249'
MODES=['supported','joint','top2']
OBJECTIVES=['benefit_vs_harm','expected_gain']

def fit(bank,y,s,strict,text,old,mode,objective):
    x,eligible,names=build(bank,s,strict,text,old,mode)
    delta=y-y[np.arange(len(y)),old,None]
    mask=eligible&(delta!=0) if objective=='benefit_vs_harm' else eligible
    weights=1/np.maximum(mask.sum(1),1)
    xx=x[mask];target=((delta[mask]>0) if objective=='benefit_vs_harm' else delta[mask]).astype(np.int32)
    mean=xx.mean(0);scale=xx.std(0).clip(.1)
    z=np.clip((xx-mean)/scale,-6,6)
    model=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,fit_intercept=True)
    model.fit(z,target,sample_weight=np.broadcast_to(weights[:,None],mask.shape)[mask])
    w=dict(mean=mean.astype(np.float32),scale=scale.astype(np.float32))
    if objective=='benefit_vs_harm':
        w.update(coef=model.coef_[0].astype(np.float32),intercept=model.intercept_.astype(np.float32)[0])
    else:
        coef=model.coef_.astype(np.float32);intercept=model.intercept_.astype(np.float32)
        if len(model.classes_)==2:
            coef=np.concatenate([np.zeros_like(coef),coef]);intercept=np.r_[np.float32(0),intercept]
        w.update(coef=coef,intercept=intercept,classes=model.classes_.astype(np.float32))
    info=dict(benefit_pairs=int((delta[mask]>0).sum()),harm_pairs=int((delta[mask]<0).sum()),
              neutral_pairs=int((delta[mask]==0).sum()),feature_names=names,
              iterations=int(model.n_iter_[0]),objective=objective,threshold='positive expected gain / probability0.5',C=1)
    return w,info

def main():
    threadpool_limits(4);round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=False)
    dump(OUT/'plan.json',dict(source_hashes={n:sha(ROOT/'research'/n) for n in ('round249_train.py','round249_features.py')},
        method='Expected accuracy-gain classifier for replacing the final frozen choice with a text-supported candidate.',
        modes=MODES,objectives=OBJECTIVES,model='Fixed L2 logistic C1; binary conditional benefit/harm or multinomial expected utility; no threshold/C/seed search.',
        target='Binary classifier conditions on benefit/harm; multinomial retains negative/neutral/positive utility outcomes. Normalize eligible pair weights per question.',
        data='Fit arbiter on folds0/1 using held-excluded OOF text/old model predictions; evaluate fold2. Final arbiter fits all TRAIN with3-fold OOF predictions.',
        selection='VAL>763 and held>=1477, then validation,held,fewer VAL harms.',test_opened=False))
    bank,_,y,ids=load('train')
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        c={k:z[k] for k in ('folds','strict_nested','strict_oof','strict_val','val_s1')}
    with np.load(ROOT/'experiments/round248/scores.npz') as z:
        cache={k:z[k] for k in z.files}
    assert np.array_equal(cache['ids'],ids)
    held=c['folds']==2;fitrows=~held
    bf=[b for b,k in zip(bank,fitrows) if k];bh=[b for b,k in zip(bank,held) if k]
    fs,fn=cache['design_base'][fitrows],cache['design_text'][fitrows]
    hs,hn=cache['held_base'],cache['held_text']
    oldf=strict_choice(bf,fs,c['strict_nested'][fitrows])
    oldh=strict_choice(bh,hs,c['strict_nested'][held])
    assert int(y[held][np.arange(held.sum()),oldh].sum())==1479
    oldoof=strict_choice(bank,cache['oof_base'],c['strict_oof'])
    records=[];val=oldv=None
    for mode,objective in [(m,o) for m in MODES for o in OBJECTIVES]:
        arm=mode+'_'+objective
        w,info=fit(bf,y[fitrows],fs,c['strict_nested'][fitrows],fn,oldf,mode,objective)
        np.savez(OUT/f'{arm}_held.npz',**w)
        pred=choose(bh,hs,c['strict_nested'][held],hn,oldh,w,mode)
        rec=dict(arm=arm,mode=mode,objective=objective,training=info,held=record(y[held],pred,oldh),validation_evaluated=False)
        records.append(rec);print('HELD',rec,flush=True)
        if rec['held']['correct']<1477:continue
        fw,full_info=fit(bank,y,cache['oof_base'],c['strict_oof'],cache['oof_text'],oldoof,mode,objective)
        np.savez(OUT/f'{arm}_full.npz',**fw)
        if val is None:
            val,_,vy,_=load('validation')
            q=np.load(ROOT/'experiments/round234/context_ground_validation.npz')['scores']
            vs=c['val_s1']+.25*(q-q.mean(1,keepdims=True))
            vn=np.load(ROOT/'experiments/round246/source241_validation_scores.npz')['scores']
            oldv=strict_choice(val,vs,c['strict_val'])
            assert int(vy[np.arange(len(vy)),oldv].sum())==763
        vp=choose(val,vs,c['strict_val'],vn,oldv,fw,mode)
        rec.update(validation=record(vy,vp,oldv),validation_evaluated=True,full_training=full_info)
        template=read(ROOT/'experiments/round235/frozen/validation_predictions.json')
        template['selected']=vp.tolist();template.pop('prediction_seconds',None)
        dump(OUT/f'validation_{arm}_predictions.json',template)
        print('VALIDATION',rec,flush=True)
        dump(OUT/'progress.json',dict(records=records,test_opened=False))
    eligible=[r for r in records if r.get('validation',{}).get('correct',0)>763]
    selected=max(eligible,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['validation']['harms'])) if eligible else None
    dump(OUT/'result.json',dict(records=records,selected=selected,test_opened=False));print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
