"""Small train-only correction based on explicit constraints, not another ensemble."""
from data_access import ROOT,CODE,load,read_jsonl
import json,time
import numpy as np
from scipy.optimize import minimize
from round4_train import OUT,crossfit_features
from round4_semantics import fit,QuantityModel,relation_evaluation
from round4_evidence import evidence
from features import metadata
from train_models import metrics,SEED


def residual_features(semantic,arithmetic,kind):
    if kind=='arithmetic':return arithmetic
    if kind=='roles':return semantic[:,:,:10]
    if kind=='diagnostics':return np.concatenate([semantic[:,:,10:],arithmetic],2)
    if kind in ('all','unbalanced_all'):return np.concatenate([semantic,arithmetic],2)
    raise ValueError(kind)


def fit_residual(x,baseline,y,l2):
    scale=np.maximum(x.reshape(-1,x.shape[-1]).std(0),.02)
    xs=x/scale;ds=[];offsets=[];weights=[]
    for i,row in enumerate(y):
        pos=np.flatnonzero(row);neg=np.flatnonzero(row==0)
        for p in pos:
            for n in neg:
                ds.append(xs[i,p]-xs[i,n]);offsets.append(baseline[i,p]-baseline[i,n]);weights.append(1/max(1,len(pos)*len(neg)))
    d=np.asarray(ds);off=np.asarray(offsets);ww=np.asarray(weights);ww/=ww.sum()
    def objective(w):
        margin=off+d@w
        value=np.sum(ww*np.logaddexp(0,-margin))+.5*l2*(w@w)
        error=-1/(1+np.exp(np.clip(margin,-40,40)))
        grad=d.T@(ww*error)+l2*w
        return value,grad
    fitted=minimize(objective,np.zeros(x.shape[-1]),jac=True,method='L-BFGS-B',options={'maxiter':400})
    if not fitted.success:raise RuntimeError(fitted.message)
    return fitted.x/scale


def main():
    plan=dict(relation_C=[1,10,100],role_class_weight='balanced',role_selection='highest macro recall on internal TRAIN reference quantities',
        kinds=['arithmetic','roles','diagnostics','all','unbalanced_all'],l2=[1.,.1,.01],
        fallback='zero correction competes in internal selection; strict tie prefers larger regularization',
        fit='Only TRAIN labels and OOF base scores; base score coefficient fixed at one',
        caveat='Internal screening reuses existing TRAIN OOF scores; not a fully nested generalization estimate',test_access=False)
    (OUT/'residual_plan.json').write_text(json.dumps(plan,indent=2))
    b,_,y=load('train');refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');mt=metadata(b)
    split=np.load(OUT/'internal_split.npz');tr=split['fit'];hold=split['holdout'];records=[]
    for c in plan['relation_C']:
        path=OUT/'balanced_internal'/str(c);tick=time.perf_counter();info=fit([refs[i] for i in tr],path,c,True)
        ev=relation_evaluation([refs[i] for i in hold],QuantityModel(path));ev['macro_recall']=float(np.mean([ev['by_class'][str(k)]['recall'] for k in (0,1,2)]))
        rec=dict(C=c,internal=ev,fit=info,seconds=time.perf_counter()-tick);records.append(rec);print('BALANCED',rec,flush=True)
    c=max(records,key=lambda r:r['internal']['macro_recall'])['C']
    (OUT/'balanced_relation_selection.json').write_text(json.dumps(dict(records=records,best_C=c),indent=2))
    cache=OUT/'balanced_crossfit_features.npz'
    if cache.exists():sem=np.load(cache)['semantic']
    else:
        sem=crossfit_features(b,refs,y,OUT/'balanced_full_crossfit',c,True);np.savez(cache,semantic=sem)
    cache=OUT/'arithmetic_train_features.npz'
    if cache.exists():arith=np.load(cache)['features']
    else:arith=evidence(b);np.savez(cache,features=arith)
    old=np.load(ROOT/'experiments/round3/stack_scores.npz');base=old['meta_oof']
    unbalanced=np.load(OUT/'full_crossfit_features.npz')['semantic'];selections=[]
    for kind in plan['kinds']:
        x=residual_features(unbalanced if kind=='unbalanced_all' else sem,arith,kind);variants=[]
        variants.append(dict(l2=None,internal=metrics(base[hold],y[hold],mt['valid'][hold],mt['hard'][hold])))
        for reg in plan['l2']:
            w=fit_residual(x[tr],base[tr],y[tr],reg);s=base[hold]+x[hold]@w
            variants.append(dict(l2=reg,internal=metrics(s,y[hold],mt['valid'][hold],mt['hard'][hold])))
        chosen=max(variants,key=lambda r:r['internal']['correct'])
        w=np.zeros(x.shape[-1]) if chosen['l2'] is None else fit_residual(x,base,y,chosen['l2'])
        path=OUT/'residual'/kind;path.mkdir(parents=True,exist_ok=True);np.savez(path/'model.npz',weight=w)
        cfg=dict(family='constraint_residual',kind=kind,l2=chosen['l2'],role_balanced=kind!='unbalanced_all')
        (path/'config.json').write_text(json.dumps(cfg,indent=2))
        selections.append(dict(kind=kind,variants=variants,chosen=chosen));print('SELECT',kind,chosen,flush=True)
    (OUT/'residual_internal_selection.json').write_text(json.dumps(selections,indent=2))
    fit(refs,OUT/'balanced_quantity_model',c,True)
    bv,_,yv=load('validation');mv=metadata(bv)
    semv=QuantityModel(OUT/'balanced_quantity_model').features(bv);arithv=evidence(bv)
    np.savez(OUT/'balanced_validation_features.npz',semantic=semv);np.savez(OUT/'arithmetic_validation_features.npz',features=arithv)
    unbalancedv=np.load(OUT/'validation_features.npz')['semantic'];result=[]
    for sel in selections:
        kind=sel['kind'];path=OUT/'residual'/kind;w=np.load(path/'model.npz')['weight']
        x=residual_features(unbalancedv if kind=='unbalanced_all' else semv,arithv,kind)
        s=old['validation']+x@w;np.savez(path/'validation_scores.npz',scores=s)
        rec=dict(kind=kind,internal_selected_l2=sel['chosen']['l2'],validation=metrics(s,yv,mv['valid'],mv['hard']))
        result.append(rec);(path/'result.json').write_text(json.dumps(rec,indent=2));print('VALIDATION',rec,flush=True)
    (OUT/'residual_comparison.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
