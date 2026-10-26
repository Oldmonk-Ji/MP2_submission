"""Fit tiny readouts on TRAIN cross-predictions; select on all official VAL."""
from round22_common import ROOT,OUT,INCUMBENT,BEST_INTERNAL,read,dump,sha,plan,load
from round22_selector import normalized,linear_design,side_features,group_inputs,probability_expert_logits,fused_scores,component_scores
from round3_train import pair_data
from features import metadata
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
import argparse,datetime,functools,shutil,time
import numpy as np
import round4_gates


def fit():
    assert not (OUT/'readouts/frozen.json').exists()
    dest=OUT/'readouts';dest.mkdir(exist_ok=True)
    b,_,y,ids=load('train')
    with np.load(OUT/'oof.npz') as z:
        assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['labels'])
        components=z['components']
    z=normalized(components);cues,evidence=side_features(b)
    np.savez(OUT/'train_side_features.npz',ids=ids,cues=cues,evidence=evidence)
    records=[];mixed=(y.sum(1)>0)&(y.sum(1)<4)
    for family in ('linear','conditional'):
        x=linear_design(z,cues,evidence,family);flat=x.reshape(-1,x.shape[-1]);mean=flat.mean(0);scale=flat.std(0).clip(.001)
        xx,yy,ww=pair_data(sparse.csr_matrix((flat-mean)/scale),y)
        for C in plan()['linear_C']:
            tick=time.perf_counter()
            model=LogisticRegression(C=C,fit_intercept=False,solver='liblinear',max_iter=2000,random_state=plan()['seed']).fit(xx,yy,sample_weight=ww)
            name=f'{family}_C{C}';cfg=dict(family=family,C=C,mean=mean.tolist(),scale=scale.tolist(),weights=model.coef_[0].tolist())
            dump(dest/f'{name}.json',cfg);records.append(dict(name=name,parameters=len(model.coef_[0]),seconds=time.perf_counter()-tick))
            print('READOUT',name,flush=True)
    groups,group_mask,valid=group_inputs(b);group_y=np.zeros_like(y)
    for j in range(4):
        for k in range(4):group_y[:,k]=np.maximum(group_y[:,k],y[:,j]*(groups[:,j]==k))
    for family in ('prob_candidate','prob_group_mean','prob_group_max'):
        tick=time.perf_counter();target=(y if family=='prob_candidate' else group_y)[mixed]
        zz=z[mixed];gg=groups[mixed];vv=valid[mixed]
        def objective(theta):
            weights=softmax(theta[:6]);temps=np.exp(theta[6:])
            logits=probability_expert_logits(zz,gg,vv,family,temps)
            prob=(softmax(logits,axis=1)*weights[None,None,:]).sum(2)
            hit=np.sum(prob*target,axis=1)
            return -np.log(np.maximum(hit,1e-12)).mean()+plan()['probability_penalty']*np.mean(theta[6:]**2)+.001*np.sum(weights*np.log(np.maximum(weights*6,1e-12)))
        initial=np.r_[np.log([.25,.25,.25,.1,.1,.05]),np.zeros(6)]
        lo,hi=np.log(plan()['probability_temperature_range'])
        opt=minimize(objective,initial,method='L-BFGS-B',bounds=[(-8,8)]*6+[(lo,hi)]*6,options=dict(maxiter=200,ftol=1e-9))
        cfg=dict(family=family,weights=softmax(opt.x[:6]).tolist(),temperatures=np.exp(opt.x[6:]).tolist())
        dump(dest/f'{family}.json',cfg)
        rec=dict(name=family,parameters=12,seconds=time.perf_counter()-tick,objective=float(opt.fun),optimizer_success=bool(opt.success),message=str(opt.message))
        records.append(rec);print('READOUT',rec,flush=True)
    dump(dest/'frozen.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        all_fitting_labels_official_train_only=True,n_train=4743,no_permanent_internal_holdout=True,
        official_validation_not_used_to_fit_parameters=True,records=records,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in dest.glob('*.json')}))


def evaluate(choice,base,y,hard):
    ii=np.arange(len(y));a=y[ii,choice].astype(bool);b=y[ii,base].astype(bool)
    return dict(n=len(y),correct=int(a.sum()),accuracy=100*float(a.mean()),
        hard_n=int(hard.sum()),hard_correct=int(a[hard].sum()),hard_accuracy=100*float(a[hard].mean()),
        repaired=int((a&~b).sum()),damaged=int((~a&b).sum()),net=int(a.sum()-b.sum()),changed=int((choice!=base).sum()))


def select():
    assert not (OUT/'selection.json').exists()
    f=read(OUT/'readouts/frozen.json')
    assert all(sha(ROOT/k)==v for k,v in f['hashes'].items())
    b,a,y,ids=load('validation');meta=metadata(b)
    from round4_selector import predict as old_predict
    incumbent=np.asarray(old_predict(b,a,INCUMBENT));assert int(y[np.arange(len(y)),incumbent].sum())==750
    with np.load(OUT/'full/eval_components.npz') as z:components=z['scores']
    # Independent inference path must reproduce all six exported experts.
    check,_=component_scores(b[:12],{'embeddings':a['embeddings'][:12]},OUT/'full')
    error=float(np.max(np.abs(check-components[:12])))
    assert np.allclose(check,components[:12],atol=4e-4,rtol=3e-4),error
    cues,evidence=side_features(b)
    base_weights=[m['weight'] for m in read(INCUMBENT/'base/config.json')['members']]
    internal_weights=[m['weight'] for m in read(BEST_INTERNAL/'baseline/config.json')['members']]
    common=dict(base_weights=base_weights,kernel_alpha=.1,gate=plan()['gate'],gate_margin=plan()['gate_margin'])
    configs={
        'refit_original':dict(common,kernel_alpha=0.,alpha=0.,readout=dict(family='linear')),
        'refit_kernel':dict(common,alpha=0.,readout=dict(family='linear')),
        'refit_previous_best_weights':dict(common,base_weights=internal_weights,alpha=0.,readout=dict(family='linear'))}
    for record in f['records']:
        name=record['name'];cfg=read(OUT/'readouts'/f'{name}.json')
        for alpha in plan()['residual_alpha']:configs[f'{name}_a{alpha}']=dict(common,alpha=alpha,readout=cfg)
    records=[];saved=dict(ids=ids,labels=y,incumbent_choice=incumbent)
    original_trace=round4_gates.trace
    round4_gates.trace=functools.lru_cache(maxsize=8000)(original_trace)
    try:
        for name,cfg in configs.items():
            s=fused_scores(b,components,cfg,cues,evidence)
            choice=round4_gates.apply_gate(b,s,cfg['gate'],cfg['gate_margin'])
            result=dict(name=name,alpha=cfg['alpha'],family=cfg['readout']['family'],
                parameters=len(cfg['readout'].get('weights',[])),**evaluate(choice,incumbent,y,meta['hard']))
            saved[name+'_choice']=choice;records.append(result);print('VALIDATION',name,result['correct'],'fix/harm',result['repaired'],result['damaged'],flush=True)
    finally:round4_gates.trace=original_trace
    chosen=max(records,key=lambda r:(r['correct'],-r['damaged'],-r['alpha'],-r['parameters']))
    family_winners={}
    for r in records:
        key=r['family'] if r['alpha'] else r['name']
        old=family_winners.get(key)
        if old is None or (r['correct'],-r['damaged'],-r['alpha'])>(old['correct'],-old['damaged'],-old['alpha']):family_winners[key]=r
    # Real inference assets only; no cached predictions/IDs are deployed.
    bundle=OUT/'selected_bundle';bundle.mkdir(exist_ok=True)
    model_meta=read(OUT/'full/done.json')
    for rel,digest in model_meta['assets'].items():
        src=OUT/'full'/rel;target=bundle/rel;target.parent.mkdir(parents=True,exist_ok=True)
        assert sha(src)==digest;shutil.copyfile(src,target)
    dump(bundle/'readout.json',configs[chosen['name']])
    asset_bytes=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    assert asset_bytes<=16*1024**2,asset_bytes
    np.savez(OUT/'validation_comparison.npz',**saved)
    selection=dict(selected=chosen,incumbent=evaluate(incumbent,incumbent,y,meta['hard']),
        records=records,family_winners=family_winners,training_questions=4743,validation_questions=798,
        no_evaluation_deletion=True,export_max_absolute_error=error,inference_asset_bytes=asset_bytes,
        incumbent_replaced=False,selected_candidate_retained=True,test_opened=False,
        no_internal_938_or_950_selection=True,
        limitation='Validation was used to choose among reported configurations; improvement is not an independent test result.')
    dump(OUT/'selection.json',selection)
    inference_sources=[ROOT/'research'/name for name in ('round22_selector.py','features.py','round3_features.py','round4_gates.py','round4_quantities.py','round4_evidence.py')]
    inference_sources += [ROOT/'starter/method.py',ROOT/'DASE7506-MP2-main/code/common.py']
    dump(OUT/'frozen.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selected=chosen['name'],training_n=4743,validation_n=798,validation_opened=True,test_opened=False,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in list(bundle.rglob('*'))+[OUT/'plan.json',OUT/'selection.json']+inference_sources if p.is_file()}))
    print('SELECTED',chosen,'asset_bytes',asset_bytes,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','select']);a=p.parse_args()
    fit() if a.stage=='fit' else select()
