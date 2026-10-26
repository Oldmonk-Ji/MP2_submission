"""Train-only out-of-fold stacking: learn complementarity, not validation weights."""
from data_access import ROOT,load
import json,shutil,time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from features import prepare,metadata
from round3_features import build,grounding
from round3_train import OUT,text_fit,pair_data
from round3_joint import fit as fit_joint
from round3_selector import leaf_scores
from train_models import fit_mlp,SEED,metrics


def normalized(s):return (s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-5)


def main():
    chosen=json.loads((OUT/'internal_selection.json').read_text())
    joint=json.loads((OUT/'joint_selection.json').read_text())['best']['name']
    duration=chosen['dense']['name'].rsplit('_',1)[1]
    names=['previous_best',f'ground_{duration}',f'answer_{duration}',joint,chosen['text']['reasoning']['name']]
    plan=dict(members=names,folds=3,split_seed=SEED+11,regularization_C=.1,
        fit='pairwise logistic regression on question-disjoint TRAIN OOF member scores, one fixed C',
        normalization='within-question mean zero and standard deviation one',
        rationale='follow-up after internal weak single-model results; no validation-fitted weights',test_access=False)
    (OUT/'stack_plan.json').write_text(json.dumps(plan,indent=2))
    b,a,y=load('train');meta=metadata(b)
    folds=list(StratifiedKFold(3,shuffle=True,random_state=SEED+11).split(np.arange(len(y)),y.sum(1)))
    p=prepare(b,a);g=grounding(b);oofs=[];ledger=[]
    for name in names:
        if name=='previous_best':
            oofs.append(np.load(ROOT/'experiments/round2/reliability/crossfit_scores.npz')['oof']);continue
        src=OUT/'full'/name;cfg=json.loads((src/'config.json').read_text());oof=np.zeros_like(y)
        x=build(b,a,cfg['features'],p,g) if cfg['family']=='mlp' else None
        for k,(fit_ids,held) in enumerate(folds):
            path=OUT/'oof'/f'{name}_fold{k}';path.mkdir(parents=True,exist_ok=True)
            if (path/'done.json').exists():
                oof[held]=np.load(path/'heldout_scores.npz')['scores'];ledger.append(json.loads((path/'done.json').read_text()));continue
            tick=time.perf_counter();print('OOF',name,k,flush=True)
            tb=[b[i] for i in fit_ids];hb=[b[i] for i in held]
            ta={'embeddings':a['embeddings'][fit_ids]};ha={'embeddings':a['embeddings'][held]}
            if cfg['family']=='mlp':
                (path/'config.json').write_text(json.dumps(cfg,indent=2))
                fit_mlp(x[fit_ids],y[fit_ids],None,None,meta['valid'][fit_ids],cfg,path,fixed_epochs=cfg['fixed_epochs'])
                from selector import model_scores
                s=model_scores(x[held],path)
            elif cfg['family']=='joint':s=fit_joint(tb,ta,y[fit_ids],hb,ha,cfg,path)
            else:s,_=text_fit(tb,y[fit_ids],hb,cfg['mode'],cfg['C'],path)
            oof[held]=s;np.savez(path/'heldout_scores.npz',scores=s)
            rec=dict(name=name,fold=k,seconds=time.perf_counter()-tick,train_questions=len(fit_ids),heldout_questions=len(held))
            (path/'done.json').write_text(json.dumps(rec,indent=2));ledger.append(rec)
        oofs.append(oof)
        del x
    z=np.stack([normalized(s) for s in oofs],2)
    # Same-question correct/incorrect differences remove irrelevant common offsets.
    from scipy import sparse
    dx,dy,weight=pair_data(sparse.csr_matrix(z.reshape(-1,len(names))),y)
    ranker=LogisticRegression(C=.1,fit_intercept=False,solver='liblinear',max_iter=1000,random_state=SEED)
    ranker.fit(dx,dy,sample_weight=weight);w=ranker.coef_[0]
    # Meta-crossfit predictions for confidence calibration later; this is not a
    # fully nested performance estimate of model/hyperparameter selection.
    meta_oof=np.zeros_like(y)
    for train,held in folds:
        xx,yy,ww=pair_data(sparse.csr_matrix(z[train].reshape(-1,len(names))),y[train])
        m=LogisticRegression(C=.1,fit_intercept=False,solver='liblinear',max_iter=1000,random_state=SEED).fit(xx,yy,sample_weight=ww)
        meta_oof[held]=z[held]@m.coef_[0]
    bundle=OUT/'stack_bundle';bundle.mkdir(exist_ok=True);members=[]
    for i,name in enumerate(names):
        src=ROOT/'experiments/round2/fixed_long' if name=='previous_best' else OUT/'full'/name
        dest=bundle/f'member{i}';dest.mkdir(exist_ok=True)
        cfg=json.loads((src/'config.json').read_text())
        if name=='previous_best':cfg['features']='base'
        (dest/'config.json').write_text(json.dumps(cfg,indent=2))
        for fn in ['model.npz']+(['vocabulary.json'] if cfg['family'] in ('joint','text') else []):shutil.copy2(src/fn,dest/fn)
        members.append(dict(name=name,directory=f'member{i}',weight=float(w[i])))
    (bundle/'config.json').write_text(json.dumps(dict(family='round3_ensemble',normalization='within_question',members=members),indent=2))
    # Read validation only after training and stack weights are frozen.
    bv,av,yv=load('validation');mv=metadata(bv)
    vs=[]
    for m in members:vs.append(leaf_scores(bv,av,bundle/m['directory']))
    zv=np.stack([normalized(s) for s in vs],2);sv=zv@w
    st=z@w;average=zv.mean(2)
    np.savez(OUT/'stack_scores.npz',oof_members=np.stack(oofs,2),validation_members=np.stack(vs,2),
             meta_oof=meta_oof,oof_fitted=st,validation=sv,weights=w)
    result=dict(plan=plan,members=members,validation=metrics(sv,yv,mv['valid'],mv['hard']),
        uniform_average_validation=metrics(average,yv,mv['valid'],mv['hard']),
        train_meta_crossfit=metrics(meta_oof,y,meta['valid'],meta['hard']),
        caveat='Meta folds share cross-fitted base scores; not fully nested model-selection evaluation.',
        oof_ledger=ledger,asset_bytes=sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file()))
    result['within_asset_limit']=result['asset_bytes']<=16*2**20
    (OUT/'stack_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
