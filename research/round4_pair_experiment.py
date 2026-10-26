"""A separate bounded test of operation-tree constraints, with no new base models."""
from data_access import ROOT,CODE,load,read_jsonl
import json,time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from round4_train import OUT
from round4_pairs import fit,PairModel
from round4_residual import fit_residual
from features import metadata
from train_models import SEED,metrics


def main():
    (OUT/'pair_plan.json').write_text(json.dumps(dict(C=[1,10,100],select='minimum internal pair-operation NLL',
        correction_l2=[1,.1,.01],families=['pair_operations','pair_plus_arithmetic'],
        fallback='zero correction competes on TRAIN internal holdout',test_access=False),indent=2))
    b,_,y=load('train');refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');mt=metadata(b)
    split=np.load(OUT/'internal_split.npz');tr=split['fit'];hold=split['holdout'];records=[]
    for c in [1,10,100]:
        path=OUT/'pair_internal'/str(c);tick=time.perf_counter();info=fit([refs[i] for i in tr],path,c)
        rec=dict(C=c,fit=info,internal=PairModel(path).evaluate([refs[i] for i in hold]),seconds=time.perf_counter()-tick)
        records.append(rec);print('PAIR',rec,flush=True)
    bestc=min(records,key=lambda r:r['internal']['nll'])['C']
    (OUT/'pair_selection.json').write_text(json.dumps(dict(records=records,best_C=bestc),indent=2))
    x=np.empty((len(b),4,8),np.float32);cost=[]
    for k,(ids,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED+21).split(np.arange(len(y)),y.sum(1))):
        dest=OUT/'pair_crossfit'/f'fold{k}';tick=time.perf_counter();fit([refs[i] for i in ids],dest,bestc)
        x[held]=PairModel(dest).features([b[i] for i in held]);cost.append(time.perf_counter()-tick)
    np.savez(OUT/'pair_train_features.npz',features=x)
    old=np.load(ROOT/'experiments/round3/stack_scores.npz');base=old['meta_oof'];arith=np.load(OUT/'arithmetic_train_features.npz')['features']
    choices=[]
    for kind,features in [('pair_operations',x),('pair_plus_arithmetic',np.concatenate([x,arith],2))]:
        variants=[dict(l2=None,internal=metrics(base[hold],y[hold],mt['valid'][hold],mt['hard'][hold]))]
        for reg in [1,.1,.01]:
            w=fit_residual(features[tr],base[tr],y[tr],reg);s=base[hold]+features[hold]@w
            variants.append(dict(l2=reg,internal=metrics(s,y[hold],mt['valid'][hold],mt['hard'][hold])))
        best=max(variants,key=lambda r:r['internal']['correct']);w=np.zeros(features.shape[-1]) if best['l2'] is None else fit_residual(features,base,y,best['l2'])
        dest=OUT/'residual'/kind;dest.mkdir(exist_ok=True);np.savez(dest/'model.npz',weight=w)
        cfg=dict(family='constraint_residual',kind=kind,l2=best['l2'])
        (dest/'config.json').write_text(json.dumps(cfg,indent=2));choices.append(dict(kind=kind,best=best,variants=variants));print('SELECT',choices[-1],flush=True)
    (OUT/'pair_residual_selection.json').write_text(json.dumps(dict(choices=choices,crossfit_seconds=cost),indent=2))
    fit(refs,OUT/'pair_model',bestc)
    bv,_,yv=load('validation');mv=metadata(bv);pv=PairModel(OUT/'pair_model').features(bv);av=np.load(OUT/'arithmetic_validation_features.npz')['features']
    np.savez(OUT/'pair_validation_features.npz',features=pv);result=[]
    for choice in choices:
        kind=choice['kind'];xx=pv if kind=='pair_operations' else np.concatenate([pv,av],2)
        path=OUT/'residual'/kind;w=np.load(path/'model.npz')['weight'];s=old['validation']+xx@w
        rec=dict(kind=kind,internal_selected_l2=choice['best']['l2'],validation=metrics(s,yv,mv['valid'],mv['hard']))
        np.savez(path/'validation_scores.npz',scores=s);(path/'result.json').write_text(json.dumps(rec,indent=2));result.append(rec);print('VALIDATION',rec,flush=True)
    (OUT/'pair_comparison.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
