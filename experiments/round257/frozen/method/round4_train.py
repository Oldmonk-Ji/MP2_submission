"""Internal-only screening of a relation-supervised quantity verifier."""
from data_access import ROOT,CODE,load,read_jsonl
import argparse,json,time
import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
from features import prepare,matrix,metadata
from round4_semantics import fit,QuantityModel,relation_evaluation
from train_models import fit_mlp,SEED,metrics
from selector import model_scores

OUT=ROOT/'experiments/round4'


def combine(base,sem,kind):
    if kind=='base':return base
    # Separate learned semantic relations from parser/coverage diagnostics.
    if kind=='parser':sem=sem[:,:,10:]
    if kind=='semantic':sem=sem[:,:,:10]
    return np.ascontiguousarray(np.concatenate([base,sem,sem-sem.mean(1,keepdims=True)],2),dtype=np.float32)


def configurations():
    return [dict(name=f'{kind}_{epochs}',family='mlp',features=kind,width=128,loss='mixed',
                 lr=.0003,weight_decay=.1,dropout=.15,ema=.98,seed=SEED,fixed_epochs=epochs)
            for kind in ('base','parser','semantic','all') for epochs in (35,100)]


def crossfit_features(bank,references,y,path,c,balanced=False):
    result=np.empty((len(bank),4,18),np.float32);ledger=[]
    for k,(tr,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED+21).split(np.arange(len(y)),y.sum(1))):
        dest=path/f'fold{k}';tick=time.perf_counter()
        if not (dest/'model.npz').exists():fit([references[i] for i in tr],dest,c,balanced)
        result[held]=QuantityModel(dest).features([bank[i] for i in held])
        ledger.append(dict(fold=k,train=len(tr),heldout=len(held),seconds=time.perf_counter()-tick))
    (path/'ledger.json').write_text(json.dumps(ledger,indent=2))
    return result


def internal():
    OUT.mkdir(exist_ok=True)
    plan=dict(seed=SEED,relation_C=[.1,1,10],relation_selection='minimum internal heldout quantity-role NLL',
        candidates=configurations(),selection='best internal question selection accuracy; matched-duration ablations on validation',
        hypothesis='quantity increases/decreases/unused role derived from TRAIN reference arithmetic',test_access=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    b,a,y=load('train');refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');m=metadata(b)
    tr,hold=train_test_split(np.arange(len(y)),test_size=.2,random_state=SEED,stratify=y.sum(1))
    np.savez(OUT/'internal_split.npz',fit=tr,holdout=hold)
    relation=[]
    for c in plan['relation_C']:
        path=OUT/'relations_internal'/str(c);start=time.perf_counter()
        info=fit([refs[i] for i in tr],path,c)
        rec=dict(C=c,fit=info,heldout=relation_evaluation([refs[i] for i in hold],QuantityModel(path)),seconds=time.perf_counter()-start)
        relation.append(rec);print('RELATION',rec,flush=True)
    bestc=min(relation,key=lambda r:r['heldout']['nll'])['C']
    (OUT/'relation_selection.json').write_text(json.dumps(dict(records=relation,best_C=bestc),indent=2))
    cache=OUT/'internal_features.npz'
    if cache.exists():sem=np.load(cache)['semantic']
    else:
        sem=np.empty((len(y),4,18),np.float32)
        sem[tr]=crossfit_features([b[i] for i in tr],[refs[i] for i in tr],y[tr],OUT/'relations_internal_crossfit',bestc)
        sem[hold]=QuantityModel(OUT/'relations_internal'/str(bestc)).features([b[i] for i in hold])
        np.savez(cache,semantic=sem)
    base=matrix(prepare(b,a),'context_checked');records=[]
    for cfg in configurations():
        path=OUT/'internal'/cfg['name'];path.mkdir(parents=True,exist_ok=True)
        if (path/'result.json').exists():records.append(json.loads((path/'result.json').read_text()));continue
        tick=time.perf_counter();print('INTERNAL',cfg['name'],flush=True)
        x=combine(base,sem,cfg['features']);(path/'config.json').write_text(json.dumps(cfg,indent=2))
        fit_mlp(x[tr],y[tr],None,None,m['valid'][tr],cfg,path,fixed_epochs=cfg['fixed_epochs'])
        s=model_scores(x[hold],path);rec=dict(name=cfg['name'],config=cfg,internal=metrics(s,y[hold],m['valid'][hold],m['hard'][hold]),seconds=time.perf_counter()-tick)
        np.savez(path/'holdout_scores.npz',scores=s);(path/'result.json').write_text(json.dumps(rec,indent=2));records.append(rec);print(rec,flush=True)
    best=max(records,key=lambda r:r['internal']['correct'])
    (OUT/'internal_selection.json').write_text(json.dumps(dict(records=records,best=best),indent=2))


def validation():
    selection=json.loads((OUT/'internal_selection.json').read_text());epochs=selection['best']['config']['fixed_epochs']
    c=json.loads((OUT/'relation_selection.json').read_text())['best_C']
    b,a,y=load('train');refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');m=metadata(b)
    cache=OUT/'full_crossfit_features.npz'
    if cache.exists():sem=np.load(cache)['semantic']
    else:
        sem=crossfit_features(b,refs,y,OUT/'relations_full_crossfit',c);np.savez(cache,semantic=sem)
    path=OUT/'quantity_model';info=fit(refs,path,c);qm=QuantityModel(path)
    # Official validation is first used after the feature/classifier settings
    # and training duration have been chosen on the internal training split.
    bv,av,yv=load('validation');mv=metadata(bv);sv=qm.features(bv)
    np.savez(OUT/'validation_features.npz',semantic=sv)
    base=matrix(prepare(b,a),'context_checked');basev=matrix(prepare(bv,av),'context_checked');records=[]
    for cfg in [c for c in configurations() if c['fixed_epochs']==epochs]:
        dest=OUT/'full'/cfg['name'];dest.mkdir(parents=True,exist_ok=True);tick=time.perf_counter();print('FULL',cfg['name'],flush=True)
        (dest/'config.json').write_text(json.dumps(cfg,indent=2))
        x=combine(base,sem,cfg['features'])
        fit_mlp(x,y,None,None,m['valid'],cfg,dest,fixed_epochs=epochs)
        s=model_scores(combine(basev,sv,cfg['features']),dest)
        rec=dict(name=cfg['name'],config=cfg,validation=metrics(s,yv,mv['valid'],mv['hard']),seconds=time.perf_counter()-tick)
        np.savez(dest/'validation_scores.npz',scores=s);(dest/'result.json').write_text(json.dumps(rec,indent=2));records.append(rec);print(rec,flush=True)
    (OUT/'validation_comparison.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['internal','validation']);args=p.parse_args()
    internal() if args.stage=='internal' else validation()
