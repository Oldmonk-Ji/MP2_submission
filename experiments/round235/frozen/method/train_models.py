"""Question-disjoint internal early stopping; train refit; validation-only selection."""
import data_access
from data_access import ROOT,load
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from features import prepare,matrix,metadata
from selector import model_scores

OUT=ROOT/'experiments'/'round2'
SEED=20261014


def metrics(s,y,valid,hard):
    choices=np.where(valid,s,-1e30).argmax(1)
    correct=y[np.arange(len(y)),choices]
    return {'correct':int(correct.sum()),'n':len(y),'accuracy':float(100*correct.mean()),
            'hard_accuracy':float(100*correct[hard].mean()),'hard_n':int(hard.sum())}


def configurations():
    result=[]
    for name,features,loss,width in [
        ('mlp_embedding','embedding','mixed',64),
        ('mlp_hybrid','hybrid','mixed',64),
        ('mlp_context','context','mixed',64),
        ('mlp_rich','context_rich','mixed',64),
        ('mlp_no_vote','context_no_vote','mixed',64),
        ('mlp_no_style','context_no_style','mixed',64),
        ('mlp_no_arithmetic','context_no_arithmetic','mixed',64),
        ('mlp_features_only','features_only','mixed',64),
        ('mlp_bce','context_rich','bce',64),
        ('mlp_wide','context_rich','mixed',128),
    ]:
        result.append({'name':name,'family':'mlp','features':features,'loss':loss,
                       'width':width,'lr':.001,'weight_decay':.01,'dropout':.15,
                       'max_epochs':45,'patience':8,'seed':SEED})
    for name,features,objective,leaves in [
        ('lgbm_binary','rich','binary',15),
        ('lgbm_rank','rich','lambdarank',15),
        ('lgbm_rank31','rich','lambdarank',31),
        ('lgbm_features_only','features_only','binary',15),
    ]:
        result.append({'name':name,'family':'lightgbm','features':features,'objective':objective,
                       'leaves':leaves,'rounds':400,'seed':SEED})
    for augmented in (False,True):
        for dropout in (0.,.15):
            result.append({'name':f'mp1_{"aug" if augmented else "real"}_d{int(100*dropout)}',
                           'family':'mlp','features':'context_rich','loss':'mixed','width':128,
                           'lr':.0003,'weight_decay':.1,'dropout':dropout,'max_epochs':180,
                           'patience':30,'seed':SEED,'augmentation':augmented,'ema':.98})
    result.append({'name':'mlp_teacher','family':'mlp','features':'context_rich','loss':'mixed',
                   'width':256,'lr':.0003,'weight_decay':.3,'dropout':.3,'max_epochs':120,
                   'patience':25,'seed':SEED,'ema':.98})
    for seed_offset in (0,1,2):
        result.append({'name':f'mlp_checked_s{seed_offset}','family':'mlp','features':'context_checked',
                       'loss':'mixed','width':128,'lr':.0003,'weight_decay':.1,'dropout':.15,
                       'max_epochs':180,'patience':30,'seed':SEED+seed_offset,'ema':.98})
    result.append({'name':'lgbm_checked','family':'lightgbm','features':'rich_checked',
                   'objective':'lambdarank','leaves':15,'rounds':400,'seed':SEED})
    result.append({'name':'lgbm_context_checked','family':'lightgbm','features':'context_checked',
                   'objective':'lambdarank','leaves':15,'rounds':600,'seed':SEED})
    result.append({'name':'lgbm_long','family':'lightgbm','features':'rich_checked',
                   'objective':'lambdarank','leaves':31,'rounds':1600,'learning_rate':.015,
                   'patience':150,'seed':SEED})
    for name,feature,loss in [
        ('fixed_no_group','rich_checked','mixed'),
        ('fixed_no_vote','checked_no_vote','mixed'),
        ('fixed_no_style','checked_no_style','mixed'),
        ('fixed_no_arithmetic','context_no_arithmetic','mixed'),
        ('fixed_no_embedding','features_checked','mixed'),
        ('fixed_bce','context_checked','bce'),
        ('fixed_short','context_checked','mixed'),
        ('fixed_long','context_checked','mixed'),
    ]:
        result.append({'name':name,'family':'mlp','features':feature,'loss':loss,'width':128,
                       'lr':.0003,'weight_decay':.1,'dropout':.15,'max_epochs':180,'patience':30,
                       'seed':SEED,'ema':.98,'fixed_epochs':10 if name=='fixed_short' else (100 if name=='fixed_long' else 35)})
    return result


def fit_mlp(x,y,fit_idx,stop_idx,valid,cfg,path,fixed_epochs=None,augmentation_data=None):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    def fit(indices,epochs,monitor=False):
        torch.manual_seed(cfg['seed'])
        rng=np.random.default_rng(cfg['seed'])
        current_x=x;current_y=y;training_indices=indices
        if cfg.get('augmentation'):
            if augmentation_data is None: raise ValueError('missing training-only augmentation')
            ax,ay,owners=augmentation_data
            keep=np.isin(owners,indices)
            current_x=np.concatenate([x,ax[keep]],axis=0)
            current_y=np.concatenate([y,ay[keep]],axis=0)
            training_indices=np.concatenate([indices,np.arange(len(x),len(current_x))])
        scaler=StandardScaler().fit(current_x[training_indices].reshape(-1,x.shape[-1]))
        xt=torch.from_numpy(((current_x-scaler.mean_)/scaler.scale_).astype(np.float32))
        yt=torch.from_numpy(current_y)
        model=nn.Sequential(nn.Linear(x.shape[-1],cfg['width']),nn.ReLU(),
                            nn.Dropout(cfg['dropout']),nn.Linear(cfg['width'],32),nn.ReLU(),nn.Linear(32,1))
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        ema=copy.deepcopy(model) if cfg.get('ema') else None
        best=-1;best_epoch=1;best_state=None;history=[]
        for epoch in range(1,epochs+1):
            model.train(); total=0
            for ids in np.array_split(rng.permutation(training_indices),max(1,int(np.ceil(len(training_indices)/128)))):
                s=model(xt[ids]).squeeze(-1)
                target=yt[ids]
                bce=nn.functional.binary_cross_entropy_with_logits(s,target)
                group=(torch.logsumexp(s,dim=1)-torch.logsumexp(s.masked_fill(target<.5,-1e9),dim=1)).mean()
                loss=bce if cfg['loss']=='bce' else group+.25*bce
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                if ema is not None:
                    with torch.no_grad():
                        for ep,mp in zip(ema.parameters(),model.parameters()):
                            ep.lerp_(mp,1-cfg['ema'])
                total+=float(loss.detach())*len(ids)
            if monitor:
                evaluation_model=ema if ema is not None else model
                evaluation_model.eval()
                with torch.no_grad(): s=evaluation_model(xt[stop_idx]).squeeze(-1).numpy()
                selected=np.where(valid[stop_idx],s,-1e30).argmax(1)
                correct=int(y[stop_idx][np.arange(len(stop_idx)),selected].sum())
                history.append({'epoch':epoch,'loss':total/len(training_indices),'holdout_correct':correct})
                if correct>best:
                    best=correct;best_epoch=epoch;best_state=copy.deepcopy(evaluation_model.state_dict())
                if epoch-best_epoch>=cfg['patience']:
                    break
        if monitor:
            model.load_state_dict(best_state)
        else:
            best_epoch=epochs
            if ema is not None: model.load_state_dict(ema.state_dict())
        model.eval()
        weights={'mean':scaler.mean_.astype(np.float32),'scale':scaler.scale_.astype(np.float32),'layers':np.asarray(3)}
        for i,layer in enumerate(m for m in model if isinstance(m,nn.Linear)):
            weights[f'w{i}']=layer.weight.detach().numpy().T.copy()
            weights[f'b{i}']=layer.bias.detach().numpy().copy()
        return weights,best_epoch,history
    if fixed_epochs is None:
        _,epochs,history=fit(fit_idx,cfg['max_epochs'],True)
    else: epochs=fixed_epochs;history=[]
    weights,_,_=fit(np.arange(len(x)),epochs,False)
    np.savez(path/'model.npz',**weights)
    return {'epochs':epochs,'internal_history':history}


def candidate_augmentation(bank,arrays,y,features,seed):
    """Resample candidates only within a TRAIN question; retain paired embeddings.

    Each mixed-label question produces 1-positive/3-negative and 2/2 groups.
    Repeated candidates simulate repeated sampling. No synthetic text or labels.
    Owners let each internal fold exclude its held-out questions completely.
    """
    rng=np.random.default_rng(seed);rows=[];emb=[];targets=[];owners=[]
    for i,row in enumerate(bank):
        pos=np.flatnonzero(y[i]>.5);neg=np.flatnonzero(y[i]<.5)
        if len(pos)==0 or len(neg)==0: continue
        for k in (1,2):
            chosen=np.concatenate([rng.choice(pos,k,replace=True),rng.choice(neg,4-k,replace=True)])
            rng.shuffle(chosen)
            rows.append({'question':row['question'],'candidates':[
                dict(row['candidates'][int(j)],index=t) for t,j in enumerate(chosen)]})
            emb.append(arrays['embeddings'][i,chosen]);targets.append(y[i,chosen]);owners.append(i)
    return matrix(prepare(rows,{'embeddings':np.asarray(emb)}),features),np.asarray(targets),np.asarray(owners)


def fit_lightgbm(x,y,fit_idx,stop_idx,valid,cfg,path,fixed_rounds=None):
    import lightgbm as lgb
    params={'objective':cfg['objective'],'metric':'None','verbosity':-1,'num_threads':4,
            'seed':cfg['seed'],'deterministic':True,'force_col_wise':True,
            'num_leaves':cfg['leaves'],'min_data_in_leaf':40,'learning_rate':cfg.get('learning_rate',.04),
            'lambda_l2':5.,'feature_fraction':.8}
    if cfg['objective']=='lambdarank':
        params.update(label_gain=[0,1],lambdarank_truncation_level=1)
    def dataset(ids):
        return lgb.Dataset(x[ids].reshape(-1,x.shape[-1]),label=y[ids].reshape(-1),group=[4]*len(ids),free_raw_data=True)
    def evaluate(pred,ds):
        p=np.asarray(pred).reshape(-1,4)
        choices=np.where(valid[stop_idx],p,-1e30).argmax(1)
        return 'selection_accuracy',float(y[stop_idx][np.arange(len(stop_idx)),choices].mean()),True
    if fixed_rounds is None:
        history={}
        m=lgb.train(params,dataset(fit_idx),num_boost_round=cfg['rounds'],valid_sets=[dataset(stop_idx)],
                    feval=evaluate,callbacks=[lgb.early_stopping(cfg.get('patience',50),verbose=False),lgb.record_evaluation(history)])
        rounds=m.best_iteration
    else: rounds=fixed_rounds;history={}
    model=lgb.train(params,dataset(np.arange(len(x))),num_boost_round=rounds)
    model.save_model(str(path/'model.txt'))
    return {'rounds':rounds,'internal_history':history}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='*');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    configs=configurations()
    if args.only: configs=[c for c in configs if c['name'] in args.only]
    (OUT/'planned_experiments.json').write_text(json.dumps(configurations(),indent=2))
    b,a,y=load('train');bv,av,yv=load('validation')
    p=prepare(b,a);pv=prepare(bv,av);meta=metadata(b);mv=metadata(bv)
    fit_idx,stop_idx=train_test_split(np.arange(len(b)),test_size=.2,random_state=SEED,stratify=y.sum(1))
    for cfg in configs:
        path=OUT/cfg['name'];path.mkdir(exist_ok=True)
        if (path/'result.json').exists():
            print('SKIP completed',cfg['name'],flush=True);continue
        print('START',cfg['name'],flush=True)
        start=time.perf_counter();x=matrix(p,cfg['features']);xv=matrix(pv,cfg['features'])
        cfg['internal_split_seed']=SEED;cfg['training_questions']=len(b);cfg['test_access']=False
        (path/'config.json').write_text(json.dumps(cfg,indent=2))
        if cfg['family']=='mlp':
            aug=candidate_augmentation(b,a,y,cfg['features'],cfg['seed']) if cfg.get('augmentation') else None
            details=fit_mlp(x,y,fit_idx,stop_idx,meta['valid'],cfg,path,
                            fixed_epochs=cfg.get('fixed_epochs'),augmentation_data=aug)
            del aug
        else:
            details=fit_lightgbm(x,y,fit_idx,stop_idx,meta['valid'],cfg,path)
        cfg.update({k:v for k,v in details.items() if k!='internal_history'})
        (path/'config.json').write_text(json.dumps(cfg,indent=2))
        s=model_scores(xv,path)
        result={'name':cfg['name'],'config':cfg,'validation':metrics(s,yv,mv['valid'],mv['hard']),
                'training_seconds':time.perf_counter()-start,'fit_uses_train_only':True,
                'internal_early_stopping_questions':0 if cfg.get('fixed_epochs') else len(stop_idx),'test_access':False}
        result['inference_asset_bytes']=sum((path/f).stat().st_size for f in ['config.json','model.npz' if cfg['family']=='mlp' else 'model.txt'])
        np.savez(path/'validation_scores.npz',scores=s)
        (path/'history.json').write_text(json.dumps(details['internal_history'],indent=2))
        (path/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        del x,xv


if __name__=='__main__':main()
