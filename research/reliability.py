"""Train-only cross-fitted confidence calibration; validation diagnostics/artifacts."""
from data_access import ROOT,CODE,load
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss,roc_auc_score,log_loss
from features import prepare,matrix,metadata
from selector import model_scores
from confidence import confidence_features
from train_models import fit_mlp,fit_lightgbm,candidate_augmentation,metrics,SEED

OUT=ROOT/'experiments'/'round2'


def aggregate(matrices,normalization):
    transformed=[]
    for s in matrices:
        if normalization=='within_question':
            s=(s-s.mean(1,keepdims=True))/np.maximum(s.std(1,keepdims=True),1e-5)
        elif normalization=='probability':
            s=1/(1+np.exp(-np.clip(s,-50,50)))
        transformed.append(s)
    return np.mean(transformed,axis=0)


def calibration_stats(conf,correct):
    bins=[];ece=0.
    # Equal-count bins are more informative with this concentrated score range.
    for ids in np.array_split(np.argsort(conf),10):
        if not len(ids):continue
        mean=float(conf[ids].mean());acc=float(correct[ids].mean())
        ece+=len(ids)/len(conf)*abs(mean-acc)
        bins.append({'n':len(ids),'confidence':mean,'accuracy':acc})
    ordering=np.argsort(-conf,kind='stable');curve=[]
    for coverage in [.25,.5,.75,.9,1.]:
        n=max(1,int(np.ceil(coverage*len(correct))))
        curve.append({'coverage':n/len(correct),'n':n,'accuracy':float(correct[ordering[:n]].mean()),
                      'errors':int(n-correct[ordering[:n]].sum())})
    return {'brier':float(brier_score_loss(correct,conf)),
            'log_loss':float(log_loss(correct,np.clip(conf,1e-8,1-1e-8),labels=[0,1])),
            'ece_equal_count_10':float(ece),'auroc':float(roc_auc_score(correct,conf)),
            'bins':bins,'risk_coverage':curve}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--members',nargs='+',required=True)
    parser.add_argument('--normalization',default='within_question');args=parser.parse_args()
    analysis=OUT/'reliability';analysis.mkdir(exist_ok=True)
    bank,arrays,y=load('train');bv,av,yv=load('validation')
    p=prepare(bank,arrays);pv=prepare(bv,av);mt=metadata(bank);mv=metadata(bv)
    fold_split=list(StratifiedKFold(3,shuffle=True,random_state=SEED+11).split(np.arange(len(y)),y.sum(1)))
    oof_members=[];validation_members=[];ledger=[]
    for name in args.members:
        path=OUT/name;cfg=json.loads((path/'config.json').read_text())
        x=matrix(p,cfg['features']);xv=matrix(pv,cfg['features'])
        aug=candidate_augmentation(bank,arrays,y,cfg['features'],cfg['seed']) if cfg.get('augmentation') else None
        oof=np.zeros_like(y)
        for fold,(train_ids,held_ids) in enumerate(fold_split):
            dest=analysis/'oof_models'/f'{name}_fold{fold}';dest.mkdir(parents=True,exist_ok=True)
            tick=time.perf_counter()
            if not (dest/'done.json').exists():
                inner_train,inner_stop=train_test_split(np.arange(len(train_ids)),test_size=.2,
                    random_state=SEED,stratify=y[train_ids].sum(1))
                (dest/'config.json').write_text(json.dumps(cfg,indent=2))
                if cfg['family']=='mlp':
                    foldaug=None
                    if aug is not None:
                        ax,ay,owners=aug;keep=np.isin(owners,train_ids)
                        lookup={int(global_id):local for local,global_id in enumerate(train_ids)}
                        foldaug=ax[keep],ay[keep],np.asarray([lookup[int(i)] for i in owners[keep]])
                    details=fit_mlp(x[train_ids],y[train_ids],inner_train,inner_stop,
                                    mt['valid'][train_ids],cfg,dest,fixed_epochs=cfg.get('fixed_epochs'),augmentation_data=foldaug)
                else:
                    tune=dict(cfg,rounds=400)
                    details=fit_lightgbm(x[train_ids],y[train_ids],inner_train,inner_stop,
                                        mt['valid'][train_ids],tune,dest)
                details.pop('internal_history',None)
                details.update(seconds=time.perf_counter()-tick,held_out_questions=len(held_ids))
                (dest/'done.json').write_text(json.dumps(details,indent=2))
            oof[held_ids]=model_scores(x[held_ids],dest)
            ledger.append({'member':name,'fold':fold,**json.loads((dest/'done.json').read_text())})
            print('OOF',name,fold,'done',flush=True)
        oof_members.append(oof);validation_members.append(model_scores(xv,path))
        del x,xv,aug
    st=aggregate(oof_members,args.normalization);sv=aggregate(validation_members,args.normalization)
    xt,ct=confidence_features(st,mt);xv,cv=confidence_features(sv,mv)
    correct_t=y[np.arange(len(y)),ct];correct_v=yv[np.arange(len(yv)),cv]
    scaler=StandardScaler().fit(xt)
    calibrator=LogisticRegression(C=.1,max_iter=1000,random_state=SEED).fit(scaler.transform(xt),correct_t)
    weight=calibrator.coef_[0]/scaler.scale_;bias=calibrator.intercept_[0]-scaler.mean_@weight
    conf=calibrator.predict_proba(scaler.transform(xv))[:,1]
    uncal=1/(1+np.exp(-np.clip(xv[:,0],-50,50)))
    np.savez(analysis/'confidence_calibrator.npz',weight=weight,bias=bias)
    np.savez(analysis/'crossfit_scores.npz',oof=st,validation=sv,oof_labels=correct_t,
             validation_correct=correct_v,confidence=conf,selected=cv)
    with (ROOT/'experiments/baseline_validation/hybrid-predictions.json').open() as f:
        baseline=json.load(f)['selected']
    base_correct=yv[np.arange(len(yv)),baseline]
    rng=np.random.default_rng(SEED);delta=[];accuracy=[];hard_delta=[]
    hard=np.flatnonzero(mv['hard'])
    for _ in range(5000):
        ids=rng.integers(0,len(yv),len(yv));h=rng.choice(hard,len(hard),replace=True)
        delta.append(float((correct_v[ids]-base_correct[ids]).mean()))
        accuracy.append(float(correct_v[ids].mean()))
        hard_delta.append(float((correct_v[h]-base_correct[h]).mean()))
    by_pattern={}
    for pattern in sorted(set(mv['patterns'])):
        mask=np.asarray(mv['patterns'])==pattern
        by_pattern[pattern]={'n':int(mask.sum()),'correct':int(correct_v[mask].sum()),
                             'accuracy':float(correct_v[mask].mean())}
    result={'members':args.members,'normalization':args.normalization,'test_access':False,
            'validation':metrics(sv,yv,mv['valid'],mv['hard']),
            'train_crossfit_selection_accuracy':float(100*correct_t.mean()),
            'calibration_fit':'3-fold question-disjoint training-only OOF; no validation calibration fitting',
            'calibrated':calibration_stats(conf,correct_v),'uncalibrated_sigmoid_score':calibration_stats(uncal,correct_v),
            'conflict_calibrated':calibration_stats(conf[mv['hard']],correct_v[mv['hard']]),
            'conflict_uncalibrated':calibration_stats(uncal[mv['hard']],correct_v[mv['hard']]),
            'accuracy_95ci_percent':(100*np.quantile(accuracy,[.025,.975])).tolist(),
            'paired_gain_over_original_hybrid_95ci_pp':(100*np.quantile(delta,[.025,.975])).tolist(),
            'hard_gain_over_original_hybrid_95ci_pp':(100*np.quantile(hard_delta,[.025,.975])).tolist(),
            'new_correct_old_wrong':int(((correct_v==1)&(base_correct==0)).sum()),
            'new_wrong_old_correct':int(((correct_v==0)&(base_correct==1)).sum()),
            'by_answer_pattern':by_pattern,'oof_training_ledger':ledger,
            'selection_bias_note':'Bootstrap intervals condition on selected models and do not correct repeated validation model selection.',
            'calibration_shift_note':'OOF base learners see 2/3 of train; final base learners see all train.'}
    failures=[]
    for i in np.flatnonzero(correct_v==0):
        failures.append({'validation_index':int(i),'question':bv[i]['question'],'selected':int(cv[i]),
                         'confidence':float(conf[i]),'pattern':mv['patterns'][i],
                         'correct_candidates':np.flatnonzero(yv[i]).tolist(),
                         'candidates':bv[i]['candidates']})
    failures.sort(key=lambda r:r['confidence'],reverse=True)
    (analysis/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    (analysis/'results.json').write_text(json.dumps(result,indent=2))
    # Self-contained inference bundle; only model/config/calibration assets are included.
    bundle=OUT/'selected_bundle';bundle.mkdir(exist_ok=True)
    members=[]
    for i,name in enumerate(args.members):
        src=OUT/name;dest=bundle/f'member{i}';dest.mkdir(exist_ok=True)
        cfg=json.loads((src/'config.json').read_text())
        for filename in ('config.json','model.npz' if cfg['family']=='mlp' else 'model.txt'):
            shutil.copy2(src/filename,dest/filename)
        members.append({'directory':f'member{i}','name':name,'weight':1/len(args.members)})
    shutil.copy2(analysis/'confidence_calibrator.npz',bundle/'confidence_calibrator.npz')
    (bundle/'config.json').write_text(json.dumps({'family':'ensemble','normalization':args.normalization,
                                               'members':members,'calibration':'confidence_calibrator.npz'},indent=2))
    assets={str(f.relative_to(bundle)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
            for f in bundle.rglob('*') if f.is_file()}
    result['asset_bytes']=sum(v['bytes'] for v in assets.values())
    assert result['asset_bytes']<=16*2**20
    (analysis/'asset_manifest.json').write_text(json.dumps(assets,indent=2))
    python=ROOT/'.venv'/'Scripts'/'python.exe'
    times=[]
    for run in range(3):
        pred=analysis/f'official-validation-predictions-{run}.json';score=analysis/f'official-validation-score-{run}.json'
        tick=time.perf_counter()
        subprocess.run([str(python),str(CODE/'predict.py'),'--method',str(ROOT/'research/selector.py'),
                        '--checkpoint',str(bundle),'--split','validation','--output',str(pred)],check=True,capture_output=True)
        subprocess.run([str(python),str(CODE/'score_predictions.py'),'--split','validation',
                        '--predictions',str(pred),'--output',str(score)],check=True,capture_output=True)
        times.append(time.perf_counter()-tick)
        official=json.loads(score.read_text());assert official['correct']==int(correct_v.sum())
        assert json.loads(pred.read_text())['selected']==cv.tolist()
    result['official_validation_score']=official
    result['local_fresh_prediction_plus_score_seconds']=times
    (analysis/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('calibrated','uncalibrated_sigmoid_score','oof_training_ledger','by_answer_pattern')},indent=2),flush=True)


if __name__=='__main__':main()
