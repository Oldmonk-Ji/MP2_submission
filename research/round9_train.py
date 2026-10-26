"""Nonlinear answer-group correction, fixed train-only screening before validation."""
from data_access import ROOT,load
import json,time,hashlib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split,StratifiedKFold
from round9_group_features import build,group_labels,choices,subset,candidate_scores
from round4_gates import apply_gate

OUT=ROOT/'experiments/round9'

def train_model(x,b,y,groups,path,leaves,rounds):
    params=dict(objective='lambdarank',metric='None',verbosity=-1,num_threads=4,learning_rate=.025,num_leaves=leaves,min_data_in_leaf=80,lambda_l2=20,lambda_l1=1,feature_fraction=.85,bagging_fraction=1.,seed=20261014,deterministic=True,force_col_wise=True,max_depth=4,lambdarank_truncation_level=4)
    dataset=lgb.Dataset(x,label=y,group=groups,init_score=b,free_raw_data=True)
    model=lgb.train(params,dataset,num_boost_round=rounds);path.parent.mkdir(parents=True,exist_ok=True);model.save_model(str(path));return model

def metric(choice,y,hard,old=None):
    correct=y[np.arange(len(y)),choice].astype(bool)
    result=dict(correct=int(correct.sum()),n=len(y),accuracy=100*float(correct.mean()),hard_correct=int(correct[hard].sum()),hard_n=int(hard.sum()))
    if old is not None:
        previous=y[np.arange(len(y)),old].astype(bool);result.update(fixed=int((correct&~previous).sum()),lost=int((~correct&previous).sum()))
    return result

def main():
    OUT.mkdir(exist_ok=True)
    configs=[dict(kind=kind,leaves=l,rounds=r,alpha=a) for kind in ('groups','candidate') for l,r in [(7,120),(7,300),(15,120)] for a in (.25,.5,1.)]
    plan=dict(configurations=configs,screening='Fixed train-only 3804/951 split of existing train OOF base scores; choose one config per feature kind. Base OOF models are not nested within this split. No fresh independent-generalization claim.',seed=20261014,quality_gate='Group correction must improve internal holdout by at least 2 questions; otherwise no validation.',hypothesis='Combine evidence by numeric answer and learn nonlinear interactions between independent scoring signals; do not invoke failed semantic extractor.',validation_policy='One frozen comparison: two internally selected variants plus existing baseline and fixed group-pooling ablation. No tuning on validation.',test_access=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    b,_,y=load('train');cache=np.load(ROOT/'experiments/round3/stack_scores.npz');members=cache['oof_members'];base=cache['meta_oof']
    arith=np.load(ROOT/'experiments/round4/arithmetic_train_features.npz')['features']
    gate=json.loads((ROOT/'experiments/round4/selected_bundle/config.json').read_text());baseline=apply_gate(b,base,gate['gate_kind'],gate['margin_limit'])
    hard=np.array([len(set(a for a in row if a is not None))>=2 for row in __import__('features').metadata(b)['answers']])
    fit,hold=train_test_split(np.arange(len(y)),test_size=.2,random_state=20261014,stratify=y.sum(1))
    np.savez(OUT/'internal_split.npz',fit=fit,holdout=hold)
    records=[];selected={};tick=time.perf_counter()
    for kind in ('groups','candidate'):
        x,gb,mapping,sizes,meta=build(b,members,base,arith,kind);gy=group_labels(y,mapping)
        for leaves,rounds in [(7,120),(7,300),(15,120)]:
            xx,bb,yy,gg=subset(x,gb,gy,mapping,sizes,fit)
            path=OUT/'internal'/f'{kind}_L{leaves}_R{rounds}.txt';m=train_model(xx,bb,yy,gg,path,leaves,rounds);delta=m.predict(x,num_threads=4)
            for alpha in (.25,.5,1.):
                choice=choices(gb+alpha*delta,mapping,len(b))
                # Identical frozen arithmetic gate remains a final, independent check.
                s=candidate_scores(gb+alpha*delta,mapping,base,b)
                choice=apply_gate(b,s,gate['gate_kind'],gate['margin_limit'])
                rec=dict(kind=kind,leaves=leaves,rounds=rounds,alpha=alpha,internal=metric(choice[hold],y[hold],hard[hold],baseline[hold]));records.append(rec);print(json.dumps(rec),flush=True)
        best=max([r for r in records if r['kind']==kind],key=lambda r:r['internal']['correct']);selected[kind]=best
        xx,bb,yy,gg=subset(x,gb,gy,mapping,sizes,np.arange(len(y)));train_model(xx,bb,yy,gg,OUT/f'{kind}.txt',best['leaves'],best['rounds'])
    base_metric=metric(baseline[hold],y[hold],hard[hold]);passed=selected['groups']['internal']['correct']>=base_metric['correct']+2
    selection=dict(selected=selected,baseline_internal=base_metric,internal_gate_passed=passed,records=records,seconds=time.perf_counter()-tick,training_questions=len(y))
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    paths=[ROOT/'research'/f for f in ['round9_train.py','round9_group_features.py','round9_selector.py']]+[OUT/'groups.txt',OUT/'candidate.txt',OUT/'selection.json',OUT/'plan.json']
    (OUT/'frozen.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    print('BASE',base_metric,'GATE',passed,flush=True)

if __name__=='__main__':main()
