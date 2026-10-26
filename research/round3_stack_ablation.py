"""Five fixed leave-one-member-out stack ablations; all weights trained on TRAIN OOF."""
from data_access import ROOT,load
import json,shutil
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from round3_train import pair_data,OUT
from round3_stack import normalized
from features import metadata
from train_models import SEED,metrics


def fit(z,y):
    xx,yy,ww=pair_data(sparse.csr_matrix(z.reshape(-1,z.shape[-1])),y)
    return LogisticRegression(C=.1,fit_intercept=False,solver='liblinear',max_iter=1000,random_state=SEED).fit(xx,yy,sample_weight=ww).coef_[0]


def main():
    original=json.loads((OUT/'stack_results.json').read_text());members=original['members']
    (OUT/'stack_ablation_plan.json').write_text(json.dumps(dict(remove=[m['name'] for m in members],
        C=.1,weights_fit='training OOF only',selection='validation correct, then fewer inference asset bytes',test_access=False),indent=2))
    b,_,y=load('train');bv,_,yv=load('validation');mt=metadata(b);mv=metadata(bv)
    data=np.load(OUT/'stack_scores.npz')
    z=np.stack([normalized(data['oof_members'][:,:,i]) for i in range(len(members))],2)
    zv=np.stack([normalized(data['validation_members'][:,:,i]) for i in range(len(members))],2)
    records=[];ws=[]
    for excluded in range(len(members)):
        keep=[i for i in range(len(members)) if i!=excluded];w=fit(z[:,:,keep],y);sv=zv[:,:,keep]@w
        bytecount=sum(f.stat().st_size for i in keep for f in (OUT/'stack_bundle'/members[i]['directory']).rglob('*') if f.is_file())
        rec=dict(name='without_'+members[excluded]['name'],excluded=excluded,keep=keep,weights=w.tolist(),
                 validation=metrics(sv,yv,mv['valid'],mv['hard']),member_asset_bytes=bytecount)
        records.append(rec);ws.append(w);print(rec,flush=True)
    best=max(records,key=lambda r:(r['validation']['correct'],-r['member_asset_bytes']))
    keep=best['keep'];w=np.asarray(best['weights']);bundle=OUT/'pruned_stack_bundle';bundle.mkdir(exist_ok=True)
    kept=[]
    for j,i in enumerate(keep):
        dest=bundle/f'member{j}';dest.mkdir(exist_ok=True)
        for src in (OUT/'stack_bundle'/members[i]['directory']).iterdir():
            if src.is_file():shutil.copy2(src,dest/src.name)
        kept.append(dict(name=members[i]['name'],directory=f'member{j}',weight=float(w[j])))
    (bundle/'config.json').write_text(json.dumps(dict(family='round3_ensemble',normalization='within_question',members=kept),indent=2))
    meta_oof=np.zeros_like(y)
    for tr,held in StratifiedKFold(3,shuffle=True,random_state=SEED+11).split(np.arange(len(y)),y.sum(1)):
        meta_oof[held]=z[held][:,:,keep]@fit(z[tr][:,:,keep],y[tr])
    np.savez(OUT/'pruned_stack_scores.npz',meta_oof=meta_oof,validation=zv[:,:,keep]@w,weights=w,
             oof_members=data['oof_members'][:,:,keep],validation_members=data['validation_members'][:,:,keep])
    result=dict(original,plan=dict(original['plan'],member_ablation=best['name']),members=kept,
                validation=best['validation'],asset_bytes=sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file()),
                train_meta_crossfit=metrics(meta_oof,y,mt['valid'],mt['hard']))
    result['within_asset_limit']=result['asset_bytes']<=16*2**20
    (OUT/'pruned_stack_results.json').write_text(json.dumps(result,indent=2))
    (OUT/'stack_ablations.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
