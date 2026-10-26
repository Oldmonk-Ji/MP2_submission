"""Refit the actual five-member baseline without audit leakage."""
from round18_common import ROOT,OUT,load_part,normalized,paired,sha
import json,time
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from train_models import fit_mlp
from round3_train import text_fit,pair_data
from round3_joint import fit as joint_fit
from round3_features import build
from round3_selector import leaf_scores
from round4_gates import apply_gate
from features import metadata

def main():
    root=OUT/'baseline';root.mkdir(exist_ok=True);assert not (root/'done.json').exists()
    b,a,y,ids=load_part('fit');d,da,dy,dev=load_part('development');dm=metadata(d);mm=metadata(b)
    original=ROOT/'experiments/round4/selected_bundle/base'
    cfg=json.loads((original/'config.json').read_text());ss=[];members=[];ledger=[]
    for i,member in enumerate(cfg['members']):
        source=original/member['directory'];c=json.loads((source/'config.json').read_text());dest=root/member['directory'];dest.mkdir(exist_ok=True)
        c.update(training_questions=len(b),training_partition='round18 fit only',round18_plan_sha=sha(OUT/'plan.json'))
        tick=time.perf_counter();print('FIT BASELINE',i,member['name'],flush=True)
        if (dest/'development_scores.npz').exists():
            with np.load(dest/'development_scores.npz') as z:s=z['scores']
        elif c['family']=='mlp':
            (dest/'config.json').write_text(json.dumps(c,indent=2));x=build(b,a,c['features'])
            fit_mlp(x,y,None,None,mm['valid'],c,dest,fixed_epochs=c['fixed_epochs']);del x
            s=leaf_scores(d,da,dest)
        elif c['family']=='joint':s=joint_fit(b,a,y,d,da,c,dest)
        else:s,_=text_fit(b,y,d,c['mode'],c['C'],dest)
        np.savez(dest/'development_scores.npz',scores=s);ss.append(normalized(s));members.append(dict(member,weight=0.))
        ledger.append(dict(member=member['name'],seconds=time.perf_counter()-tick));print(ledger[-1],flush=True)
    z=np.stack(ss,2);dx,ty,ww=pair_data(sparse.csr_matrix(z.reshape(-1,5)),dy)
    ranker=LogisticRegression(C=.1,fit_intercept=False,solver='liblinear',max_iter=1000,random_state=18018).fit(dx,ty,sample_weight=ww)
    for member,w in zip(members,ranker.coef_[0]):member['weight']=float(w)
    config=dict(family='round3_ensemble',normalization='within_question',members=members)
    (root/'config.json').write_text(json.dumps(config,indent=2));s=z@ranker.coef_[0]
    choice=apply_gate(d,s,'parse_and_repair',4);np.savez(root/'development.npz',scores=s,choices=choice)
    assets=[p for p in root.rglob('*') if p.name in ('config.json','model.npz','vocabulary.json')]
    result=dict(fit_questions=len(b),development_questions=len(d),fit_ids_sha=sha(OUT/'plan.json'),
                development=paired(choice,choice,dy,dm['hard']),ledger=ledger,
                inference_asset_bytes=sum(p.stat().st_size for p in assets),assets={str(p.relative_to(root)):sha(p) for p in assets},
                no_original_weights_loaded=True,no_audit_selection=True)
    (root/'done.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['development']),flush=True)

if __name__=='__main__':main()
