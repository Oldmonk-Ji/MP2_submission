"""Validation comparisons and a small declared ensemble comparison; no test IO."""
from data_access import ROOT,load
from pathlib import Path
import json
import shutil
import numpy as np
from features import metadata
from train_models import metrics

OUT=ROOT/'experiments'/'round2'


def main():
    bank,_,y=load('validation');meta=metadata(bank)
    records=[];all_scores={}
    for path in sorted(OUT.iterdir()):
        if not (path/'result.json').exists(): continue
        result=json.loads((path/'result.json').read_text())
        with np.load(path/'validation_scores.npz') as f: s=f['scores']
        all_scores[path.name]=s
        records.append(result)
    neural=sorted([r for r in records if r['config']['family']=='mlp'],key=lambda r:r['validation']['correct'],reverse=True)
    trees=sorted([r for r in records if r['config']['family']=='lightgbm'],key=lambda r:r['validation']['correct'],reverse=True)
    member_sets=[('neural2',[r['name'] for r in neural[:2]]),
                 ('mixed2',[neural[0]['name'],trees[0]['name']]),
                 ('mixed3',[neural[0]['name'],neural[1]['name'],trees[0]['name']])]
    if 'mlp_checked_s2' in all_scores:
        member_sets += [
            ('classic3',['mp1_real_d15','mlp_no_style','lgbm_rank']),
            ('complement4',['mp1_real_d15','mlp_no_style','lgbm_rank','mlp_checked_s0']),
            ('checked_seeds3',['mlp_checked_s0','mlp_checked_s1','mlp_checked_s2'])]
    ensemble_records=[]
    # MP1 lesson: arithmetic probability mixtures differ from logit mixtures.
    for label,names in member_sets:
        for aggregation in ('logits','within_question','probability'):
            values=[]
            for name in names:
                s=all_scores[name]
                if aggregation=='within_question': s=(s-s.mean(1,keepdims=True))/np.maximum(s.std(1,keepdims=True),1e-5)
                elif aggregation=='probability': s=1/(1+np.exp(-np.clip(s,-50,50)))
                values.append(s)
            s=np.mean(values,axis=0)
            ensemble_records.append({'name':f'ensemble_{label}_{aggregation}','members':names,
                                     'aggregation':aggregation,'validation':metrics(s,y,meta['valid'],meta['hard'])})
    if 'mlp_checked_s0' in all_scores:
        names=['mp1_real_d15','mlp_no_style','lgbm_rank','mlp_checked_s0']
        normalized=[(all_scores[n]-all_scores[n].mean(1,keepdims=True))/np.maximum(all_scores[n].std(1,keepdims=True),1e-5) for n in names]
        for weight in (.25,.5,.75):
            weights=[(1-weight)/3]*3+[weight]
            s=sum(w*score for w,score in zip(weights,normalized))
            ensemble_records.append({'name':f'ensemble_complement_weight{weight}','members':names,
                                     'weights':weights,'aggregation':'within_question',
                                     'validation':metrics(s,y,meta['valid'],meta['hard'])})
    choices=[np.where(meta['valid'],s,-1e30).argmax(1) for s in all_scores.values()]
    correctness=np.asarray([y[np.arange(len(y)),c] for c in choices])
    diagnostic={'individual_models':len(records),'ensemble_candidates':len(ensemble_records),
                'any_model_correct_oracle_accuracy':float(100*correctness.max(0).mean()),
                'all_models_wrong_questions':int((correctness.max(0)==0).sum()),
                'oracle_is_not_deployable':True,'test_access':False}
    report={'individual':sorted(records,key=lambda r:r['validation']['correct'],reverse=True),
            'ensemble':sorted(ensemble_records,key=lambda r:r['validation']['correct'],reverse=True),
            'diagnostic':diagnostic}
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2))
    for r in report['individual'][:5]+report['ensemble'][:5]:
        print(r['name'],r['validation'])
    print(json.dumps(diagnostic))


if __name__=='__main__':main()
