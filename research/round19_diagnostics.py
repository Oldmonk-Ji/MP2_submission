"""Fixed-checkpoint, descriptive generalization and coverage checks after audit."""
from round19_common import OUT,load_part,dump
from round19_evaluate import verify_frozen,score_record,encoder
from round19_model import load_model,tokens
from features import metadata
import json,platform
import numpy as np
import torch
import sklearn


def pair_quality(scores,y):
    diff=y[:,:,None]-y[:,None,:];mask=diff>0
    delta=scores[:,:,None]-scores[:,None,:]
    return dict(n=int(mask.sum()),correct=int(((delta>0)&mask).sum()),accuracy=float((delta[mask]>0).mean()))


def main():
    f=verify_frozen();assert (OUT/'audit_result.json').exists();torch.set_num_threads(4)
    chosen=[r for r in f['secondary'] if r['name'] in ('legacy_short','regularized_long',f['primary']['name'])]
    b,a,y,_=load_part('fit');db,da,dy,_=load_part('development');records={}
    with np.load(OUT/'audit_predictions.npz') as z:saved={k:z[k] for k in z.files}
    for rec in chosen:
        s=score_record(rec,b,a)
        with np.load(OUT/rec['path']/'development.npz') as z:ds=z['scores']
        records[rec['name']]=dict(fit=pair_quality(s,y),development=pair_quality(ds,dy),
             audit=pair_quality(saved[rec['name']+'_scores'],saved['labels']))
    # Coverage only, no further model fitting or selection.
    ab,aa,_,_=load_part('audit');coverage={}
    for rec in [r for r in f['secondary'] if r['family']=='neural']:
        model,v,cfg=load_model(OUT/rec['path']);_,c=encoder(cfg)(ab,aa,v,cfg);coverage[rec['name']]=c
    result=dict(posthoc=True,not_used_for_selection=True,model_weights_unchanged=True,
        pair_metric='Positive versus negative aggregate candidate score, counted within each question; not claim-level relation extraction quality.',
        pair_quality=records,audit_coverage=coverage,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,sklearn=sklearn.__version__,threads=4))
    dump(OUT/'posthoc_diagnostics.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
