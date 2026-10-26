"""Post-audit descriptive diagnostics only; never tunes or changes a model."""
from round18_common import OUT, load_part, read_plan, sha
from round18_evaluate import verify_frozen, pair_metrics
from round18_model import load_model, encode_bank, predict_data, objective
from features import metadata
import json, platform, time
import numpy as np
import sklearn
import torch


def main():
    verify_frozen();assert (OUT/'audit_result.json').exists()
    torch.set_num_threads(4);results={}
    for mode in ('joint','independent'):
        model,vocab,cfg=load_model(OUT/mode);part_results={}
        for part in ('fit','development'):
            b,a,y,ids=load_part(part)
            x,coverage=encode_bank(b,a,vocab,cfg);s,m=predict_data(model,x,batch=cfg['batch'])
            choices=np.where(metadata(b)['valid'],s,-1e30).argmax(1)
            part_results[part]=dict(pair_quality=pair_metrics(m,y),
                raw_correct=int(y[np.arange(len(y)),choices].sum()),n=len(y),
                eval_loss=float(objective(torch.from_numpy(s),torch.from_numpy(m),torch.from_numpy(y))))
        with np.load(OUT/'audit_predictions.npz') as z:
            s=z[mode+'_scores'];m=z[mode+'_margins'];y=z['labels']
            part_results['audit']=dict(pair_quality=pair_metrics(m,y),
                eval_loss=float(objective(torch.from_numpy(s),torch.from_numpy(m),torch.from_numpy(y))))
        results[mode]=part_results
    result=dict(posthoc=True,not_used_for_selection=True,no_model_changes=True,results=results,
                environment=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
                                 torch=torch.__version__,threads=torch.get_num_threads(),platform=platform.platform()))
    (OUT/'posthoc_diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
