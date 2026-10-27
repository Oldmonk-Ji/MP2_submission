"""Portable selector: NumPy MLP or LightGBM, no PyTorch import at inference."""
import json
from pathlib import Path
import numpy as np
from features import prepare,matrix,metadata


def model_scores(x,path):
    path=Path(path)
    cfg=json.loads((path/'config.json').read_text())
    if cfg['family']=='mlp':
        with np.load(path/'model.npz',allow_pickle=False) as m:
            h=(x.reshape(-1,x.shape[-1])-m['mean'])/m['scale']
            for i in range(int(m['layers'])):
                h=h@m[f'w{i}']+m[f'b{i}']
                if i<int(m['layers'])-1: h=np.maximum(h,0)
        return h.reshape(x.shape[:2])
    if cfg['family']=='lightgbm':
        import lightgbm as lgb
        model=lgb.Booster(model_file=str(path/'model.txt'))
        return model.predict(x.reshape(-1,x.shape[-1]),raw_score=True,num_threads=4).reshape(x.shape[:2])
    raise ValueError(cfg['family'])


def scores(bank,arrays,checkpoint):
    path=Path(checkpoint)
    cfg=json.loads((path/'config.json').read_text())
    p=prepare(bank,arrays)
    if cfg['family']=='ensemble':
        out=np.zeros((len(bank),4),dtype=np.float64)
        for member in cfg['members']:
            sub=path/member['directory']
            subcfg=json.loads((sub/'config.json').read_text())
            s=model_scores(matrix(p,subcfg['features']),sub)
            if cfg.get('normalization')=='within_question':
                s=(s-s.mean(1,keepdims=True))/np.maximum(s.std(1,keepdims=True),1e-5)
            elif cfg.get('normalization')=='probability':
                s=1/(1+np.exp(-np.clip(s,-50,50)))
            out+=member['weight']*s
        return out
    return model_scores(matrix(p,cfg['features']),path)


def predict(bank,arrays,checkpoint):
    if not bank: return []
    s=scores(bank,arrays,checkpoint)
    valid=metadata(bank)['valid']
    # A candidate the fixed scorer cannot parse can never be correct.
    s=np.where(valid,s,-1e30)
    return s.argmax(1).astype(int).tolist()


def predict_with_confidence(bank,arrays,checkpoint):
    """Optional analysis interface; still chooses a candidate for EVERY question."""
    from confidence import confidence_features
    if not bank:return [],[]
    s=scores(bank,arrays,checkpoint)
    x,choice=confidence_features(s,metadata(bank))
    with np.load(Path(checkpoint)/'confidence_calibrator.npz',allow_pickle=False) as c:
        logit=x@c['weight']+c['bias']
    return choice.astype(int).tolist(),(1/(1+np.exp(-np.clip(logit,-50,50)))).tolist()
