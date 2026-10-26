from pathlib import Path
import json,numpy as np
from round68_selector import components
from round73_selector import augmented
from round73_features import extract
from round74_quantity import matrix,probabilities,weighted
from round4_gates import apply_gate

def predict(bank,arrays,checkpoint):
    if not bank:return []
    import lightgbm as lgb
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());ss,g=components(bank,arrays,p/'baseline')
    extracted=[extract(r) for r in bank];root=np.asarray([r[0] for r in extracted]);usage=[r[1] for r in extracted]
    x,bs=augmented(bank,ss,cfg['weights'],g,root,'root')
    if cfg['relevance']:
        q,ptr=matrix(bank)
        with np.load(p/'quantity.npz') as w:prob=probabilities(q,ptr,w)
        f=weighted(usage,prob);x=np.concatenate([x,f,f-f.mean(1,keepdims=True)],2)
    m=lgb.Booster(model_file=str(p/'meta.txt'));s=bs+m.predict(x.reshape(-1,x.shape[-1]),num_threads=4).reshape(-1,4)
    return apply_gate(bank,s,'parse_and_repair',4).astype(int).tolist()
