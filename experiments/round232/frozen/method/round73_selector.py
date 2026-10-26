from pathlib import Path
import json,numpy as np
from round68_selector import components,design
from round73_features import build
from round4_gates import apply_gate

def augmented(bank,ss,weights,g,root,arm):
    x,bs=design(bank,ss,weights,g)
    if arm!='control':x=np.concatenate([x,root,root-root.mean(1,keepdims=True)],2)
    return x,bs

def predict(bank,arrays,checkpoint):
    if not bank:return []
    import lightgbm as lgb
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());ss,g=components(bank,arrays,p/'baseline')
    root=build(bank) if cfg['arm']!='control' else None
    x,bs=augmented(bank,ss,cfg['weights'],g,root,cfg['arm']);m=lgb.Booster(model_file=str(p/'meta.txt'))
    s=bs+m.predict(x.reshape(-1,x.shape[-1]),num_threads=4).reshape(-1,4)
    return apply_gate(bank,s,'parse_and_repair',4).astype(int).tolist()
