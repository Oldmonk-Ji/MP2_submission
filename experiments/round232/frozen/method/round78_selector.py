from pathlib import Path
import json,numpy as np
from round68_selector import components
from round73_selector import augmented
from round78_features import build
from round4_gates import apply_gate

def design(bank,ss,weights,g,root,extra,arm):
    x,bs=augmented(bank,ss,weights,g,root,'root')
    if arm=='influence':x=np.concatenate([x,extra,extra-extra.mean(1,keepdims=True)],2)
    return x,bs

def predict(bank,arrays,checkpoint):
    if not bank:return []
    import lightgbm as lgb
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());ss,g=components(bank,arrays,p/'baseline');root,extra=build(bank)
    x,bs=design(bank,ss,cfg['weights'],g,root,extra,cfg['arm']);m=lgb.Booster(model_file=str(p/'meta.txt'))
    return apply_gate(bank,bs+m.predict(x.reshape(-1,x.shape[-1]),num_threads=4).reshape(-1,4),'parse_and_repair',4).astype(int).tolist()
