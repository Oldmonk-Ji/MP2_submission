from pathlib import Path
import json,numpy as np
from round80_features import build
from round79_selector import project,design as previous
from round68_selector import components
from round22_selector import normalized
from round4_gates import apply_gate

def augment(x,score):return np.concatenate([x,score[:,:,None],(score-score.mean(1,keepdims=True))[:,:,None],normalized(score)[:,:,None]],2)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    import lightgbm as lgb
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());ss,g=components(bank,arrays,p/'baseline');root,extra,flow=build(bank,cfg['arm'])
    with np.load(p/'space.npz') as w:context=project(arrays,w)
    with np.load(p/'flow.npz') as w:score=np.asarray(flow@w['weight']).reshape(-1,4)
    x,bs=previous(bank,ss,cfg['weights'],g,root,extra,context,'context_influence');x=augment(x,score);m=lgb.Booster(model_file=str(p/'meta.txt'))
    return apply_gate(bank,bs+m.predict(x.reshape(-1,x.shape[-1]),num_threads=4).reshape(-1,4),'parse_and_repair',4).astype(int).tolist()
