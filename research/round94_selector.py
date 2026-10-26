from pathlib import Path
import json,numpy as np
from round83_features import build,summarize
from round82_relations import probabilities
from round94_context import build as target_features,join
from round80_selector import augment
from round79_selector import project,design as previous
from round68_selector import components
from round4_gates import apply_gate

def extend(x,relation):return np.concatenate([x,relation,relation-relation.mean(1,keepdims=True)],2)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    import lightgbm as lgb
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());ss,g=components(bank,arrays,p/'baseline');root,extra,flow,pairs,ptr,packets=build(bank)
    with np.load(p/'space.npz') as w:context=project(arrays,w)
    with np.load(p/'flow.npz') as w:score=np.asarray(flow@w['weight']).reshape(-1,4)
    x,bs=previous(bank,ss,cfg['weights'],g,root,extra,context,'context_influence');x=augment(x,score)
    if cfg['arm']=='relations':
        with np.load(p/'relation.npz') as w:prob=probabilities(join(pairs,target_features(bank)),w)
        x=extend(x,summarize(prob,ptr,packets))
    m=lgb.Booster(model_file=str(p/'meta.txt'));return apply_gate(bank,bs+m.predict(x.reshape(-1,x.shape[-1]),num_threads=4).reshape(-1,4),'parse_and_repair',4).astype(int).tolist()
