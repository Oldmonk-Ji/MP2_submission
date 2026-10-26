from pathlib import Path
import json,numpy as np
from round136_features import build,summarize
from round136_inference import probabilities,question_context
from round80_selector import augment
from round79_selector import project,design as previous
from round68_selector import components
from round4_gates import apply_gate

def extend(x,relation):return np.concatenate([x,relation,relation-relation.mean(1,keepdims=True)],2)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    import lightgbm as lgb
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());ss,g=components(bank,arrays,p/'baseline');root,extra,flow,pairs,ptr,packets,structure=build(bank,cfg['structure_arm'])
    with np.load(p/'space.npz') as w:context=project(arrays,w)
    with np.load(p/'flow.npz') as w:score=np.asarray(flow@w['weight']).reshape(-1,4)
    x,bs=previous(bank,ss,cfg['weights'],g,root,extra,context,'context_influence');x=augment(x,score)
    if cfg['arm']=='relations':
        with np.load(p/'relation.npz') as w:prob=probabilities(pairs,question_context(arrays),np.repeat(np.arange(len(bank)),np.diff(ptr)),w)
        x=extend(x,summarize(prob,ptr,packets))
    if cfg['structure_arm']!='control':
        with np.load(p/'structure.npz') as w:score=np.asarray(structure@w['weight']).reshape(-1,4)
        x=augment(x,score)
    m=lgb.Booster(model_file=str(p/'meta.txt'));return apply_gate(bank,bs+m.predict(x.reshape(-1,x.shape[-1]),num_threads=4).reshape(-1,4),'parse_and_repair',4).astype(int).tolist()
