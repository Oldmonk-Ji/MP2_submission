"""Answer-group evidence correction. Inference reads inputs and frozen assets only."""
from pathlib import Path
import json
import numpy as np
from round3_selector import leaf_scores
from round9_group_features import build,normalize,candidate_scores
from round4_evidence import evidence
from round4_gates import apply_gate

def scores(bank,arrays,checkpoint):
    import lightgbm as lgb
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text());basepath=path/'base';bc=json.loads((basepath/'config.json').read_text());ss=[leaf_scores(bank,arrays,basepath/m['directory']) for m in bc['members']]
    members=np.stack(ss,2);base=sum(normalize(s)*m['weight'] for s,m in zip(ss,bc['members']))
    x,gb,mapping,sizes,meta=build(bank,members,base,evidence(bank),cfg['kind']);model=lgb.Booster(model_file=str(path/'correction.txt'));delta=model.predict(x,num_threads=4)
    return candidate_scores(gb+cfg['alpha']*delta,mapping,base,bank)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    cfg=json.loads((Path(checkpoint)/'config.json').read_text());s=scores(bank,arrays,checkpoint)
    return apply_gate(bank,s,cfg['gate_kind'],cfg['margin_limit']).astype(int).tolist()
