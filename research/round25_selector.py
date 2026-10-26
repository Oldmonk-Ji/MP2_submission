"""Shared-feature NumPy ensemble of scratch trained candidate scorers."""
import json
from pathlib import Path
import numpy as np
from features import prepare
from round3_features import build, grounding
from round3_selector import leaf_scores
from round22_selector import dense_forward, normalized
from round4_gates import apply_gate


def scores(bank, arrays, checkpoint):
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text())
    p=prepare(bank,arrays);g=grounding(bank);cache={};result=np.zeros((len(bank),4))
    for member in cfg['members']:
        sub=path/member['directory'];mc=json.loads((sub/'config.json').read_text())
        if mc['family']=='mlp':
            kind=mc['features']
            if kind not in cache:cache[kind]=build(bank,arrays,kind,prepared=p,ground=g)
            with np.load(sub/'model.npz',allow_pickle=False) as z:s=dense_forward(cache[kind],z)
        else:s=leaf_scores(bank,arrays,sub)
        result+=member['weight']*normalized(s)
    return result


def predict(bank,arrays,checkpoint):
    if not bank:return []
    return apply_gate(bank,scores(bank,arrays,checkpoint),'parse_and_repair',4).astype(int).tolist()
