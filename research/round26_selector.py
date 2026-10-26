"""A frozen learned arithmetic-hypothesis residual over an eligible ensemble."""
import json
from pathlib import Path
import numpy as np
from round26_programs import transform
from round25_selector import scores
from round4_gates import apply_gate


def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text())
    s=scores(bank,arrays,path/'baseline')
    with np.load(path/'residual.npz',allow_pickle=False) as z:w=z['weight']
    delta=np.asarray(transform(bank,cfg['context'])@w).reshape(-1,4)
    return apply_gate(bank,s+cfg['alpha']*delta,'parse_and_repair',4).astype(int).tolist()
