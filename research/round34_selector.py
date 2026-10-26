from pathlib import Path
import json
import numpy as np
from round34_features import transform
from round29_programs import parse
from round25_selector import scores
from round4_gates import apply_gate


def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text());base=scores(bank,arrays,path/'baseline')
    x=transform([(r['question'],parse(r['question'],c['text'])) for r in bank for c in r['candidates']])
    with np.load(path/'model.npz',allow_pickle=False) as z:delta=np.asarray(x@z['weight']).reshape(-1,4)
    return apply_gate(bank,base+cfg['alpha']*delta,'parse_and_repair',4).astype(int).tolist()
