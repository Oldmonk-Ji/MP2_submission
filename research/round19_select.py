from round19_common import OUT, read_plan, normalized, paired
import functools, json
import numpy as np
import round4_gates
from features import metadata
round4_gates.trace=functools.lru_cache(maxsize=8192)(round4_gates.trace)


def selection_key(v):return (v['correct'],-v['damaged'],-v['changed'],-v['alpha'],-v.get('asset_bytes',0))


def evaluate_development(scores,bank,y,asset_bytes):
    with np.load(OUT/'baseline/development.npz') as z:base=z['scores'];bc=z['choices']
    meta=metadata(bank);records=[]
    for alpha in read_plan()['alphas']:
        choices=round4_gates.apply_gate(bank,base+alpha*normalized(scores),'parse_and_repair',4)
        records.append(dict(alpha=alpha,asset_bytes=asset_bytes,**paired(choices,bc,y,meta['hard'])))
    single=round4_gates.apply_gate(bank,scores,'parse_and_repair',4)
    return dict(selected=max(records,key=selection_key),variants=records,standalone=paired(single,bc,y,meta['hard']))
