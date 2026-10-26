"""Compact multiscale RBF residual, train-only anchors and projections."""
import json
from pathlib import Path
import numpy as np
from features import prepare
from round3_features import grounding
from round25_selector import scores
from round4_gates import apply_gate


def compact(bank,arrays,space):
    p=prepare(bank,arrays)
    e=np.clip(((p['embedding']-space['pca_mean'])@space['pca'].T)/space['pca_scale'],-8,8)
    small=np.concatenate([p['base'],p['numeric'][:,:,5:],p['checks2'],grounding(bank)],2)
    small=np.clip((small-space['small_mean'])/space['small_scale'],-8,8)
    return np.concatenate([e/8,(e-e.mean(1,keepdims=True))/8,small/np.sqrt(small.shape[-1])],2).reshape(len(bank)*4,-1).astype(np.float32)


def kernel(x,space):
    anchors=space['anchors']
    distance=np.maximum((x*x).sum(1)[:,None]+(anchors*anchors).sum(1)[None,:]-2*x@anchors.T,0)/space['distance_median']
    return np.concatenate([np.exp(-g*distance) for g in (.3,3.,10.)],1).astype(np.float32)


def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text())
    s=scores(bank,arrays,path/'baseline')
    with np.load(path/'kernel.npz',allow_pickle=False) as z:space={k:z[k] for k in z.files}
    k=kernel(compact(bank,arrays,space),space)
    delta=((k-space['mean'])/space['scale'])@space['weight']
    return apply_gate(bank,s+cfg['alpha']*delta.reshape(-1,4),'parse_and_repair',4).astype(int).tolist()
