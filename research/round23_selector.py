"""Small nonlinear fusion of the same full-training experts; input only."""
from pathlib import Path
import numpy as np
from round22_selector import read,normalized,component_scores,side_features,linear_design,base_scores
from features import metadata
from round4_gates import apply_gate


def design(bank,components,base_weights,kind,cues=None,evidence=None):
    z=normalized(components);m=metadata(bank);same=np.zeros((len(bank),4,4),bool)
    for i,answers in enumerate(m['answers']):
        same[i]=[[j==k or (a is not None and a==b) for k,b in enumerate(answers)] for j,a in enumerate(answers)]
    other=~same
    group_mean=np.einsum('nij,njk->nik',same,z)/np.maximum(same.sum(2,keepdims=True),1)
    group_max=np.where(same[:,:,:,None],z[:,None,:,:],-1e6).max(2)
    other_max=np.where(other[:,:,:,None],z[:,None,:,:],-1e6).max(2)
    other_max=np.where(other.any(2)[:,:,None],other_max,0)
    counts=same.sum(2,keepdims=True)/4
    base=base_scores(components,base_weights,0.)
    parts=[z,group_mean,group_max,z-other_max,counts,base[:,:,None]]
    if kind=='conditional':
        if cues is None:cues,evidence=side_features(bank)
        parts.append(linear_design(z,cues,evidence,'conditional')[:,:,6:])
    return np.concatenate(parts,2).astype(np.float32),base


def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=read(path/'tree_fusion.json')
    components,extra=component_scores(bank,arrays,path)
    if cfg['scale']:
        import lightgbm as lgb
        cues,evidence=side_features(bank,extra) if cfg['features']=='conditional' else (None,None)
        x,base=design(bank,components,cfg['base_weights'],cfg['features'],cues,evidence)
        model=lgb.Booster(model_file=str(path/'readout_tree.txt'))
        delta=model.predict(x.reshape(-1,x.shape[-1]),raw_score=True,num_threads=4).reshape(-1,4)
        scores=base+cfg['scale']*delta
    else:scores=base_scores(components,cfg['base_weights'],0.)
    return apply_gate(bank,scores,cfg['gate'],cfg['gate_margin']).astype(int).tolist()
