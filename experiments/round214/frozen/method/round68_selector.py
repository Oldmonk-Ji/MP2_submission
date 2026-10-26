"""OOF-trained stacking of unchanged full-TRAIN experts."""
from pathlib import Path
import json,numpy as np
from features import prepare,metadata
from round3_features import build,grounding
from round3_selector import leaf_scores
from round22_selector import dense_forward,normalized
from round4_gates import apply_gate

def components(bank,arrays,path):
    path=Path(path);cfg=json.loads((path/'config.json').read_text());p=prepare(bank,arrays);g=grounding(bank);cache={};ss=[]
    for member in cfg['members']:
        sub=path/member['directory'];mc=json.loads((sub/'config.json').read_text())
        if mc['family']=='mlp':
            kind=mc['features']
            if kind not in cache:cache[kind]=build(bank,arrays,kind,prepared=p,ground=g)
            with np.load(sub/'model.npz') as w:s=dense_forward(cache[kind],w)
        else:s=leaf_scores(bank,arrays,sub)
        ss.append(s)
    return np.stack(ss,2),g

def design(bank,ss,weights,g):
    z=normalized(ss);base=z@np.asarray(weights);meta=metadata(bank);own=[];diff=[]
    for i,answers in enumerate(meta['answers']):
        aa=[];dd=[]
        for j,a in enumerate(answers):
            same=np.array([k==j or (a is not None and a==b) for k,b in enumerate(answers)])
            alt=~same;hi=z[i,same].max(0);lo=z[i,alt].max(0) if alt.any() else hi
            aa.append(hi);dd.append(hi-lo)
        own.append(aa);diff.append(dd)
    x=np.concatenate([z,np.asarray(own),np.asarray(diff),g[:,:,:12],g[:,:,:12]-g[:,:,:12].mean(1,keepdims=True),
        base[:,:,None],z.std(2,keepdims=True)],2).astype(np.float32)
    return x,base

def predict(bank,arrays,checkpoint):
    if not bank:return []
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());ss,g=components(bank,arrays,p/'baseline');x,bs=design(bank,ss,cfg['weights'],g)
    if cfg['arm']=='linear':
        with np.load(p/'meta.npz') as w:s=((x-w['mean'])/w['scale'])@w['weight']
    else:
        import lightgbm as lgb
        m=lgb.Booster(model_file=str(p/'meta.txt'));s=bs+m.predict(x.reshape(-1,x.shape[-1]),num_threads=4).reshape(-1,4)
    return apply_gate(bank,s,'parse_and_repair',4).astype(int).tolist()
