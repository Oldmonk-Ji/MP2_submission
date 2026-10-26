from pathlib import Path
import json,numpy as np
from round68_selector import components,design
from round70_selector import scores as attention_scores
from round59_selector import features
from round4_gates import apply_gate

def augmented(bank,ss,weights,g,confidence):
    x,bs=design(bank,ss,weights,g)
    if confidence:
        centered=ss-ss.mean(1,keepdims=True);std=ss.std(1,keepdims=True);z=centered/(std+1e-5)
        p=np.exp(ss-ss.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        entropy=-(p*np.log(p.clip(1e-8,1))).sum(1,keepdims=True)/np.log(4)
        x=np.concatenate([x,centered,p,z*np.log1p(std),z*(1-entropy)],2).astype(np.float32)
    return x,bs

def predict(bank,arrays,checkpoint):
    if not bank:return []
    import lightgbm as lgb
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());ss,g=components(bank,arrays,p/'baseline')
    if cfg['attention']:
        x,o=features(bank,arrays);s=attention_scores(x,o,p/'attention.npz',True);ss=np.concatenate([ss,s[:,:,None]],2)
    x,bs=augmented(bank,ss,cfg['weights'],g,cfg['confidence'])
    m=lgb.Booster(model_file=str(p/'meta.txt'));s=bs+m.predict(x.reshape(-1,x.shape[-1]),num_threads=4).reshape(-1,4)
    return apply_gate(bank,s,'parse_and_repair',4).astype(int).tolist()
