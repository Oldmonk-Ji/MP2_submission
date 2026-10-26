from pathlib import Path
import json,numpy as np,torch
from round59_selector import features
from round70_model import Model
from round25_selector import scores as baseline
from round22_selector import normalized
from round4_gates import apply_gate

def scores(x,o,path,attention):
    torch.set_num_threads(4)
    with np.load(path) as w:
        m=Model(x.shape[-1],attention);m.load_state_dict({k[6:]:torch.from_numpy(w[k].copy()) for k in w.files if k.startswith('state:')});m.eval()
        x=np.clip((x-w['mean'])/w['scale'],-8,8)
    ss=[]
    with torch.no_grad():
        for start in range(0,len(x),128):ss.append(m(torch.from_numpy(x[start:start+128]),torch.from_numpy(1-o[start:start+128])).numpy())
    return np.concatenate(ss)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());x,o=features(bank,arrays);s=scores(x,o,p/'model.npz',cfg['attention'])
    if cfg['alpha']<1:s=(1-cfg['alpha'])*normalized(baseline(bank,arrays,p/'baseline'))+cfg['alpha']*normalized(s)
    return apply_gate(bank,s,'parse_and_repair',4).astype(int).tolist()
