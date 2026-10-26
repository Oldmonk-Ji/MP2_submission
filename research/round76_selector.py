from pathlib import Path
import json,numpy as np,torch
from round76_text import candidate_sentences,encode
from round76_model import Model
from round25_selector import scores as baseline
from round4_gates import apply_gate

def score_tokens(sequences,path):
    path=Path(path);cfg=json.loads((path/'config.json').read_text());vocab=json.loads((path/'vocabulary.json').read_text());x,seg=encode(sequences,vocab)
    torch.set_num_threads(4);m=Model(len(vocab),cfg['ordered'])
    with np.load(path/'model.npz') as w:m.load_state_dict({k:torch.from_numpy(w[k].copy()) for k in w.files})
    m.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(x),64):out.append(m(torch.from_numpy(x[start:start+64]),torch.from_numpy(seg[start:start+64])).numpy())
    return np.concatenate(out)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);sequences=[s for r in bank for s in candidate_sentences(r)]
    residual=score_tokens(sequences,path).reshape(-1,4);bs=baseline(bank,arrays,path/'baseline')
    return apply_gate(bank,bs+residual,'parse_and_repair',4).astype(int).tolist()
