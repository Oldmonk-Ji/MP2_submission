from round22_common import ROOT,read,dump
from round76_model import Model
from round76_text import encode
from round76_selector import score_tokens
import torch,numpy as np

if __name__=='__main__':
    torch.set_num_threads(4);p=ROOT/'experiments/round77/diagnostic/ordered';cfg=read(p/'config.json');voc=read(p/'vocabulary.json')
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    ts=read(ROOT/'experiments/round76/train_tokens.json');ss=[s for i in np.flatnonzero(f!=2) for s in ts[i]][:64];x,seg=encode(ss,voc)
    m=Model(len(voc),cfg['ordered'])
    with np.load(p/'model.npz') as w:m.load_state_dict({k:torch.from_numpy(w[k].copy()) for k in w.files})
    m.eval();result={}
    for fast in (True,False):
        torch.backends.mha.set_fastpath_enabled(fast);scores={}
        for device in ('cpu','cuda'):
            m=m.to(device)
            with torch.no_grad():scores[device]=m(torch.tensor(x,device=device),torch.tensor(seg,device=device)).cpu().numpy()
        result[str(fast)]=dict(max_error=float(np.max(np.abs(scores['cpu']-scores['cuda']))),first8cpu=scores['cpu'][:8].tolist(),first8gpu=scores['cuda'][:8].tolist(),candidate_argmax_differences=int((scores['cpu'].reshape(-1,4).argmax(1)!=scores['cuda'].reshape(-1,4).argmax(1)).sum()))
    dump(ROOT/'experiments/round77/numerics.json',result);print(result)
