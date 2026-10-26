"""Matched continuation from frozen local-policy warm starts on repaired evidence."""
from data_access import ROOT
from round22_common import read,dump,sha
import round48_train as training
import torch,numpy as np
from torch import nn

OUT=ROOT/'experiments/round49'

if __name__=='__main__':
    files=[ROOT/'experiments/round48'/stage/f'warm_s{seed}'/'model.npz' for stage in ('diagnostic','full') for seed in (48048,48049)]
    dump(OUT/'warm_start_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in files},reference_recomputed_on_new_action_bank=True))
    original_fit=training.fit
    def fit(m,x,ptr,rewards,mask,fitrows,pairrows,xp,xn,seed,device,arm,reference=None):
        if reference is not None:return original_fit(m,x,ptr,rewards,mask,fitrows,pairrows,xp,xn,seed,device,arm,reference)
        stage='full' if len(fitrows)==4743 else 'diagnostic';path=ROOT/'experiments/round48'/stage/f'warm_s{seed}'
        with np.load(path/'model.npz') as w:
            for i,l in enumerate(v for v in m.net if isinstance(v,nn.Linear)):
                with torch.no_grad():l.weight.copy_(torch.from_numpy(w[f'w{i}'].T.copy()).to(device));l.bias.copy_(torch.from_numpy(w[f'b{i}'].copy()).to(device))
        return m.eval(),read(path/'training.json')['history']
    training.OUT=OUT;training.fit=fit;training.main()
