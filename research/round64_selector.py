"""Answer-level evidence aggregation ablation, independent of enumeration count."""
from pathlib import Path
import json,numpy as np
from round56_selector import infer
from round25_selector import scores as baseline
from round22_selector import normalized
from round50_selector import complete
from round4_gates import apply_gate
from scipy.special import logsumexp
from round46_selector import probability

def select(bank,bs,s,mask,matches,pool):
    values=np.zeros_like(bs);p=probability(s,mask)
    for i in range(len(bank)):
        for j in range(4):
            z=p[i,mask[i]&matches[i,:,j]]
            if len(z):values[i,j]=z.max() if pool=='max' else z.sum()/(len(z) if pool=='mean' else 1)
    values=np.log(values.clip(1e-6,1))
    base=apply_gate(bank,bs,'parse_and_repair',4)
    c=apply_gate(bank,bs+.25*normalized(values),'parse_and_repair',4)
    return np.where(complete(bank,matches),c,base)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text())
    bs=baseline(bank,arrays,p/'baseline');s,m,k=infer(bank,[p/'policy0',p/'policy1'])
    return select(bank,bs,s,m,k,cfg['pool']).astype(int).tolist()
