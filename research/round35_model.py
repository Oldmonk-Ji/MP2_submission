"""Joint question relation supervision and official candidate representation."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from round32_graph import GraphModel


class JointModel(GraphModel):
    def __init__(self,vocabulary_size):
        super().__init__(vocabulary_size)
        self.register_buffer('embedding_mean',torch.zeros(1024))
        self.register_buffer('embedding_scale',torch.ones(1024))
        self.candidate=nn.Sequential(nn.Linear(2052,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))

    def joint_score(self,logp,roles,pairs,embeddings):
        semantic=self.score(logp,roles,pairs)
        available=((roles>=0).any(-1)|(pairs>=0).any((-1,-2))).float()
        semantic=semantic*available
        e=(embeddings-self.embedding_mean)/self.embedding_scale
        local=torch.stack([semantic,semantic-semantic.mean(1,keepdim=True),available,available.mean(1,keepdim=True).expand_as(available)],-1)
        features=torch.cat([e,e-e.mean(1,keepdim=True),local],-1)
        return self.candidate(features).squeeze(-1)


def load(path):
    path=Path(path);vocab=json.loads((path/'vocabulary.json').read_text(encoding='utf-8'));model=JointModel(len(vocab))
    with np.load(path/'model.npz',allow_pickle=False) as z:model.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    return model.eval(),vocab


@torch.no_grad()
def predict_encoded(model,q,roles,pairs,embeddings,device='cpu',batch=64):
    from round29_model import trim
    out=[];model.eval()
    for start in range(0,len(q),batch):
        ix=slice(start,start+batch)
        qq=torch.as_tensor(trim(q[ix]),device=device)
        rr=torch.as_tensor(roles[ix],device=device);pp=torch.as_tensor(pairs[ix],device=device);ee=torch.as_tensor(embeddings[ix],device=device)
        out.append(model.joint_score(model(qq),rr,pp,ee).cpu().numpy())
    return np.concatenate(out)
