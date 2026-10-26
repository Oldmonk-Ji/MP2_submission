"""Small policy over explicit executable relation hypotheses."""
import numpy as np,torch
from torch import nn
from scipy import sparse
from pathlib import Path
from round44_programs import LIMIT,DIM

def model():return nn.Sequential(nn.Linear(DIM,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
def weights(m):
    w={}
    for i,l in enumerate(x for x in m if isinstance(x,nn.Linear)):w[f'w{i}']=l.weight.detach().cpu().numpy().T.copy();w[f'b{i}']=l.bias.detach().cpu().numpy().copy()
    return w
def forward(x,w):
    h=x@w['w0']+w['b0'];h=np.maximum(h,0);h=np.maximum(h@w['w1']+w['b1'],0)
    return (h@w['w2']+w['b2']).reshape(-1)
def predict(m,x):return forward(x,weights(m)).reshape(-1,LIMIT)
def valid_mask(mask):
    m=mask.copy();m[~m.any(1),0]=True;return m
def probability(s,mask):
    s=np.where(valid_mask(mask),s,-1e9);s=s-s.max(1,keepdims=True);p=np.exp(s);return p/p.sum(1,keepdims=True)
def decisions(s,mask,matches,base,base_scores):
    p=probability(s,mask);mass=np.einsum('nh,nhc->nc',p,matches);hard=base.copy()
    for k,j in enumerate(safe_argmax(s,mask)):
        eligible=np.flatnonzero(matches[k,j]) if mask[k].any() else []
        if len(eligible):hard[k]=max(eligible,key=lambda i:base_scores[k,i])
    from round22_selector import normalized
    from round4_gates import apply_gate
    residual=normalized(np.log(mass.clip(1e-6,1)))
    return hard,residual,p
def safe_argmax(s,mask):return np.where(valid_mask(mask),s,-1e9).argmax(1)

def loss(scores,rewards,mask,ref,arm,kl=.02):
    scores=scores.masked_fill(~mask,-1e9);lp=scores.log_softmax(1);p=lp.exp();rlp=ref.masked_fill(~mask,-1e9).log_softmax(1)
    penalty=(p*(lp-rlp)).sum(1)
    if arm=='sft':
        available=rewards.gt(.5).any(1)
        nll=torch.logsumexp(scores,1)-torch.logsumexp(scores.masked_fill(rewards<.5,-1e9),1)
        target=torch.where(available,nll,torch.zeros_like(nll))
    else:target=-(p*rewards).sum(1)
    return (target+kl*penalty).mean()
