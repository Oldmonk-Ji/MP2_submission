"""Scratch text comparator: strided local encoding and cross attention."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

SEED=241241
EPOCHS=60

class TextComparator(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.embedding=nn.Embedding(vocab,24,padding_idx=0)
        self.number=nn.Linear(4,24)
        self.conv=nn.Conv1d(24,24,3,stride=2,padding=1)
        self.norm=nn.LayerNorm(24)
        self.attention=nn.Linear(24,16,bias=False)
        self.compare=nn.Linear(96,32)
        self.head=nn.Sequential(nn.Linear(256,48),nn.ReLU(),nn.Dropout(.15),nn.Linear(48,1))
        self.dropout=nn.Dropout(.1)

    def local(self,ids,numbers):
        mask=ids[:,::2]!=0
        x=self.dropout(self.embedding(ids)+self.number(numbers))
        h=self.conv(x.transpose(1,2)).transpose(1,2)+x[:,::2]
        return F.relu(self.norm(h)),mask

    def pool(self,h,mask):
        mean=(h*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        maximum=h.masked_fill(~mask[:,:,None],-1e4).amax(1)
        return torch.cat([mean,maximum],-1)

    def forward(self,q,c,qn,cn):
        n=q.shape[0]
        qh,qm=self.local(q,qn)
        ch,cm=self.local(c.reshape(-1,c.shape[-1]),cn.reshape(-1,cn.shape[-2],4))
        auxiliary_q=(qh*qm[:,:,None]).sum(1)/qm.sum(1,keepdim=True).clamp_min(1)
        auxiliary_c=((ch*cm[:,:,None]).sum(1)/cm.sum(1,keepdim=True).clamp_min(1)).reshape(n,4,-1).mean(1)
        qh=qh[:,None].expand(-1,4,-1,-1).reshape(n*4,qh.shape[1],24)
        qm=qm[:,None].expand(-1,4,-1).reshape(n*4,-1)
        alignment=self.attention(qh)@self.attention(ch).transpose(1,2)/4
        qc=F.softmax(alignment.masked_fill(~cm[:,None,:],-1e4),-1)@ch
        cq=F.softmax(alignment.transpose(1,2).masked_fill(~qm[:,None,:],-1e4),-1)@qh
        qcompare=F.relu(self.compare(torch.cat([qh,qc,qh-qc,qh*qc],-1)))
        ccompare=F.relu(self.compare(torch.cat([ch,cq,ch-cq,ch*cq],-1)))
        h=torch.cat([self.pool(qcompare,qm),self.pool(ccompare,cm)],-1).reshape(n,4,-1)
        raw=self.head(torch.cat([h,h-h.mean(1,keepdim=True)],-1)).squeeze(-1)
        return raw,auxiliary_q,auxiliary_c

def loss(raw,q,c,y):
    mixed=(y.sum(1)>0)&(y.sum(1)<4)
    margin=raw[:,:,None]-raw[:,None,:]
    mask=y[:,:,None]*(1-y[:,None,:])
    pair=(F.softplus(-margin)*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
    group=torch.logsumexp(raw,1)-torch.logsumexp(raw.masked_fill(y<.5,-1e4),1)
    rank=(pair[mixed]+.5*group[mixed]).mean() if mixed.any() else raw.sum()*0
    # All sentences come from supplied TRAIN; topic alignment has no external supervision.
    similarity=F.normalize(q,dim=-1)@F.normalize(c,dim=-1).T/.1
    target=torch.arange(len(q))
    contrast=.5*(F.cross_entropy(similarity,target)+F.cross_entropy(similarity.T,target))
    return rank+.25*F.binary_cross_entropy_with_logits(raw,y)+.05*contrast

def predict(model,data,rows=None):
    if rows is None:rows=np.arange(len(data['q']))
    result=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(rows),64):
            ix=rows[start:start+64]
            result.append(model(**{k:torch.as_tensor(v[ix]) for k,v in data.items()})[0].numpy())
    return np.concatenate(result)

def train(data,y,rows,vocab,path):
    import time
    torch.set_num_threads(4);torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    model=TextComparator(vocab);ema=copy.deepcopy(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=.0001)
    tensors={k:torch.as_tensor(v) for k,v in data.items()};labels=torch.as_tensor(y)
    tick=time.perf_counter();history=[]
    for epoch in range(1,EPOCHS+1):
        model.train();order=rng.permutation(rows);total=0.
        for start in range(0,len(order),64):
            ix=order[start:start+64]
            raw,q,c=model(**{k:v[ix] for k,v in tensors.items()})
            value=loss(raw,q,c,labels[ix])
            assert torch.isfinite(value)
            optimizer.zero_grad();value.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.mul_(.98).add_(b,alpha=.02)
            total+=float(value.detach())*len(ix)
        scheduler.step()
        if epoch%10==0:
            rec=dict(epoch=epoch,loss=total/len(rows),seconds=time.perf_counter()-tick)
            history.append(rec);print('TRAIN',path.parent.name,rec,flush=True)
    # Evaluate the exact FP16-stored values, using float32 arithmetic.
    state={k:v.detach().numpy().astype(np.float16) for k,v in ema.state_dict().items()}
    np.savez(path,**state)
    ema.load_state_dict({k:torch.as_tensor(v.astype(np.float32)) for k,v in state.items()})
    return ema,history
