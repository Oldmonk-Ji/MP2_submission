"""Two-layer raw-text Transformer, no external pretrained weights or corpus."""
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from round245_text import positions
from round241_network import loss

SEED=245245
EPOCHS=40
PRETRAIN=6

class Comparator(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.embedding=nn.Embedding(vocab,48,padding_idx=0)
        self.number=nn.Linear(4,48)
        self.conv=nn.Conv1d(48,48,7,stride=4,padding=3)
        self.local_norm=nn.LayerNorm(48)
        self.segment=nn.Embedding(2,48)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(48,4,96,.1,activation='relu',batch_first=True,norm_first=True) for _ in range(2)])
        self.final_norm=nn.LayerNorm(48)
        self.head=nn.Sequential(nn.Linear(192,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,1))
        self.dropout=nn.Dropout(.1)
        self.register_buffer('position',torch.as_tensor(positions()),persistent=False)

    def local(self,ids,numbers):
        x=self.dropout(self.embedding(ids)+self.number(numbers))
        h=self.conv(x.transpose(1,2)).transpose(1,2)+x[:,::4]
        return F.relu(self.local_norm(h)),ids[:,::4]!=0

    def encode(self,q,c,qn,cn):
        n=len(q);qh,qm=self.local(q,qn)
        ch,cm=self.local(c.reshape(n*4,-1),cn.reshape(n*4,-1,4))
        qh=qh[:,None].expand(-1,4,-1,-1).reshape(n*4,-1,48)
        qm=qm[:,None].expand(-1,4,-1).reshape(n*4,-1)
        h=torch.cat([qh+self.segment.weight[0],ch+self.segment.weight[1]],1)+self.position
        mask=torch.cat([qm,cm],1)
        for layer in self.layers:h=layer(h,src_key_padding_mask=~mask)
        return self.final_norm(h),qm,cm

    def forward(self,q,c,qn,cn):
        n=len(q);h,qm,cm=self.encode(q,c,qn,cn);ql=qm.shape[1]
        qmean=(h[:,:ql]*qm[:,:,None]).sum(1)/qm.sum(1,keepdim=True).clamp_min(1)
        ch=h[:,ql:]
        mean=(ch*cm[:,:,None]).sum(1)/cm.sum(1,keepdim=True).clamp_min(1)
        maximum=ch.masked_fill(~cm[:,:,None],-1e4).amax(1)
        pooled=torch.cat([mean,maximum],-1).reshape(n,4,96)
        raw=self.head(torch.cat([pooled,pooled-pooled.mean(1,keepdim=True)],-1)).squeeze(-1)
        return raw,qmean.reshape(n,4,48).mean(1),mean.reshape(n,4,48).mean(1)

def export(model,path):
    state={k:v.detach().numpy() for k,v in model.state_dict().items()}
    embedding=state.pop('embedding.weight')
    scale=np.maximum(np.abs(embedding).max(1)/127,1e-5).astype(np.float16)
    state={k:v.astype(np.float16) for k,v in state.items()}
    state['embedding.weight']=np.clip(np.rint(embedding/scale.astype(np.float32)[:,None]),-127,127).astype(np.int8)
    state['embedding_scale']=scale
    np.savez(path,**state)
    decoded={k:torch.as_tensor(v.astype(np.float32)) for k,v in state.items() if k!='embedding_scale'}
    decoded['embedding.weight']=torch.as_tensor(state['embedding.weight'].astype(np.float32)*scale.astype(np.float32)[:,None])
    model.load_state_dict(decoded)

def predict(model,data,rows=None):
    if rows is None:rows=np.arange(len(data['q']))
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(rows),32):
            ix=rows[start:start+32]
            result.append(model(**{k:torch.as_tensor(v[ix]) for k,v in data.items()})[0].numpy())
    return np.concatenate(result)

def train(data,y,rows,vocab,path):
    torch.set_num_threads(4);torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    model=Comparator(vocab);decoder=nn.Linear(48,vocab)
    optimizer=torch.optim.AdamW(list(model.parameters())+list(decoder.parameters()),lr=.001,weight_decay=.02)
    tensors={k:torch.as_tensor(v) for k,v in data.items()};labels=torch.as_tensor(y)
    tick=time.perf_counter();history=[]
    for epoch in range(1,PRETRAIN+1):
        model.train();order=rng.permutation(rows);total=0.;count=0
        for start in range(0,len(order),64):
            ix=order[start:start+64];batch={k:v[ix].clone() for k,v in tensors.items()}
            qt=batch['q'][:,::4].clone();ct=batch['c'][:,:,::4].clone()
            mq=(qt>=4)&(torch.rand(qt.shape)<.15);mc=(ct>=4)&(torch.rand(ct.shape)<.15)
            batch['q'][:,::4]=torch.where(mq,torch.ones_like(qt),qt)
            batch['c'][:,:,::4]=torch.where(mc,torch.ones_like(ct),ct)
            batch['qn'][:,::4]=batch['qn'][:,::4].masked_fill(mq[:,:,None],0)
            batch['cn'][:,:,::4]=batch['cn'][:,:,::4].masked_fill(mc[:,:,:,None],0)
            h,_,_=model.encode(**batch)
            target=torch.cat([qt[:,None].expand(-1,4,-1),ct],-1).reshape(len(ix)*4,-1)
            chosen=torch.cat([mq[:,None].expand(-1,4,-1),mc],-1).reshape(len(ix)*4,-1)
            if not chosen.any():continue
            value=F.cross_entropy(decoder(h[chosen]),target[chosen])
            optimizer.zero_grad();value.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            n=int(chosen.sum());total+=float(value.detach())*n;count+=n
        if epoch%2==0:
            rec=dict(stage='TRAIN-only masked word/number binding',epoch=epoch,loss=total/max(count,1),seconds=time.perf_counter()-tick)
            history.append(rec);print('PRETRAIN',path.parent.name,rec,flush=True)
    del decoder,optimizer
    ema=copy.deepcopy(model);optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.03)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=.00008)
    for epoch in range(1,EPOCHS+1):
        model.train();order=rng.permutation(rows);total=0.
        for start in range(0,len(order),32):
            ix=order[start:start+32];raw,q,c=model(**{k:v[ix] for k,v in tensors.items()})
            value=loss(raw,q,c,labels[ix]);assert torch.isfinite(value)
            optimizer.zero_grad();value.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.mul_(.98).add_(b,alpha=.02)
            total+=float(value.detach())*len(ix)
        scheduler.step()
        if epoch%10==0:
            rec=dict(stage='supervised selection',epoch=epoch,loss=total/len(rows),seconds=time.perf_counter()-tick)
            history.append(rec);print('TRAIN',path.parent.name,rec,flush=True)
    export(ema,path)
    return ema,history
