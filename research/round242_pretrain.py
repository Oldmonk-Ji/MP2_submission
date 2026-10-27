"""Local masked-word pretraining on provided TRAIN, then supervised selection."""
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from round241_network import TextComparator,loss,EPOCHS,SEED
from round241_text import SPECIAL,CLEN,QLEN

PRETRAIN_EPOCHS=6

def train(data,y,rows,vocab,path):
    torch.set_num_threads(4);torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    model=TextComparator(vocab)
    decoder=nn.Linear(24,vocab)
    optimizer=torch.optim.AdamW(list(model.parameters())+list(decoder.parameters()),lr=.002,weight_decay=.01)
    q=np.pad(data['q'][rows],((0,0),(0,CLEN-QLEN)))
    qn=np.pad(data['qn'][rows],((0,0),(0,CLEN-QLEN),(0,0)))
    words=torch.as_tensor(np.concatenate([q,data['c'][rows].reshape(-1,CLEN)]))
    numbers=torch.as_tensor(np.concatenate([qn,data['cn'][rows].reshape(-1,CLEN,4)]))
    tick=time.perf_counter();history=[]
    for epoch in range(1,PRETRAIN_EPOCHS+1):
        model.train();decoder.train();total=0.;count=0
        order=rng.permutation(len(words))
        for start in range(0,len(order),128):
            ix=order[start:start+128]
            original=words[ix];masked=original.clone()
            target=original[:,::2]
            choose=(target>=len(SPECIAL))&(torch.rand(target.shape)<.15)
            if not choose.any():continue
            masked[:,::2]=torch.where(choose,torch.ones_like(target),target)
            representation,_=model.local(masked,numbers[ix])
            value=F.cross_entropy(decoder(representation[choose]),target[choose])
            optimizer.zero_grad();value.backward()
            nn.utils.clip_grad_norm_(list(model.parameters())+list(decoder.parameters()),5)
            optimizer.step();n=int(choose.sum());total+=float(value.detach())*n;count+=n
        if epoch%2==0:
            rec=dict(stage='TRAIN-only masked pretraining',epoch=epoch,loss=total/max(count,1),seconds=time.perf_counter()-tick)
            history.append(rec);print('PRETRAIN',path.parent.name,rec,flush=True)
    del words,numbers,decoder,optimizer
    # No pretraining head is exported. The inference architecture is unchanged.
    ema=copy.deepcopy(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=.0001)
    tensors={k:torch.as_tensor(v) for k,v in data.items()};labels=torch.as_tensor(y)
    rng=np.random.default_rng(SEED)
    for epoch in range(1,EPOCHS+1):
        model.train();order=rng.permutation(rows);total=0.
        for start in range(0,len(order),64):
            ix=order[start:start+64]
            raw,q,c=model(**{k:v[ix] for k,v in tensors.items()})
            value=loss(raw,q,c,labels[ix]);assert torch.isfinite(value)
            optimizer.zero_grad();value.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.mul_(.98).add_(b,alpha=.02)
            total+=float(value.detach())*len(ix)
        scheduler.step()
        if epoch%10==0:
            rec=dict(stage='supervised selection',epoch=epoch,loss=total/len(rows),seconds=time.perf_counter()-tick)
            history.append(rec);print('TRAIN',path.parent.name,rec,flush=True)
    state={k:v.detach().numpy().astype(np.float16) for k,v in ema.state_dict().items()}
    np.savez(path,**state)
    ema.load_state_dict({k:torch.as_tensor(v.astype(np.float32)) for k,v in state.items()})
    return ema,history
