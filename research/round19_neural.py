"""Bounded real-candidate experiment: capacity, regularization, longer traces."""
from round19_common import OUT, load_part, read_plan, dump, sha
from round19_model import (Comparator,vocabulary,encode_bank,tensor_batch,objective,predict_data,save_model,LEFT,RIGHT)
from round19_select import evaluate_development
import copy, json, time
import numpy as np
import torch


def training_loss(scores,margins,y,smooth):
    if not smooth:return objective(scores,margins,y)
    diff=y[:,LEFT]-y[:,RIGHT];eligible=diff.ne(0);count=eligible.sum(1)
    z=diff*margins
    loss=((1-smooth)*torch.nn.functional.softplus(-z)+smooth*torch.nn.functional.softplus(z))*eligible
    pair=loss.sum(1)/count.clamp_min(1)
    group=torch.logsumexp(scores,1)-torch.logsumexp(scores.masked_fill(y<.5,-1e4),1)
    return (pair[count>0]+.25*group[count>0]).mean()


def fit_config(cfg,b,a,y,db,da,dy,path):
    path.mkdir(exist_ok=True);v=vocabulary(b,cfg['vocab_limit'])
    dump(path/'config.json',cfg);dump(path/'vocabulary.json',v)
    Encoder=encode_bank;Model=Comparator
    if cfg.get('anchored'):
        from round19_anchored import encode_anchored,AnchoredComparator
        Encoder=encode_anchored;Model=AnchoredComparator
    x,coverage=Encoder(b,a,v,cfg);dx,dcov=Encoder(db,da,v,cfg)
    dump(path/'coverage.json',dict(fit=coverage,development=dcov))
    e=a['embeddings'].reshape(-1,1024);mean=e.mean(0);scale=e.std(0).clip(.001)
    torch.manual_seed(cfg['seed']);torch.set_num_threads(4);rng=np.random.default_rng(cfg['seed'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=Model(cfg,len(v),mean,scale).to(device);ema=copy.deepcopy(model) if cfg['ema'] else None
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    mixed=np.flatnonzero((y.sum(1)>0)&(y.sum(1)<4));best_loss=float('inf');best_epoch=0;history=[];tick=time.perf_counter()
    for epoch in range(1,cfg['epochs']+1):
        start=time.perf_counter();model.train();loss_sum=0.
        order=rng.permutation(mixed)
        for i in range(0,len(order),cfg['batch']):
            ix=order[i:i+cfg['batch']];target=torch.tensor(y[ix],device=device)
            s,m=model(**tensor_batch(x,ix,device));loss=training_loss(s,m,target,cfg['smooth'])
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if ema is not None:
                with torch.no_grad():
                    for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,1-cfg['ema'])
            loss_sum+=float(loss.detach())*len(ix)
        evaluated=ema if ema is not None else model
        s,m=predict_data(evaluated,dx,device,cfg['batch'])
        dev_loss=float(objective(torch.tensor(s),torch.tensor(m),torch.tensor(dy)))
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(mixed),development_loss=dev_loss,seconds=time.perf_counter()-start))
        if dev_loss<best_loss-1e-4:
            best_loss=dev_loss;best_epoch=epoch;save_model(evaluated,path)
            np.savez(path/'development.npz',scores=s,margins=m)
        dump(path/'history.json',history)
        print(cfg['name'],'epoch',epoch,'train',round(loss_sum/len(mixed),4),'dev_loss',round(dev_loss,4),'best',best_epoch,'sec',round(history[-1]['seconds'],2),flush=True)
        if epoch-best_epoch>=cfg['patience']:break
    asset_bytes=sum((path/k).stat().st_size for k in ('model.npz','config.json','vocabulary.json'))
    with np.load(path/'development.npz') as z:s=z['scores']
    result=dict(name=cfg['name'],family='neural',path=str(path.relative_to(OUT)),epoch=best_epoch,
                development_loss=best_loss,epochs_run=len(history),seconds=time.perf_counter()-tick,
                parameters=sum(p.numel() for p in model.parameters()),asset_bytes=asset_bytes,
                device=device,plan_sha=sha(OUT/'plan.json'),**evaluate_development(s,db,dy,asset_bytes))
    dump(path/'selected.json',result);return result


def main():
    assert not (OUT/'frozen.json').exists();assert (OUT/'baseline/done.json').exists()
    b,a,y,_=load_part('fit');db,da,dy,_=load_part('development');records=[]
    root=OUT/'neural';root.mkdir(exist_ok=True)
    configs=list(read_plan()['neural'])
    if (OUT/'amendment_01.json').exists():configs+=json.loads((OUT/'amendment_01.json').read_text())['neural']
    for cfg in configs:
        path=root/cfg['name']
        if (path/'selected.json').exists():r=json.loads((path/'selected.json').read_text())
        else:r=fit_config(cfg,b,a,y,db,da,dy,path)
        records.append(r);print('NEURAL SELECTED',r['name'],r['epoch'],r['selected'],flush=True)
    dump(root/'selection.json',records)


if __name__=='__main__':main()
