"""Matched listwise-vs-pairwise objectives, full eligible TRAIN participation."""
from round22_common import ROOT,load,dump,sha
from round3_features import build
from round59_train import Model
from round59_selector import forward
from round25_selector import scores as baseline
from round22_selector import normalized
from round4_gates import apply_gate
import numpy as np,torch,time,copy,shutil
from torch import nn
OUT=ROOT/'experiments/round66'

def fit(x,y,ids,path,seed,arm):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);mean=x.mean((0,1));scale=x.std((0,1)).clip(.01)
    xx=torch.tensor(np.clip((x-mean)/scale,-8,8),device='cuda');yy=torch.tensor(y,device='cuda')
    model=Model(x.shape[-1],False).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);ema=copy.deepcopy(model)
    for epoch in range(80):
        model.train()
        for ix in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/128))):
            t=yy[ix];s=model(xx[ix],None);bce=nn.functional.binary_cross_entropy_with_logits(s,t)
            if arm=='listwise':rank=(torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(t<.5,-1e9),1)).mean()
            else:
                valid=t[:,:,None]*(1-t[:,None,:]);margin=s[:,:,None]-s[:,None,:]
                rank=(nn.functional.softplus(-margin)*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
                rank=rank.mean()
            loss=rank+.25*bce;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,.02)
    ema.eval();w=dict(mean=mean,scale=scale)
    for name in ('enc','solo','out'):
        l=getattr(ema,name);w[name+'w']=l.weight.detach().cpu().numpy().T.copy();w[name+'b']=l.bias.detach().cpu().numpy().copy()
    path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**w);dump(path.with_suffix('.json'),dict(training_ids=ids.tolist(),seed=seed,arm=arm,epochs=80))
    return w

if __name__=='__main__':
    OUT.mkdir();torch.set_num_threads(4);tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['listwise','pairwise'],seeds=[66001,66002],epochs=80,alpha=[.1,.25,.5,1.],test_opened=False,source_hash=sha(__file__)))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');x=build(b,a,'base');vx=build(v,va,'base')
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    bs=normalized(baseline(v,va,ROOT/'experiments/round25/selected_bundle'));bc=apply_gate(v,bs,'parse_and_repair',4);records=[]
    for arm in ('listwise','pairwise'):
        hs=[];ss=[]
        for seed in (66001,66002):
            w=fit(x[f!=2],y[f!=2],ids[f!=2],OUT/'diagnostic'/f'{arm}_{seed}.npz',seed,arm);hs.append(forward(x[f==2],None,w,False))
            w=fit(x,y,ids,OUT/'full'/f'{arm}_{seed}.npz',seed,arm);ss.append(forward(vx,None,w,False));print('FIT',arm,seed,'seconds',time.perf_counter()-tick,flush=True)
        h=np.mean(hs,0);s=np.mean(ss,0);hb=[r for r,k in zip(b,f) if k==2];hc=apply_gate(hb,h,'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(correct=int(y[f==2][np.arange(len(hb)),hc].sum()),n=len(hb)))
        np.savez(OUT/f'{arm}_scores.npz',validation=s,held=h)
        for alpha in (.1,.25,.5,1.):
            c=apply_gate(v,(1-alpha)*bs+alpha*normalized(s),'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
            r=dict(arm=arm,alpha=alpha,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()),predictions=c.tolist());records.append(r);print({k:v for k,v in r.items() if k!='predictions'},flush=True)
    dump(OUT/'validation_records.json',records);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
