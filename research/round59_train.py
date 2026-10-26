from round22_common import ROOT,load,read,dump,sha
from round59_selector import features,forward
from round25_selector import scores as base_scores
from round22_selector import normalized
from round4_gates import apply_gate
import torch,numpy as np,time,copy,shutil
from torch import nn
OUT=ROOT/'experiments/round59'

class Model(nn.Module):
    def __init__(self,d,pair):
        super().__init__();self.enc=nn.Linear(d,64);self.solo=nn.Linear(64,32);self.out=nn.Linear(32,1)
        self.pair=nn.Linear(256,32);self.pout=nn.Linear(32,1);self.usepair=pair;self.drop=nn.Dropout(.15)
    def forward(self,x,o):
        h=self.drop(self.enc(x).relu());s=self.out(self.solo(h).relu()).squeeze(-1)
        if not self.usepair:return s
        a=h[:,:,None,:].expand(-1,4,4,-1);b=h[:,None,:,:].expand(-1,4,4,-1)
        z=torch.cat([a,b,a-b,a*b],-1);r=self.pout(self.pair(z).relu()).squeeze(-1)
        return s+((r-r.transpose(1,2))*o).sum(2)/o.sum(2).clamp_min(1)

def fit(x,o,y,ids,seed,pair,path):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);mean=x.mean((0,1));scale=x.std((0,1)).clip(.01)
    xx=torch.tensor(np.clip((x-mean)/scale,-8,8),device='cuda');oo=torch.tensor(o,device='cuda');yy=torch.tensor(y,device='cuda')
    model=Model(x.shape[-1],pair).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);ema=copy.deepcopy(model)
    for epoch in range(80):
        model.train()
        for ix in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/128))):
            s=model(xx[ix],oo[ix]);t=yy[ix]
            loss=(torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(t<.5,-1e9),1)).mean()+.25*nn.functional.binary_cross_entropy_with_logits(s,t)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,.02)
    ema.eval();w=dict(mean=mean,scale=scale)
    for name in ('enc','solo','out','pair','pout'):
        l=getattr(ema,name);w[name+'w']=l.weight.detach().cpu().numpy().T.copy();w[name+'b']=l.bias.detach().cpu().numpy().copy()
    with torch.no_grad():s=ema(xx[:8],oo[:8]).cpu().numpy()
    assert np.allclose(s,forward(x[:8],o[:8],w,pair),atol=2e-5)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**w);dump(path.with_suffix('.json'),dict(training_ids=ids.tolist(),seed=seed,epochs=80,loss=float(loss),pair=pair))
    return w

if __name__=='__main__':
    OUT.mkdir();torch.set_num_threads(4);start=time.perf_counter()
    dump(OUT/'plan.json',dict(arms=['independent','pairwise'],seeds=[59001,59002],epochs=80,alpha=[.25,.5,1.],test_opened=False,goal_test_correct=402,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round59*.py')}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');x,o=features(b,a);vx,vo=features(v,va)
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:fold=z['folds']
    bs=base_scores(v,va,ROOT/'experiments/round25/selected_bundle');base=apply_gate(v,bs,'parse_and_repair',4);records=[]
    for pair in (False,True):
        arm='pairwise' if pair else 'independent';ss=[];hs=[]
        for seed in (59001,59002):
            w=fit(x[fold!=2],o[fold!=2],y[fold!=2],ids[fold!=2],seed,pair,OUT/'diagnostic'/f'{arm}_{seed}.npz')
            hs.append(forward(x[fold==2],o[fold==2],w,pair))
            w=fit(x,o,y,ids,seed,pair,OUT/'full'/f'{arm}_{seed}.npz');ss.append(forward(vx,vo,w,pair))
            print('TRAINED',arm,seed,'seconds',time.perf_counter()-start,flush=True)
        h=np.mean(hs,0);hc=apply_gate([r for r,k in zip(b,fold) if k==2],h,'parse_and_repair',4)
        dump(OUT/f'{arm}_held.json',dict(correct=int(y[fold==2][np.arange(len(h)),hc].sum()),n=len(h)))
        s=normalized(np.mean(ss,0));np.savez(OUT/f'{arm}_scores.npz',validation=s,held=h)
        for alpha in (.25,.5,1.):
            c=apply_gate(v,alpha*s+(1-alpha)*normalized(bs),'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),base]
            rec=dict(arm=arm,alpha=alpha,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()),predictions=c.tolist())
            records.append(rec);print('VAL', {k:v for k,v in rec.items() if k!='predictions'},flush=True)
    dump(OUT/'validation_records.json',records)
    best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['alpha']))
    bundle=OUT/'bundle';bundle.mkdir()
    for seed in (59001,59002):shutil.copy2(OUT/'full'/f"{best['arm']}_{seed}.npz",bundle/f'model{seed}.npz')
    if best['alpha']<1:shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
    dump(bundle/'config.json',dict(pair=best['arm']=='pairwise',alpha=best['alpha'],models=[f'model{s}.npz' for s in (59001,59002)]))
    size=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert size<=16*1024**2
    dump(OUT/f"validation_{best['arm']}_predictions.json",dict(selected=best['predictions']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle='experiments/round59/bundle'),selected='round59' if best['correct']>752 else 'round25',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-start,asset_bytes=size,device='RTX A2000 Laptop CUDA',test_opened=False))
