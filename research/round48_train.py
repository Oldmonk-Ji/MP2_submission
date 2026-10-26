"""Matched SFT/RLVR with directly supervised local relation decisions."""
from data_access import ROOT
from round22_common import read,dump,sha,load
from round48_local import DIM,LIMIT,node_scores,aggregate
from round46_selector import select
from round46_train import nll,metrics
from round45_rlvr import logprob,advantages,objective,checks
from scipy import sparse
from torch import nn
import torch,numpy as np,copy,time

OUT=ROOT/'experiments/round48'

class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(DIM,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,packet):
        x,groups,counts=packet
        ns=self.net(x).flatten().clamp(-20,20)
        sums=torch.zeros(len(counts),device=x.device).index_add_(0,groups,(-ns).exp())
        scores=-(sums/counts.clamp_min(1)).clamp_min(1e-20).log()
        return torch.where(counts>0,scores,torch.zeros_like(scores)).reshape(-1,LIMIT)

def weights(m):
    result={}
    for i,l in enumerate(x for x in m.net if isinstance(x,nn.Linear)):
        result[f'w{i}']=l.weight.detach().cpu().numpy().T.copy();result[f'b{i}']=l.bias.detach().cpu().numpy().copy()
    return result

def packet(x,ptr,ix,device):
    programs=(ix[:,None]*LIMIT+np.arange(LIMIT)).reshape(-1);starts=ptr[programs];counts=ptr[programs+1]-starts
    groups=np.repeat(np.arange(len(programs)),counts)
    rows=np.repeat(starts,counts)+np.arange(counts.sum())-np.repeat(np.cumsum(counts)-counts,counts)
    return torch.as_tensor(x[rows].toarray(),device=device),torch.as_tensor(groups,device=device),torch.as_tensor(counts,device=device)

def fit(m,x,ptr,rewards,mask,fitrows,pairrows,xp,xn,seed,device,arm,reference=None):
    rng=np.random.default_rng(seed+(100 if reference is not None else 0));gen=torch.Generator(device=device);gen.manual_seed(seed+123)
    opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);history=[]
    for epoch in range(20 if reference is None else 10):
        total={k:0. for k in ('loss','expected_reward','relation_loss','mixed_group_fraction')};count=0
        for ix in np.array_split(rng.permutation(fitrows),int(np.ceil(len(fitrows)/32))):
            xx=packet(x,ptr,ix,device);rr=torch.as_tensor(rewards[ix],device=device);mm=torch.as_tensor(mask[ix],device=device)
            pick=rng.choice(pairrows,32,replace=len(pairrows)<32);px=torch.as_tensor(xp[pick].toarray(),device=device);nx=torch.as_tensor(xn[pick].toarray(),device=device)
            with torch.no_grad():
                old=logprob(m(xx),mm);actions=torch.multinomial(old.exp(),16,replacement=True,generator=gen)
                sampled=rr.gather(1,actions);adv=advantages(sampled);oldlp=old.gather(1,actions);mixed=sampled.std(1,unbiased=False).gt(0).float().mean()
            for inner in range(1 if reference is None else 2):
                s=m(xx);lp=logprob(s,mm)
                if reference is None:loss=nll(s,rr,mm)
                elif arm=='sft':
                    ref=torch.as_tensor(reference[ix],device=device);loss=nll(s,rr,mm)+.02*(lp.exp()*(lp-ref)).sum(1).mean()
                else:loss,_=objective(s,mm,rr,torch.as_tensor(reference[ix],device=device),arm,actions,oldlp,adv)
                aux=nn.functional.softplus(m.net(nx)-m.net(px)).mean();loss=loss+.2*aux
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                for k,v in dict(loss=loss.detach(),expected_reward=(lp.exp()*rr).sum(1).mean().detach(),relation_loss=aux.detach(),mixed_group_fraction=mixed).items():total[k]+=float(v)*len(ix)
                count+=len(ix)
        history.append(dict(epoch=epoch+1,**{k:v/count for k,v in total.items()}))
    return m.eval(),history

def main():
    plan=read(OUT/'plan.json')
    for p,h in plan['source_hashes'].items():assert sha(ROOT/p)==h,p
    for p,h in read(OUT/'prepared_freeze.json')['hashes'].items():assert sha(ROOT/p)==h,p
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';start=time.perf_counter();b,_,y,ids=load('train')
    with np.load(OUT/'train_targets.npz') as z:
        rewards=z['rewards'];raw=z['mask'];matches=z['matches'];folds=z['folds'];base=z['base'];bs=z['base_scores'];assert np.array_equal(ids,z['ids'])
    mask=raw.copy();mask[~mask.any(1),0]=True
    x=sparse.load_npz(OUT/'node_features.npz');ptr=np.load(OUT/'program_node_ptr.npy')
    xp=sparse.load_npz(OUT/'constraint_positive.npz');xn=sparse.load_npz(OUT/'constraint_negative.npz');parents=np.load(OUT/'constraint_parent_ids.npy')
    parentfold=dict(zip(map(int,ids),map(int,folds)));held=np.flatnonzero(folds==2)
    from features import parse_answer
    conflict=np.asarray([len({parse_answer(c['text']) for c in r['candidates']}-{None})>=2 for r in b]);predictions={};records=[]
    # Validate the independent NumPy executor for the compositional scorer.
    torch.manual_seed(48048);probe=Policy().to(device);take=np.asarray([0,1,2,3]);xx=packet(x,ptr,take,device)
    with torch.no_grad():ts=probe(xx).cpu().numpy()
    ns=aggregate(node_scores(x,weights(probe)),ptr)[take];assert np.allclose(ts,ns,atol=1e-5)
    dump(OUT/'objective_checks.json',dict(**checks(),local_numpy_torch_agree=True))
    for stage in ('diagnostic','full'):
        fitrows=np.flatnonzero(folds!=2) if stage=='diagnostic' else np.arange(len(ids))
        pairrows=np.asarray([i for i,p in enumerate(parents) if stage=='full' or parentfold[int(p)]!=2])
        for seed in plan['seeds']:
            torch.manual_seed(seed);warm,h=fit(Policy().to(device),x,ptr,rewards,mask,fitrows,pairrows,xp,xn,seed,device,'sft')
            path=OUT/stage/f'warm_s{seed}';path.mkdir(parents=True);np.savez(path/'model.npz',**weights(warm));dump(path/'training.json',dict(history=h,training_ids=ids[fitrows].tolist()))
            ref=logprob(torch.from_numpy(aggregate(node_scores(x,weights(warm)),ptr)),torch.from_numpy(mask)).numpy()
            print(dict(stage=stage,arm='warm',seed=seed,last=h[-1],seconds=time.perf_counter()-start),flush=True)
            for arm in plan['arms']:
                m,h=fit(copy.deepcopy(warm),x,ptr,rewards,mask,fitrows,pairrows,xp,xn,seed,device,arm,ref)
                path=OUT/stage/f'{arm}_s{seed}';path.mkdir(parents=True);w=weights(m);np.savez(path/'model.npz',**w)
                dump(path/'training.json',dict(history=h,training_ids=ids[fitrows].tolist(),constraint_parents=sorted({int(parents[j]) for j in pairrows})))
                if stage=='diagnostic':predictions[f'{arm}_{seed}']=aggregate(node_scores(x,w),ptr)[held]
                print(dict(stage=stage,arm=arm,seed=seed,last=h[-1],seconds=time.perf_counter()-start),flush=True)
        if stage=='diagnostic':
            for arm in plan['arms']:
                s=sum(predictions[f'{arm}_{seed}'] for seed in plan['seeds'])/2
                for mode in ('execute_top','fixed_residual'):
                    c=select([b[i] for i in held],bs[held],s,raw[held],matches[held],mode)
                    rec=metrics(y[held],c,base[held],conflict[held]);rec.update(arm=arm,mode=mode);records.append(rec);print('LOCAL_HELD',rec,flush=True)
            dump(OUT/'diagnostic_result.json',records);np.savez(OUT/'diagnostic_scores.npz',held=held,**predictions)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-start,device=device,torch=torch.__version__,test_opened=False))

if __name__=='__main__':main()
