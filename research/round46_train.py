"""Matched source-supervised SFT / exact reward / finite-action GRPO."""
from data_access import ROOT
from round22_common import load,read,dump,sha
from round46_graph import DIM,LIMIT
from round46_selector import forward,select,probability
from round45_rlvr import logprob,advantages,objective,checks
from scipy import sparse
from torch import nn
import torch,numpy as np,copy,time,json

OUT=ROOT/'experiments/round46'

def model():return nn.Sequential(nn.Linear(DIM,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
def weights(m):
    result={}
    for i,l in enumerate(x for x in m if isinstance(x,nn.Linear)):
        result[f'w{i}']=l.weight.detach().cpu().numpy().T.copy();result[f'b{i}']=l.bias.detach().cpu().numpy().copy()
    return result

def nll(s,r,mask):
    s=s.masked_fill(~mask,-1e9);available=r.gt(.5).any(1)
    loss=torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(r<.5,-1e9),1)
    return torch.where(available,loss,torch.zeros_like(loss)).mean()

def fit(m,x,rewards,mask,fitrows,pairrows,xp,xn,seed,device,arm,reference=None):
    rng=np.random.default_rng(seed+(100 if reference is not None else 0));gen=torch.Generator(device=device);gen.manual_seed(seed+123)
    opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);history=[]
    for epoch in range(20 if reference is None else 10):
        total={k:0. for k in ('loss','expected_reward','relation_loss','mixed_group_fraction')};count=0
        for ix in np.array_split(rng.permutation(fitrows),int(np.ceil(len(fitrows)/32))):
            xx=torch.as_tensor(x[(ix[:,None]*LIMIT+np.arange(LIMIT)).reshape(-1)].toarray(),device=device)
            rr=torch.as_tensor(rewards[ix],device=device);mm=torch.as_tensor(mask[ix],device=device)
            pick=rng.choice(pairrows,32,replace=len(pairrows)<32)
            px=torch.as_tensor(xp[pick].toarray(),device=device);nx=torch.as_tensor(xn[pick].toarray(),device=device)
            with torch.no_grad():
                old=logprob(m(xx).reshape(len(ix),LIMIT),mm)
                actions=torch.multinomial(old.exp(),16,replacement=True,generator=gen)
                sampled=rr.gather(1,actions);adv=advantages(sampled);oldlp=old.gather(1,actions);mixed=sampled.std(1,unbiased=False).gt(0).float().mean()
            for inner in range(1 if reference is None else 2):
                s=m(xx).reshape(len(ix),LIMIT);lp=logprob(s,mm)
                if reference is None:loss=nll(s,rr,mm)
                elif arm=='sft':
                    ref=torch.as_tensor(reference[ix],device=device);loss=nll(s,rr,mm)+.02*(lp.exp()*(lp-ref)).sum(1).mean()
                else:
                    loss,_=objective(s,mm,rr,torch.as_tensor(reference[ix],device=device),arm,actions,oldlp,adv)
                aux=nn.functional.softplus(m(nx)-m(px)).mean();loss=loss+.2*aux
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                for k,v in dict(loss=loss.detach(),expected_reward=(lp.exp()*rr).sum(1).mean().detach(),relation_loss=aux.detach(),mixed_group_fraction=mixed).items():total[k]+=float(v)*len(ix)
                count+=len(ix)
        history.append(dict(epoch=epoch+1,**{k:v/count for k,v in total.items()}))
    return m.eval(),history

def metrics(y,c,base,mask):
    ok=y[np.arange(len(y)),c].astype(bool);old=y[np.arange(len(y)),base].astype(bool)
    return dict(n=len(y),correct=int(ok.sum()),accuracy=100*float(ok.mean()),repairs=int((ok&~old).sum()),harms=int((~ok&old).sum()),
                conflict_n=int(mask.sum()),conflict_correct=int(ok[mask].sum()))

def main():
    plan=read(OUT/'plan.json')
    for p,h in plan['source_hashes'].items():assert sha(ROOT/p)==h,p
    for p,h in read(OUT/'prepared_freeze.json')['hashes'].items():assert sha(ROOT/p)==h,p
    dump(OUT/'objective_checks.json',checks());torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu';start=time.perf_counter()
    b,_,y,ids=load('train')
    with np.load(OUT/'train_targets.npz') as z:
        rewards=z['rewards'];rawmask=z['mask'];matches=z['matches'];folds=z['folds'];base=z['base'];bs=z['base_scores'];assert np.array_equal(ids,z['ids'])
    mask=rawmask.copy();mask[~mask.any(1),0]=True
    x=sparse.load_npz(OUT/'train_features.npz');xp=sparse.load_npz(OUT/'constraint_positive.npz');xn=sparse.load_npz(OUT/'constraint_negative.npz');parents=np.load(OUT/'constraint_parent_ids.npy')
    parentfold=dict(zip(map(int,ids),map(int,folds)));held=np.flatnonzero(folds==2)
    from features import parse_answer
    conflict=np.asarray([len({parse_answer(c['text']) for c in r['candidates']}-{None})>=2 for r in b])
    predictions={};summary=[]
    for stage in ('diagnostic','full'):
        fitrows=np.flatnonzero(folds!=2) if stage=='diagnostic' else np.arange(len(ids))
        pairrows=np.asarray([i for i,p in enumerate(parents) if stage=='full' or parentfold[int(p)]!=2])
        for seed in plan['seeds']:
            torch.manual_seed(seed);m=model().to(device)
            warm,h=fit(m,x,rewards,mask,fitrows,pairrows,xp,xn,seed,device,'sft')
            path=OUT/stage/f'warm_s{seed}';path.mkdir(parents=True)
            np.savez(path/'model.npz',**weights(warm));dump(path/'training.json',dict(history=h,training_n=len(fitrows),training_ids=ids[fitrows].tolist()))
            ref=logprob(torch.from_numpy(forward(x,weights(warm))),torch.from_numpy(mask)).numpy()
            print(dict(stage=stage,arm='warm',seed=seed,last=h[-1],seconds=round(time.perf_counter()-start,1)),flush=True)
            for arm in plan['arms']:
                m,h=fit(copy.deepcopy(warm),x,rewards,mask,fitrows,pairrows,xp,xn,seed,device,arm,ref)
                path=OUT/stage/f'{arm}_s{seed}';path.mkdir(parents=True);w=weights(m)
                np.savez(path/'model.npz',**w)
                dump(path/'training.json',dict(history=h,training_ids=ids[fitrows].tolist(),constraint_parents=sorted({int(parents[j]) for j in pairrows})))
                if stage=='diagnostic':
                    predictions[f'{arm}_{seed}']=forward(x,w)[held]
                    ph=np.asarray([j for j,p in enumerate(parents) if parentfold[int(p)]==2])
                    # Pair matrices have arbitrary row count; direct MLP scores rather than LIMIT reshape.
                    def flat(xx):return (np.maximum(np.maximum(xx@w['w0']+w['b0'],0)@w['w1']+w['b1'],0)@w['w2']+w['b2']).reshape(-1)
                    predictions[f'pair_{arm}_{seed}']=flat(xp[ph])-flat(xn[ph])
                print(dict(stage=stage,arm=arm,seed=seed,last=h[-1],seconds=round(time.perf_counter()-start,1)),flush=True)
        if stage=='diagnostic':
            for arm in plan['arms']:
                s=sum(predictions[f'{arm}_{seed}'] for seed in plan['seeds'])/len(plan['seeds'])
                for mode in ('execute_top','fixed_residual'):
                    c=select([b[i] for i in held],bs[held],s,rawmask[held],matches[held],mode)
                    rec=metrics(y[held],c,base[held],conflict[held]);rec.update(arm=arm,mode=mode)
                    rec['held_relation_pairs']=len(predictions[f'pair_{arm}_{plan["seeds"][0]}'])
                    rec['held_relation_correct']=int((sum(predictions[f'pair_{arm}_{seed}'] for seed in plan['seeds'])>0).sum())
                    summary.append(rec);print('HELD',rec,flush=True)
            dump(OUT/'diagnostic_result.json',summary);np.savez(OUT/'diagnostic_scores.npz',held=held,**predictions)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-start,device=device,torch=torch.__version__,training_n=len(ids),test_opened=False))

if __name__=='__main__':main()
