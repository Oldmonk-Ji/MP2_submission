"""TRAIN-only matched ablation of process supervision, not another VAL sweep.

Same graph/proposals, seeds, shuffled rows, optimizer budget and unused sampled
actions as round46_train.fit; only the relation-gradient coefficient is zero.
No additional held parent annotations enter gradients.
"""
from data_access import ROOT
from round22_common import read,dump,sha,load
from round46_train import model,weights,nll,metrics
from round46_selector import forward,select
from round46_graph import LIMIT
from round45_rlvr import logprob
from scipy import sparse
from torch import nn
import torch,numpy as np,time,datetime

OUT=ROOT/'experiments/round47'

def fit(m,x,rewards,mask,fitrows,pairrows,xp,xn,seed,device,reference=None):
    rng=np.random.default_rng(seed+(100 if reference is not None else 0));gen=torch.Generator(device=device);gen.manual_seed(seed+123)
    opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);history=[]
    for epoch in range(20 if reference is None else 10):
        total_loss=0.;count=0
        for ix in np.array_split(rng.permutation(fitrows),int(np.ceil(len(fitrows)/32))):
            xx=torch.as_tensor(x[(ix[:,None]*LIMIT+np.arange(LIMIT)).reshape(-1)].toarray(),device=device)
            rr=torch.as_tensor(rewards[ix],device=device);mm=torch.as_tensor(mask[ix],device=device)
            pick=rng.choice(pairrows,32,replace=len(pairrows)<32)
            px=torch.as_tensor(xp[pick].toarray(),device=device);nx=torch.as_tensor(xn[pick].toarray(),device=device)
            with torch.no_grad():
                old=logprob(m(xx).reshape(len(ix),LIMIT),mm)
                torch.multinomial(old.exp(),16,replacement=True,generator=gen)
            for inner in range(1 if reference is None else 2):
                s=m(xx).reshape(len(ix),LIMIT);lp=logprob(s,mm);loss=nll(s,rr,mm)
                if reference is not None:
                    ref=torch.as_tensor(reference[ix],device=device);loss=loss+.02*(lp.exp()*(lp-ref)).sum(1).mean()
                aux=nn.functional.softplus(m(nx)-m(px)).mean();loss=loss+0.*aux
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                total_loss+=float(loss.detach())*len(ix);count+=len(ix)
        history.append(dict(epoch=epoch+1,loss=total_loss/count))
    return m.eval(),history

def main():
    dest=OUT/'no_process_supervision';dest.mkdir()
    dump(dest/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),
        comparison='Only process auxiliary coefficient .2 -> 0; unchanged graph, source programs never inserted as actions, matched seeds/batches/steps, fixed held fold2.',
        validation_used=False,test_opened=False,selection='Diagnostic only; not an additional official-validation candidate.'))
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';start=time.perf_counter()
    with np.load(OUT/'train_targets.npz') as z:
        ids=z['ids'];folds=z['folds'];r=z['rewards'];mask=z['mask'];matches=z['matches'];base=z['base'];bs=z['base_scores'];y=z['labels']
    raw=mask.copy();mask[~mask.any(1),0]=True;held=np.flatnonzero(folds==2);fitrows=np.flatnonzero(folds!=2)
    x=sparse.load_npz(OUT/'train_features.npz');xp=sparse.load_npz(OUT/'constraint_positive.npz');xn=sparse.load_npz(OUT/'constraint_negative.npz')
    parents=np.load(OUT/'constraint_parent_ids.npy');fitids=set(ids[fitrows]);pairrows=np.asarray([i for i,p in enumerate(parents) if p in fitids]);scores=[]
    for seed in (46046,46047):
        torch.manual_seed(seed);m=model().to(device);warm,h1=fit(m,x,r,mask,fitrows,pairrows,xp,xn,seed,device)
        ref=logprob(torch.from_numpy(forward(x,weights(warm))),torch.from_numpy(mask)).numpy()
        m,h2=fit(warm,x,r,mask,fitrows,pairrows,xp,xn,seed,device,ref);w=weights(m)
        np.savez(dest/f'model_s{seed}.npz',**w);dump(dest/f'training_s{seed}.json',dict(warm=h1,continuation=h2,training_ids=ids[fitrows].tolist(),process_supervision_used=False))
        scores.append(forward(x,w)[held]);print(dict(noaux_seed=seed,seconds=time.perf_counter()-start),flush=True)
    b,_,_,_=load('train');b=[b[i] for i in held]
    from features import parse_answer
    conflict=np.asarray([len({parse_answer(c['text']) for c in row['candidates']}-{None})>=2 for row in b]);records=[]
    for mode in ('execute_top','fixed_residual'):
        c=select(b,bs[held],np.mean(scores,axis=0),raw[held],matches[held],mode)
        rec=metrics(y[held],c,base[held],conflict);rec['mode']=mode;records.append(rec)
    np.savez(dest/'held_scores.npz',held=held,scores=np.mean(scores,axis=0))
    dump(dest/'result.json',dict(records=records,seconds=time.perf_counter()-start,test_opened=False,official_validation_used=False))
    print('NO_PROCESS_SUPERVISION',records,flush=True)

if __name__=='__main__':main()
