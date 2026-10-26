"""Paper-inspired, parent-balanced hard relation contrasts; not a paper replica."""
from data_access import ROOT
from round22_common import read,dump,sha
import round48_train as training
from round50_selector import select
from round46_train import nll
from round45_rlvr import logprob
from torch import nn
import torch,numpy as np,shutil,datetime

OUT=ROOT/'experiments/round55';SOURCE=ROOT/'experiments/round52'
PARENTS=None

def fit(m,x,ptr,rewards,mask,fitrows,pairrows,xp,xn,seed,device,arm,reference=None):
    stage='full' if len(fitrows)==4743 else 'diagnostic'
    if reference is None:
        p=SOURCE/stage/f'warm_s{seed}'
        with np.load(p/'model.npz') as w:
            for i,l in enumerate(v for v in m.net if isinstance(v,nn.Linear)):
                with torch.no_grad():l.weight.copy_(torch.from_numpy(w[f'w{i}'].T.copy()).to(device));l.bias.copy_(torch.from_numpy(w[f'b{i}'].copy()).to(device))
        return m.eval(),read(p/'training.json')['history']
    rng=np.random.default_rng(seed+100);hard_rng=np.random.default_rng(seed+55000)
    parents=np.unique(PARENTS[pairrows]);groups=[pairrows[PARENTS[pairrows]==p] for p in parents]
    opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);history=[];selected=np.zeros(len(PARENTS),np.int64)
    pxall=torch.as_tensor(xp[pairrows].toarray(),device=device);nxall=torch.as_tensor(xn[pairrows].toarray(),device=device)
    index={int(p):i for i,p in enumerate(pairrows)}
    for epoch in range(10):
        with torch.no_grad():margins=(m.net(pxall)-m.net(nxall)).flatten().cpu().numpy()
        hardest=np.asarray([min(group,key=lambda j:(margins[index[int(j)]],int(j))) for group in groups])
        total={k:0. for k in ('loss','expected_reward','relation_loss')};count=0
        for ix in np.array_split(rng.permutation(fitrows),int(np.ceil(len(fitrows)/32))):
            xx=training.packet(x,ptr,ix,device);rr=torch.as_tensor(rewards[ix],device=device);mm=torch.as_tensor(mask[ix],device=device)
            # Same ordinary RNG calls/order as control; independent RNG for mining.
            pick=rng.choice(pairrows,32,replace=len(pairrows)<32)
            pick[:16]=hard_rng.choice(hardest,16,replace=len(hardest)<16)
            assert set(pick)<=set(pairrows);np.add.at(selected,pick,1)
            px=torch.as_tensor(xp[pick].toarray(),device=device);nx=torch.as_tensor(xn[pick].toarray(),device=device)
            ref=torch.as_tensor(reference[ix],device=device)
            for _ in range(2):
                s=m(xx);lp=logprob(s,mm);aux=nn.functional.softplus(m.net(nx)-m.net(px)).mean()
                loss=nll(s,rr,mm)+.02*(lp.exp()*(lp-ref)).sum(1).mean()+.2*aux
                assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                for k,v in dict(loss=loss,expected_reward=(lp.exp()*rr).sum(1).mean(),relation_loss=aux).items():total[k]+=float(v.detach())*len(ix)
                count+=len(ix)
        history.append(dict(epoch=epoch+1,**{k:v/count for k,v in total.items()}))
    dump(OUT/f'{stage}_sampling_s{seed}.json',dict(eligible_parents=parents.tolist(),eligible_pairs=pairrows.tolist(),pair_draws=selected.tolist(),
        half_uniform_pairs_half_parent_balanced_hard=True,held_process_pairs_never_mined=stage=='diagnostic',test_opened=False))
    return m.eval(),history

if __name__=='__main__':
    OUT.mkdir()
    for name in ('train_targets.npz','node_features.npz','program_node_ptr.npy','constraint_positive.npz','constraint_negative.npz','constraint_pairs.json','constraint_parent_ids.npy','held_source_review_before_reference.json','held_execution_checks.json'):
        shutil.copy2(SOURCE/name,OUT/name)
    PARENTS=np.load(OUT/'constraint_parent_ids.npy')
    plan=read(SOURCE/'plan.json');plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),arms=['sft'],
        objective='Same model/programs/labels/budget as round52 SFT, half uniform and half parent-balanced hardest known relation pairs.',
        paper='https://aclanthology.org/2022.findings-emnlp.316/',adaptation_not_replication=True,
        no_external_data_or_weights=True,control='round52 SFT frozen warm20 plus continuation10',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research/round55_hard_relations.py',ROOT/'research/round55_evaluate.py')})
    dump(OUT/'plan.json',plan)
    warmfiles=[SOURCE/stage/f'warm_s{seed}'/'model.npz' for stage in ('diagnostic','full') for seed in (48048,48049)]
    dump(OUT/'warm_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in warmfiles}))
    dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()}))
    training.OUT=OUT;training.DIM=2048;training.fit=fit;training.select=select;training.main()
