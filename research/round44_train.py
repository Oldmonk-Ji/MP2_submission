from data_access import ROOT
from round22_common import read,dump,sha,load
from round44_programs import LIMIT,DIM
from round44_policy import model,weights,predict,decisions,valid_mask,loss,forward
from round40_evaluate import metrics
from round4_gates import apply_gate
from scipy import sparse
import torch,numpy as np,copy,time,json

OUT=ROOT/'experiments/round44';ARMS=['sft','reward','reward_relations'];SEEDS=[44044,44045]

def run():
    plan=read(OUT/'plan.json')
    for p,h in plan['source_hashes'].items():assert sha(ROOT/p)==h
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';start=time.perf_counter()
    x=sparse.load_npz(OUT/'train_features.npz');xp=sparse.load_npz(OUT/'constraint_positive.npz');xn=sparse.load_npz(OUT/'constraint_negative.npz');parents=np.load(OUT/'constraint_parent_ids.npy')
    with np.load(OUT/'train_targets.npz') as z:
        rewards=z['rewards'];mask=z['mask'];matches=z['matches'];folds=z['folds'];ids=z['ids'];labels=z['labels']
    b,a,y,_=load('train');assert np.array_equal(y,labels);held=np.flatnonzero(folds==2);fitix=np.flatnonzero(folds!=2)
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:base=z['choices'];bs=z['scores']
    parentfold={int(i):int(f) for i,f in zip(ids,folds)};pairheld=np.asarray([j for j,i in enumerate(parents) if parentfold[int(i)]==2]);records=[];predictions={}
    dump(OUT/'diagnostic_partition.json',dict(fit_ids=ids[fitix].tolist(),held_ids=ids[held].tolist(),
        fit_constraint_parents=sorted({int(i) for i in parents if parentfold[int(i)]!=2}),held_constraint_parents=sorted({int(i) for i in parents if parentfold[int(i)]==2})))
    def train(m,fitrows,pairrows,arm,epochs,seed,reference=None):
        rng=np.random.default_rng(seed);torch.manual_seed(seed);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01)
        warming=reference is None
        if reference is None:reference=np.zeros_like(rewards)
        history=[]
        for ep in range(epochs):
            total=0.;auxsum=0.;m.train()
            for ix in np.array_split(rng.permutation(fitrows),int(np.ceil(len(fitrows)/64))):
                flat=(ix[:,None]*LIMIT+np.arange(LIMIT)[None,:]).reshape(-1)
                xx=torch.as_tensor(x[flat].toarray(),device=device);s=m(xx).reshape(-1,LIMIT)
                r=torch.as_tensor(rewards[ix],device=device);mm=torch.as_tensor(valid_mask(mask[ix]),device=device);ref=torch.as_tensor(reference[ix],device=device)
                objective=loss(s,r,mm,ref,arm,kl=0 if warming else .02)
                # Draw the same auxiliary indices in every arm; only its
                # predeclared coefficient differs, isolating relation supervision.
                pick=rng.choice(pairrows,16,replace=len(pairrows)<16) if len(pairrows) else []
                if arm=='reward_relations' and len(pick):
                    ps=m(torch.as_tensor(xp[pick].toarray(),device=device)).squeeze(-1);ns=m(torch.as_tensor(xn[pick].toarray(),device=device)).squeeze(-1)
                    aux=nn.functional.softplus(ns-ps).mean();objective=objective+.2*aux;auxsum+=float(aux.detach())*len(ix)
                opt.zero_grad();objective.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();total+=float(objective.detach())*len(ix)
            history.append(dict(epoch=ep+1,loss=total/len(fitrows),relation_loss=auxsum/len(fitrows)))
        return m.eval(),history
    from torch import nn
    for stage in ('diagnostic','full'):
        fitrows=fitix if stage=='diagnostic' else np.arange(len(y));pairrows=np.asarray([j for j,i in enumerate(parents) if stage=='full' or parentfold[int(i)]!=2])
        for seed in SEEDS:
            torch.manual_seed(seed);warm=model().to(device);warm,wh=train(warm,fitrows,pairrows,'sft',10,seed)
            ref=predict(warm,x)
            if stage=='diagnostic':predictions[f'warm_{seed}']=ref[held]
            for arm in ARMS:
                m,h=train(copy.deepcopy(warm),fitrows,pairrows,arm,20,seed+2026,ref)
                out=OUT/stage/f'{arm}_s{seed}';out.mkdir(parents=True);np.savez(out/'model.npz',**weights(m))
                dump(out/'config.json',dict(arm=arm,seed=seed,training_n=len(fitrows),warm_epochs=10,continuation_epochs=20,feature_dimensions=DIM,proposal_limit=LIMIT,auxiliary_weight=.2 if arm=='reward_relations' else 0.,kl=.02))
                dump(out/'training.json',dict(warm_history=wh,history=h,training_ids=ids[fitrows].tolist(),constraint_parents=sorted({int(i) for i in parents[pairrows]})))
                if stage=='diagnostic':
                    predictions[f'{arm}_{seed}']=predict(m,x)[held]
                    if len(pairheld):predictions[f'constraint_{arm}_{seed}']=forward(xp[pairheld],weights(m))-forward(xn[pairheld],weights(m))
                print(dict(stage=stage,arm=arm,seed=seed,loss=h[-1],seconds=round(time.perf_counter()-start,1)),flush=True)
        if stage=='diagnostic':
            for arm in ['warm']+ARMS:
                s=sum(predictions[f'{arm}_{seed}'] for seed in SEEDS)/len(SEEDS)
                hard,res,p=decisions(s,mask[held],matches[held],base[held],bs[held]);fused=apply_gate([b[k] for k in held],bs[held]+.25*res,'parse_and_repair',4)
                expected=(p*rewards[held]).sum(1)
                for name,c in [('execute_top',hard),('fixed_residual',fused)]:
                    r=metrics([b[k] for k in held],y[held],base[held],c,[dict(answer=None)]*len(held));r.update(arm=arm,mode=name,program_reward=float(expected.mean()),proposal_n=int(mask[held].any(1).sum()),oracle_correct_programs=int(rewards[held].any(1).sum()))
                    if arm!='warm' and len(pairheld):
                        margin=sum(predictions[f'constraint_{arm}_{seed}'] for seed in SEEDS)/len(SEEDS);r['held_relation_pairs']=len(margin);r['held_relation_pairs_correct']=int((margin>0).sum())
                    records.append(r);print('HELD',r,flush=True)
            dump(OUT/'diagnostic_result.json',records);np.savez(OUT/'diagnostic_scores.npz',held=held,**predictions)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-start,device=device,test_opened=False))

if __name__=='__main__':run()
