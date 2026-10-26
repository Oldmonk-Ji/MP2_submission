"""Finite-action, on-policy GRPO with deterministic TRAIN rewards.

The action is a candidate or a complete executable relation hypothesis, not
a generated language sequence. Exact categorical KL replaces sampled KL.
"""
from data_access import ROOT
from round22_common import load,read,dump,sha
from round22_selector import dense_forward,normalized
from round43_dpo import network,restore,fit,weights as candidate_weights
from round44_policy import model as relation_model,weights as relation_weights,forward,valid_mask
from round44_selector import select
from round3_features import build
from round4_gates import apply_gate
from round40_evaluate import metrics
from round14_prepare import family
from scipy import sparse
from torch import nn
import torch,numpy as np,datetime,copy,time,sys

OUT=ROOT/'experiments/round45'
ARMS={'candidate':['exact','grpo'],'relation':['exact','grpo','grpo_relations']}
SEEDS={'candidate':[25025,25026],'relation':[44044,44045]}

def logprob(scores,mask):return scores.masked_fill(~mask,-1e9).log_softmax(1)
def advantages(rewards):return (rewards-rewards.mean(1,keepdim=True))/rewards.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)
def clipped_surrogate(ratio,adv):return torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)

def objective(s,mask,rewards,ref,arm,actions,old_logp,adv):
    lp=logprob(s,mask);p=lp.exp();kl=(p*(lp-ref)).sum(1).mean()
    if arm=='exact':policy=-(p*rewards).sum(1).mean();clipfrac=s.new_tensor(0.)
    else:
        ratio=(lp.gather(1,actions)-old_logp).exp()
        policy=-clipped_surrogate(ratio,adv).mean();clipfrac=((ratio<.8)|(ratio>1.2)).float().mean()
    return policy+.02*kl,dict(kl=kl.detach(),clip_fraction=clipfrac.detach(),expected_reward=(p*rewards).sum(1).mean().detach())

def checks():
    s=torch.zeros((1,2),requires_grad=True);m=torch.ones_like(s,dtype=torch.bool);r=torch.tensor([[1.,0.]])
    act=torch.tensor([[0,1]]);old=logprob(s.detach(),m).gather(1,act);a=advantages(r)
    loss,_=objective(s,m,r,logprob(s.detach(),m),'grpo',act,old,a);loss.backward()
    assert s.grad[0,0]<0<s.grad[0,1]
    assert torch.equal(advantages(torch.ones((2,16))),torch.zeros((2,16)))
    assert torch.equal(advantages(torch.zeros((2,16))),torch.zeros((2,16)))
    ratio=torch.tensor([1.5,.5],requires_grad=True);val=clipped_surrogate(ratio,torch.tensor([1.,-1.]));val.sum().backward()
    assert torch.allclose(val,torch.tensor([1.2,-.8])) and torch.equal(ratio.grad,torch.zeros(2))
    mask=torch.tensor([[True,False,True]]);lp=logprob(torch.zeros((1,3)),mask)
    sampled=torch.multinomial(lp.exp(),1000,replacement=True);assert (sampled!=1).all()
    ref=logprob(torch.tensor([[.2,-.1]]),m).detach();z=torch.tensor([[.4,-.3]],requires_grad=True)
    loss,_=objective(z,m,r,ref,'exact',act,old,a);loss.backward();assert z.grad[0,0]<0<z.grad[0,1]
    # Refreshing behavior policy must change rollout probabilities, independently of frozen reference.
    p0=logprob(torch.tensor([[0.,0.]]),m).exp();p1=logprob(torch.tensor([[2.,0.]]),m).exp();assert not torch.allclose(p0,p1)
    return dict(preference_gradient=True,uniform_group_zero_advantage=True,positive_negative_clip_saturation=True,
        invalid_actions_never_sampled=True,exact_reward_gradient=True,behavior_distinct_from_reference=True)

def prepare():
    OUT.mkdir();check=checks();dump(OUT/'objective_checks.json',check)
    frozen={}
    for manifest in ['experiments/round43_44_completion_manifest.json','experiments/round38_42_completion_manifest.json']:
        for p,h in read(ROOT/manifest)['hashes'].items():assert sha(ROOT/p)==h,p
    inputs=[ROOT/'research/round45_rlvr.py',ROOT/'research/round45_selector.py',ROOT/'research/round45_evaluate.py']
    inputs += [ROOT/'experiments/round44'/f for f in ['train_features.npz','train_targets.npz','constraint_positive.npz','constraint_negative.npz','constraint_parent_ids.npy','constraint_pairs.json']]
    for stage in ('diagnostic','full'):
        inputs += [ROOT/f'experiments/round44/{stage}/sft_s{seed}/model.npz' for seed in SEEDS['relation']]
    inputs += [ROOT/f'experiments/round25/models/ground_uniform_positive_s{seed}_e35/model.npz' for seed in SEEDS['candidate']]
    frozen={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),arms=ARMS,seeds=SEEDS,
        source_and_input_hashes=frozen,epochs=10,updates_per_rollout=2,group_size=16,clip=.2,kl=.02,batch_size=64,
        learning_rate=dict(candidate=.0001,relation=.0003),weight_decay=.01,relation_weight=.2,
        scope='Finite one-step categorical GRPO, not token-autoregressive LLM RL. Current behavior refreshes each batch. Exact categorical KL, frozen SFT reference.',
        candidate_warm='Full: round25 ground epoch35. Diagnostic: same35epoch SFT/EMA fit only folds0+1, saved beforeRL.',
        relation_warm='Frozen round44 SFT30epoch, diagnostic/full corresponding partition. All arms share each seed warm.',
        training='All4743 final; 3161 fit/1582 grouped held diagnostics. Parent-bound relation supervision. No further exclusions.',
        all_equal_rewards='Zero policy advantage, no fabricated negatives or dynamic filtering; KL still applied.',
        deterministic_reward='Candidate: official parsed TRAIN answer equality. Relation: exact Fraction program execution equals TRAIN answer and maps to an original candidate.',
        dropout='Disabled for both exact and GRPO continuation, so importance ratios measure policy change, not dropout noise.',
        matched_controls='Same init, minibatches, learning rates and optimizer steps within level; exact arm still draws rollouts for matched diagnostics.',
        deployment='Candidate replaces ground members4,5 at fixed.125; relation top-execute or fixed.25 residual, frozen round44 algorithms.',
        selection='Full798validation: correct, fewer harms, prefercontrolties; asset cap16MiB; referenceCPU timing requires separate certification.',
        no_hyperparameter_sweep=True,test_opened=False,method_source='https://arxiv.org/abs/2402.03300',
        rules_source='https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md'))
    print('FROZEN',check,flush=True)

def load_relation(path):
    m=relation_model()
    with np.load(path/'model.npz') as w:
        for i,l in enumerate(v for v in m if isinstance(v,nn.Linear)):
            with torch.no_grad():l.weight.copy_(torch.from_numpy(w[f'w{i}'].T.copy()));l.bias.copy_(torch.from_numpy(w[f'b{i}'].copy()))
    return m

def train(m,getx,rewards,mask,reference,fitrows,pairrows,xp,xn,arm,level,seed,device):
    m.eval();rng=np.random.default_rng(seed+45045);gen=torch.Generator(device=device);gen.manual_seed(seed+45045)
    opt=torch.optim.AdamW(m.parameters(),lr=.0001 if level=='candidate' else .0003,weight_decay=.01);history=[]
    for epoch in range(10):
        total={k:0. for k in ('loss','kl','clip_fraction','expected_reward','relation_loss','mixed_group_fraction','sample_reward')};count=0
        for ix in np.array_split(rng.permutation(fitrows),int(np.ceil(len(fitrows)/64))):
            xx=getx(ix);rr=torch.as_tensor(rewards[ix],device=device);mm=torch.as_tensor(mask[ix],device=device);ref=torch.as_tensor(reference[ix],device=device)
            with torch.no_grad():
                old=logprob(m(xx).reshape(len(ix),-1),mm);actions=torch.multinomial(old.exp(),16,replacement=True,generator=gen)
                sampled=rr.gather(1,actions);adv=advantages(sampled);oldlp=old.gather(1,actions).detach()
                mixed=sampled.std(1,unbiased=False).gt(0).float().mean()
            pick=rng.choice(pairrows,16,replace=len(pairrows)<16) if len(pairrows) else []
            for inner in range(2):
                scores=m(xx).reshape(len(ix),-1);loss,stats=objective(scores,mm,rr,ref,arm,actions,oldlp,adv)
                aux=scores.new_tensor(0.)
                if arm=='grpo_relations' and len(pick):
                    ps=m(torch.as_tensor(xp[pick].toarray(),device=device));ns=m(torch.as_tensor(xn[pick].toarray(),device=device))
                    aux=nn.functional.softplus(ns-ps).mean();loss=loss+.2*aux
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                stats.update(loss=loss.detach(),relation_loss=aux.detach(),mixed_group_fraction=mixed,sample_reward=sampled.mean())
                for k,v in stats.items():total[k]+=float(v)*len(ix)
                count+=len(ix)
        history.append(dict(epoch=epoch+1,**{k:v/count for k,v in total.items()}))
    return m.eval(),history

def run():
    plan=read(OUT/'plan.json')
    for p,h in plan['source_and_input_hashes'].items():assert sha(ROOT/p)==h,p
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';start=time.perf_counter()
    b,a,y,ids=load('train');ground=build(b,a,'ground');dump(OUT/'training_ids.json',ids.tolist())
    with np.load(ROOT/'experiments/round44/train_targets.npz') as z:
        rewards=z['rewards'];mask=valid_mask(z['mask']);rawmask=z['mask'];matches=z['matches'];folds=z['folds'];assert np.array_equal(z['ids'],ids)
    x=sparse.load_npz(ROOT/'experiments/round44/train_features.npz');xp=sparse.load_npz(ROOT/'experiments/round44/constraint_positive.npz');xn=sparse.load_npz(ROOT/'experiments/round44/constraint_negative.npz');parents=np.load(ROOT/'experiments/round44/constraint_parent_ids.npy')
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:base=z['choices'];bs=z['scores']
    held=np.flatnonzero(folds==2);fitix=np.flatnonzero(folds!=2);parentfold={int(i):int(f) for i,f in zip(ids,folds)}
    assert not {family(b[i]['question']) for i in fitix}&{family(b[i]['question']) for i in held}
    dump(OUT/'diagnostic_partition.json',dict(fit_ids=ids[fitix].tolist(),held_ids=ids[held].tolist(),fit_constraint_parents=sorted({int(p) for p in parents if parentfold[int(p)]!=2})))
    predictions={};records=[]
    for stage in ('diagnostic','full'):
        fitrows=fitix if stage=='diagnostic' else np.arange(len(y))
        for level in ('candidate','relation'):
            pairrows=np.asarray([j for j,p in enumerate(parents) if stage=='full' or parentfold[int(p)]!=2]) if level=='relation' else []
            for seed in SEEDS[level]:
                if level=='candidate':
                    if stage=='full':warm,mean,scale=restore(ROOT/f'experiments/round25/models/ground_uniform_positive_s{seed}_e35');warm.to(device)
                    else:
                        mean=ground[fitrows].mean((0,1));scale=ground[fitrows].std((0,1)).clip(.001);torch.manual_seed(seed);warm=network(ground.shape[-1]).to(device)
                        warm,_=fit(warm,torch.as_tensor((ground[fitrows]-mean)/scale,device=device),torch.as_tensor(y[fitrows],device=device),'sft',35,seed,.0003,.1,True)
                    features=torch.as_tensor((ground-mean)/scale,device=device);getx=lambda ix:features[ix]
                    saveweights=lambda m:candidate_weights(m,mean,scale)
                    with torch.no_grad():reference=logprob(warm.eval()(features).squeeze(-1),torch.ones(y.shape,dtype=torch.bool,device=device)).cpu().numpy()
                    target=y;valid=np.ones_like(y,dtype=bool)
                else:
                    warm=load_relation(ROOT/f'experiments/round44/{stage}/sft_s{seed}').to(device).eval()
                    getx=lambda ix:torch.as_tensor(x[(ix[:,None]*32+np.arange(32)).reshape(-1)].toarray(),device=device)
                    saveweights=relation_weights;rs=forward(x,saveweights(warm)).reshape(-1,32)
                    reference=logprob(torch.from_numpy(rs),torch.from_numpy(mask)).numpy();target=rewards;valid=mask
                warmout=OUT/stage/f'{level}_warm_s{seed}';warmout.mkdir(parents=True);np.savez(warmout/'model.npz',**saveweights(warm))
                if stage=='diagnostic':predictions[f'{level}_warm_{seed}']=dense_forward(ground[held],saveweights(warm)) if level=='candidate' else forward(x,saveweights(warm)).reshape(-1,32)[held]
                for arm in ARMS[level]:
                    m,h=train(copy.deepcopy(warm),getx,target,valid,reference,fitrows,pairrows,xp,xn,arm,level,seed,device)
                    out=OUT/stage/f'{level}_{arm}_s{seed}';out.mkdir(parents=True);w=saveweights(m);np.savez(out/'model.npz',**w)
                    dump(out/'config.json',dict(family='mlp',features='ground' if level=='candidate' else 'relation',level=level,arm=arm,seed=seed,training_n=len(fitrows),epochs=10,updates_per_rollout=2))
                    dump(out/'training.json',dict(history=h,training_ids=ids[fitrows].tolist(),constraint_parents=sorted({int(parents[j]) for j in pairrows})))
                    if stage=='diagnostic':
                        predictions[f'{level}_{arm}_{seed}']=dense_forward(ground[held],w) if level=='candidate' else forward(x,w).reshape(-1,32)[held]
                        if level=='relation':
                            ph=np.asarray([j for j,p in enumerate(parents) if parentfold[int(p)]==2]);predictions[f'constraint_{arm}_{seed}']=forward(xp[ph],w)-forward(xn[ph],w)
                    print(dict(stage=stage,level=level,arm=arm,seed=seed,last=h[-1],seconds=round(time.perf_counter()-start,1)),flush=True)
        if stage=='diagnostic':
            for level in ('candidate','relation'):
                for arm in ['warm']+ARMS[level]:
                    ss=[predictions[f'{level}_{arm}_{seed}'] for seed in SEEDS[level]];s=sum(normalized(v) if level=='candidate' else v for v in ss)/2
                    modes=['fixed_members'] if level=='candidate' else ['execute_top','fixed_residual']
                    for mode in modes:
                        c=apply_gate([b[k] for k in held],bs[held]+.25*s,'parse_and_repair',4) if level=='candidate' else select([b[k] for k in held],bs[held],s,rawmask[held],matches[held],mode)
                        r=metrics([b[k] for k in held],y[held],base[held],c,[dict(answer=None)]*len(held));r.update(level=level,arm=arm,mode=mode)
                        if level=='relation' and arm!='warm':
                            margins=sum(predictions[f'constraint_{arm}_{seed}'] for seed in SEEDS[level])/2;r.update(held_relation_pairs=len(margins),held_relation_correct=int((margins>0).sum()))
                        records.append(r);print('HELD',r,flush=True)
            dump(OUT/'diagnostic_result.json',records);np.savez(OUT/'diagnostic_scores.npz',held=held,**predictions)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-start,device=device,test_opened=False))

if __name__=='__main__':globals()[sys.argv[1]]()
