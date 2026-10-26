"""Matched fixed-candidate DPO / pairwise / supervised continuation.

Reference policy is a frozen copy of the common warm start, never a teacher
with access to a held question. Positive pairs follow official final numbers.
"""
from data_access import ROOT
from round22_common import load,read,dump,sha
from round22_selector import normalized,dense_forward
from round3_features import build
from round4_gates import apply_gate
from round14_prepare import family
import numpy as np,torch,copy,time,datetime,sys
from torch import nn

OUT=ROOT/'experiments/round43'
ARMS=['sft','pairwise','dpo'];SEEDS=[25025,25026]

def network(d):return nn.Sequential(nn.Linear(d,96),nn.ReLU(),nn.Dropout(.2),nn.Linear(96,32),nn.ReLU(),nn.Linear(32,1))
def supervised(s,y):return -(s.log_softmax(1)*y/y.sum(1,keepdim=True)).sum(1)+.25*nn.functional.binary_cross_entropy_with_logits(s,y,reduction='none').mean(1)
def objective(s,y,arm,ref=None):
    if arm=='sft':return supervised(s,y).mean()
    mask=y[:,:,None].gt(.5)&y[:,None,:].lt(.5);delta=s[:,:,None]-s[:,None,:]
    if arm=='dpo':delta=delta-(ref[:,:,None]-ref[:,None,:]) # beta fixed at 1
    pair=(nn.functional.softplus(-delta)*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
    # No invented negative for all-correct questions; identical SFT fallback.
    return torch.where(mask.any((1,2)),pair,supervised(s,y)).mean()

def weights(model,mean,scale):
    d=dict(mean=mean,scale=scale,layers=np.asarray(3))
    for i,m in enumerate(x for x in model if isinstance(x,nn.Linear)):
        d[f'w{i}']=m.weight.detach().cpu().numpy().T.copy();d[f'b{i}']=m.bias.detach().cpu().numpy().copy()
    return d
def restore(path):
    with np.load(path/'model.npz') as z:
        mean=z['mean'].copy();scale=z['scale'].copy();m=network(len(mean))
        for i,l in enumerate(x for x in m if isinstance(x,nn.Linear)):
            with torch.no_grad():l.weight.copy_(torch.from_numpy(z[f'w{i}'].T.copy()));l.bias.copy_(torch.from_numpy(z[f'b{i}'].copy()))
    return m,mean,scale

def fit(model,x,y,arm,epochs,seed,lr,decay,ema=False):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=decay)
    reference=copy.deepcopy(model).eval();average=copy.deepcopy(model);history=[]
    with torch.no_grad():rs=reference(x).squeeze(-1).detach()
    for epoch in range(epochs):
        total=0.;model.train()
        for ix in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/128))):
            s=model(x[ix]).squeeze(-1);loss=objective(s,y[ix],arm,rs[ix]);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if ema:
                with torch.no_grad():
                    for ep,mp in zip(average.parameters(),model.parameters()):ep.lerp_(mp,.02)
            total+=loss.item()*len(ix)
        history.append(total/len(y))
    return (average if ema else model).eval(),history

def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists()
    dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),arms=ARMS,seeds=SEEDS,
        primary='Frozen-reference DPO vs ordinary pairwise logistic vs SFT continuation, same model/input/init/batches/steps/optimizer.',
        feature='ground',warm_epochs=35,continuation_epochs=20,beta=1.,lr=.0001,weight_decay=.01,
        diagnostic='Hold entire normalized-template fold2 out of warm start and continuation. Folds from round22; historically explored TRAIN, not pristine data.',
        full='Every final model uses all4743. Common warm starts are frozen round25 ground_uniform_positive checkpoints at epoch35.',
        deployment='Replace only the two ground MLP members of round25, same weights. No blend/temperature/epoch sweep; choose control if no gain.',
        pair_weights='All correct/incorrect pairs per question, normalized per question. All-correct questions use identical SFT fallback in all arms.',
        train_references_only=True,test_opened=False,rules_url='https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md',
        method_source='https://arxiv.org/abs/2305.18290',source_hash=sha(ROOT/'research/round43_dpo.py')))

def run():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';start=time.perf_counter()
    b,a,y,ids=load('train');x=build(b,a,'ground');dump(OUT/'training_ids.json',ids.tolist())
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:assert np.array_equal(ids,z['ids']);folds=z['folds'];base=z['choices'];bs=z['scores']
    fitix=np.flatnonzero(folds!=2);held=np.flatnonzero(folds==2)
    assert not {family(b[i]['question']) for i in fitix}&{family(b[i]['question']) for i in held}
    dump(OUT/'diagnostic_partition.json',dict(fit_ids=ids[fitix].tolist(),held_ids=ids[held].tolist()))
    records=[];pred={};fullscores={}
    for stage in ('diagnostic','full'):
        fitrows=fitix if stage=='diagnostic' else np.arange(len(y))
        for seed in SEEDS:
            if stage=='diagnostic':
                mean=x[fitrows].mean((0,1));scale=x[fitrows].std((0,1)).clip(.001)
                torch.manual_seed(seed);warm=network(x.shape[-1]).to(device)
                xx=torch.as_tensor((x[fitrows]-mean)/scale,device=device);yy=torch.as_tensor(y[fitrows],device=device)
                warm,h=fit(warm,xx,yy,'sft',35,seed,.0003,.1,True)
            else:
                warm,mean,scale=restore(ROOT/f'experiments/round25/models/ground_uniform_positive_s{seed}_e35');warm.to(device)
                xx=torch.as_tensor((x-mean)/scale,device=device);yy=torch.as_tensor(y,device=device)
            if stage=='diagnostic':pred[f'warm_{seed}']=dense_forward(x[held],weights(warm,mean,scale))
            for arm in ARMS:
                model,h=fit(copy.deepcopy(warm),xx,yy,arm,20,seed+43043,.0001,.01)
                out=OUT/stage/f'{arm}_s{seed}';out.mkdir(parents=True);w=weights(model,mean,scale);np.savez(out/'model.npz',**w)
                dump(out/'config.json',dict(family='mlp',features='ground',arm=arm,reference='common_frozen_warm_start',beta=1.,seed=seed,training_n=len(fitrows),epochs=20))
                dump(out/'training.json',dict(loss=h,training_ids=ids[fitrows].tolist()))
                if stage=='diagnostic':pred[f'{arm}_{seed}']=dense_forward(x[held],w)
                print(dict(stage=stage,arm=arm,seed=seed,loss=h[-1],seconds=round(time.perf_counter()-start,2)),flush=True)
        if stage=='diagnostic':
            for arm in ['warm']+ARMS:
                s=sum(normalized(pred[f'{arm}_{seed}']) for seed in SEEDS)/len(SEEDS)
                chosen=s.argmax(1);combined=apply_gate([b[k] for k in held],bs[held]+.25*s,'parse_and_repair',4)
                ok=y[held,combined].astype(bool);old=y[held,base[held]].astype(bool)
                r=dict(arm=arm,n=len(held),standalone_correct=int(y[held,chosen].sum()),with_oof_base_correct=int(ok.sum()),base_correct=int(old.sum()),repairs=int((ok&~old).sum()),harms=int((~ok&old).sum()))
                records.append(r);print('HELD',r,flush=True)
            dump(OUT/'diagnostic_result.json',records);np.savez(OUT/'diagnostic_scores.npz',held=held,**pred)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-start,device=device,test_opened=False))

if __name__=='__main__':globals()[sys.argv[1]]()
