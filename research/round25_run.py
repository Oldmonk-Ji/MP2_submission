"""Controlled supervision and seed diversity, every final expert fits 4743."""
from data_access import ROOT
from round22_common import load,read,dump,sha
from round22_selector import normalized,dense_forward
from features import prepare,metadata
from round3_features import build,grounding
import numpy as np
import torch
from torch import nn
import copy,datetime,functools,shutil,time,os
import round4_gates

OUT=ROOT/'experiments/round25'


def train(x,y,vx,kind,loss_name,seed):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    mean=x.mean((0,1));scale=x.std((0,1)).clip(.001)
    xt=torch.as_tensor((x-mean)/scale,device=device);yt=torch.as_tensor(y,device=device)
    model=nn.Sequential(nn.Linear(x.shape[-1],96),nn.ReLU(),nn.Dropout(.2),nn.Linear(96,32),nn.ReLU(),nn.Linear(32,1)).to(device)
    ema=copy.deepcopy(model);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1)
    losses=[];saved={};tick=time.perf_counter()
    for epoch in range(1,101):
        model.train();total=0.
        for ids in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/128))):
            s=model(xt[ids]).squeeze(-1);target=yt[ids]
            if loss_name=='mixed':
                group=torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(target<.5,-1e9),1)
            else:
                # Every numerically correct candidate contributes, even if another positive is easier.
                group=-(s.log_softmax(1)*target/target.sum(1,keepdim=True)).sum(1)
            bce=nn.functional.binary_cross_entropy_with_logits(s,target,reduction='none').mean(1)
            loss=(group+.25*bce).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,.02)
            total+=float(loss.detach())*len(ids)
        losses.append(total/len(y))
        if epoch in (35,100):
            name=f'{kind}_{loss_name}_s{seed}_e{epoch}';path=OUT/'models'/name;path.mkdir(parents=True)
            weights=dict(mean=mean,scale=scale,layers=np.asarray(3))
            for i,layer in enumerate(m for m in ema if isinstance(m,nn.Linear)):
                weights[f'w{i}']=layer.weight.detach().cpu().numpy().T.copy();weights[f'b{i}']=layer.bias.detach().cpu().numpy().copy()
            np.savez(path/'model.npz',**weights)
            cfg=dict(family='mlp',features=kind,loss=loss_name,seed=seed,width=96,dropout=.2,lr=.0003,
                weight_decay=.1,ema=.98,epochs=epoch,training_n=len(y),device=device)
            dump(path/'config.json',cfg);dump(path/'training.json',dict(losses=losses,seconds=time.perf_counter()-tick))
            saved[name]=dense_forward(vx,weights)
            print('TRAINED',name,'loss',losses[-1],'seconds',round(time.perf_counter()-tick,2),flush=True)
    return saved


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists()
    plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        kinds=['ground','answer'],losses=['mixed','uniform_positive'],seeds=[25025,25026],epochs=[35,100],
        blend=[.25,.5,1.],training_n=4743,validation_n=798,test_opened=False,
        comparison='Equal training exposure, width, optimizer, seeds. Uniform positive vs positive marginal likelihood.',
        ensemble='Fixed equal average across two seeds; ground, answer, or both. No per-item routing.',
        baseline='round22 five experts; also predeclared compact baseline drops 7.9MB negative-weight joint branch to make room',
        selection='Full official validation, eligible asset bytes <=16MiB, then correct, fewer harms, size',
        excluded_training_ids=read(ROOT/'experiments/round22/plan.json')['excluded_training_ids'])
    dump(OUT/'plan.json',plan);torch.set_num_threads(4)
    b,a,y,ids=load('train');v,va,vy,_=load('validation');dump(OUT/'training_ids.json',ids.tolist())
    print('FEATURES',flush=True);p=prepare(b,a);g=grounding(b);vp=prepare(v,va);vg=grounding(v)
    all_scores={};start=time.perf_counter()
    for kind in plan['kinds']:
        x=build(b,a,kind,prepared=p,ground=g);vx=build(v,va,kind,prepared=vp,ground=vg)
        for loss in plan['losses']:
            for seed in plan['seeds']:all_scores.update(train(x,y,vx,kind,loss,seed))
        np.savez(OUT/'validation_scores.npz',**all_scores)
    with np.load(ROOT/'experiments/round22/full/eval_components.npz') as z:components=z['scores'][:,:,:5]
    with np.load(ROOT/'experiments/round22/validation_comparison.npz') as z:old=z['incumbent_choice'];control=z['refit_original_choice']
    oldok=vy[np.arange(len(v)),old].astype(bool);controlok=vy[np.arange(len(v)),control].astype(bool);hard=metadata(v)['hard']
    weights=read(ROOT/'experiments/round22/selected_bundle/readout.json')['base_weights']
    src=ROOT/'experiments/round22/selected_bundle/baseline'
    records=[];choices={};configs={}
    original=round4_gates.trace;round4_gates.trace=functools.lru_cache(maxsize=8000)(original)
    def evaluate(name,members,s):
        c=round4_gates.apply_gate(v,s,'parse_and_repair',4);ok=vy[np.arange(len(v)),c].astype(bool)
        size=sum(sum(f.stat().st_size for f in path.iterdir() if f.name in ('config.json','model.npz','vocabulary.json')) for path,w in members)
        cfg=dict(members=[dict(directory=f'member{i}',weight=float(w)) for i,(path,w) in enumerate(members)])
        import json
        size+=len((json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').replace('\n',os.linesep).encode('utf-8'))
        r=dict(name=name,correct=int(ok.sum()),n=len(v),hard_correct=int(ok[hard].sum()),hard_n=int(hard.sum()),
            repaired=int((ok&~oldok).sum()),damaged=int((~ok&oldok).sum()),net=int(ok.sum()-oldok.sum()),
            repaired_control=int((ok&~controlok).sum()),damaged_control=int((~ok&controlok).sum()),
            bytes=size,eligible=size<=16*1024**2)
        records.append(r);choices[name]=c;configs[name]=(members,cfg)
    for compact in (False,True):
        members=[(src/f'member{i}',w) for i,w in enumerate(weights) if not(compact and i==3)]
        s=sum(w*normalized(components[:,:,i]) for i,w in enumerate(weights) if not(compact and i==3))
        bn='compact' if compact else 'full';evaluate(bn,members,s)
        for loss in plan['losses']:
            for ep in plan['epochs']:
                for kinds in [('ground',),('answer',),('ground','answer')]:
                    names=[f'{kind}_{loss}_s{seed}_e{ep}' for kind in kinds for seed in plan['seeds']]
                    added=sum(normalized(all_scores[n]) for n in names)/len(names)
                    for alpha in plan['blend']:
                        ms=members+[(OUT/'models'/n,alpha/len(names)) for n in names]
                        evaluate(f'{bn}_{loss}_{"-".join(kinds)}_e{ep}_a{alpha}',ms,s+alpha*added)
    round4_gates.trace=original
    winner=max((r for r in records if r['eligible']),key=lambda r:(r['correct'],-r['damaged'],-r['bytes']))
    dump(OUT/'selection.json',dict(selected=winner,records=records,seconds=time.perf_counter()-start,test_opened=False))
    np.savez(OUT/'validation_choices.npz',**choices)
    members,cfg=configs[winner['name']];bundle=OUT/'selected_bundle';bundle.mkdir()
    for i,(path,w) in enumerate(members):
        dest=bundle/f'member{i}';dest.mkdir()
        for f in path.iterdir():
            if f.name in ('config.json','model.npz','vocabulary.json'):shutil.copyfile(f,dest/f.name)
    dump(bundle/'config.json',cfg)
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets)
    assert size==winner['bytes'] and size<=16*1024**2
    dump(OUT/'asset_manifest.json',dict(bytes=size,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets}))
    print('SELECTED',winner,flush=True)
    for r in sorted(records,key=lambda r:r['correct'],reverse=True)[:10]:print('TOP',r,flush=True)


if __name__=='__main__':main()
