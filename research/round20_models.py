"""Reuse frozen model recipes; refit all learned quantities on each retained subset."""
from round20_common import ROOT, OUT, dump, read, plan, sha, normalized
import copy, json, time, warnings
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from features import metadata
from round3_train import pair_data, text_fit
from round3_features import build
from round3_selector import leaf_scores, scores as baseline_scores
from round3_joint import fit as joint_fit
from train_models import fit_mlp
from round19_model import Comparator, vocabulary, encode_bank, tensor_batch, objective, predict_data, save_model, load_model
from round19_neural import training_loss
from round19_kernel import compact_inputs, standardize, design, score as kernel_scores
import round4_gates
import torch


def fit_member(index, b, a, y, d, da, path):
    path.mkdir(parents=True, exist_ok=True)
    if (path/'completed.json').exists():
        with np.load(path/'development_scores.npz') as z: return z['scores']
    source = ROOT/'experiments/round4/selected_bundle/base'/f'member{index}'
    cfg = read(source/'config.json')
    cfg.update(training_questions=len(b), training_partition='round20 retained FIT subset', round20_plan_sha=sha(OUT/'plan.json'))
    start = time.perf_counter()
    print('MEMBER', path, 'n', len(b), flush=True)
    if cfg['family'] == 'mlp':
        dump(path/'config.json', cfg)
        x = build(b, a, cfg['features'])
        fit_mlp(x, y, None, None, metadata(b)['valid'], cfg, path, fixed_epochs=cfg['fixed_epochs'])
        s = leaf_scores(d, da, path)
    elif cfg['family'] == 'joint':
        s = joint_fit(b, a, y, d, da, cfg, path)
    else:
        s, _ = text_fit(b, y, d, cfg['mode'], cfg['C'], path)
    np.savez(path/'development_scores.npz', scores=s)
    dump(path/'completed.json', dict(fit_questions=len(b), evaluation_questions=len(d), seconds=time.perf_counter()-start))
    return s


def fit_baseline(b, a, y, d, da, dy, path):
    path.mkdir(parents=True, exist_ok=True)
    if (path/'done.json').exists(): return read(path/'done.json')
    original = read(ROOT/'experiments/round4/selected_bundle/base/config.json')
    members, ss = [], []
    for i, member in enumerate(original['members']):
        s = fit_member(i, b, a, y, d, da, path/member['directory'])
        ss.append(normalized(s)); members.append(dict(member, weight=0.))
    z = np.stack(ss, 2)
    dx, ty, ww = pair_data(sparse.csr_matrix(z.reshape(-1,5)), dy)
    ranker = LogisticRegression(C=plan()['baseline_stack_C'], fit_intercept=False, solver='liblinear', max_iter=1000, random_state=18018).fit(dx,ty,sample_weight=ww)
    for m,w in zip(members,ranker.coef_[0]): m['weight']=float(w)
    dump(path/'config.json', dict(family='round3_ensemble', normalization='within_question', members=members))
    s=z@ranker.coef_[0]; choice=round4_gates.apply_gate(d,s,'parse_and_repair',4)
    np.savez(path/'development.npz',scores=s,choices=choice)
    assets=[p for p in path.rglob('*') if p.name in ('config.json','model.npz','vocabulary.json')]
    result=dict(fit_questions=len(b),development_questions=len(d),development_correct=int(dy[np.arange(len(d)),choice].sum()),
                inference_asset_bytes=sum(p.stat().st_size for p in assets), assets={str(p.relative_to(path)):sha(p) for p in assets})
    dump(path/'done.json',result)
    print('BASELINE',path.name,result['development_correct'],flush=True)
    return result


def fit_neural(b,a,y,d,da,dy,path):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'done.json').exists():return read(path/'done.json')
    cfg=dict(plan()['neural']);v=vocabulary(b,cfg['vocab_limit'])
    dump(path/'config.json',cfg);dump(path/'vocabulary.json',v)
    x,cov=encode_bank(b,a,v,cfg);dx,dcov=encode_bank(d,da,v,cfg)
    dump(path/'coverage.json',dict(fit=cov,development=dcov))
    e=a['embeddings'].reshape(-1,1024);mean=e.mean(0);scale=e.std(0).clip(.001)
    torch.manual_seed(cfg['seed']);torch.set_num_threads(4);rng=np.random.default_rng(cfg['seed'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=Comparator(cfg,len(v),mean,scale).to(device);ema=copy.deepcopy(model)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    mixed=np.flatnonzero((y.sum(1)>0)&(y.sum(1)<4));best_loss=float('inf');best_epoch=0;history=[];start=time.perf_counter()
    for epoch in range(1,cfg['epochs']+1):
        tick=time.perf_counter();model.train();total=0.;order=rng.permutation(mixed)
        for j in range(0,len(order),cfg['batch']):
            ix=order[j:j+cfg['batch']];target=torch.tensor(y[ix],device=device)
            s,m=model(**tensor_batch(x,ix,device));loss=training_loss(s,m,target,cfg['smooth'])
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,1-cfg['ema'])
            total+=float(loss.detach())*len(ix)
        s,m=predict_data(ema,dx,device,cfg['batch'])
        dev_loss=float(objective(torch.tensor(s),torch.tensor(m),torch.tensor(dy)))
        history.append(dict(epoch=epoch,train_loss=total/len(mixed),development_loss=dev_loss,seconds=time.perf_counter()-tick))
        if dev_loss<best_loss-1e-4:
            best_loss=dev_loss;best_epoch=epoch;save_model(ema,path);np.savez(path/'development.npz',scores=s,margins=m)
        dump(path/'history.json',history)
        print('LONG',path.parent.name,epoch,'fit',round(total/len(mixed),4),'dev',round(dev_loss,4),'best',best_epoch,'sec',round(history[-1]['seconds'],1),flush=True)
        if epoch-best_epoch>=cfg['patience']:break
    result=dict(epoch=best_epoch,development_loss=best_loss,epochs_run=len(history),seconds=time.perf_counter()-start,
        mixed_questions=len(mixed),optimizer_updates=sum(int(np.ceil(len(mixed)/cfg['batch'])) for _ in history),
        parameters=sum(p.numel() for p in model.parameters()),device=device,
        inference_asset_bytes=sum((path/k).stat().st_size for k in ('model.npz','config.json','vocabulary.json')))
    dump(path/'done.json',result)
    return result


def fit_kernel(b,a,y,d,da,path):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'done.json').exists():return read(path/'done.json')
    cfg=plan()['kernel'];rng=np.random.default_rng(cfg['seed']);tick=time.perf_counter()
    e=a['embeddings'].reshape(-1,1024)
    pca=PCA(n_components=cfg['pca'],svd_solver='randomized',random_state=cfg['seed']).fit(e)
    space=dict(pca_mean=pca.mean_.astype(np.float32),components=pca.components_.astype(np.float32),
               pca_scale=np.sqrt(np.maximum(pca.explained_variance_,pca.explained_variance_[0]*.01)).astype(np.float32))
    x=compact_inputs(b,a,space);n=cfg['pca'];small=x.shape[-1]-3*n
    space.update(x_mean=x.mean(0),x_scale=x.std(0).clip(.001),block_scale=np.r_[np.full(3*n,1/np.sqrt(n)),np.full(small,1/np.sqrt(small))].astype(np.float32))
    x=standardize(x,space);basis=x[rng.choice(len(x),cfg['anchors'],replace=False)]
    sample=x[rng.choice(len(x),min(1024,len(x)),replace=False)]
    distances=np.maximum((sample**2).sum(1)[:,None]+(basis**2).sum(1)[None,:]-2*sample@basis.T,0)
    space.update(anchors=basis,distance_median=np.array(np.median(distances),np.float32))
    np.savez(path/'space.npz',**space)
    dx=standardize(compact_inputs(d,da,space),space)
    x=design(x,space,cfg['gamma_factor']);dx=design(dx,space,cfg['gamma_factor'])
    mean=x.mean(0);scale=x.std(0).clip(.001);x=(x-mean)/scale;dx=(dx-mean)/scale
    xx,yy,ww=pair_data(sparse.csr_matrix(x),y)
    with warnings.catch_warnings(record=True) as caught:
        ranker=LogisticRegression(C=cfg['C'],fit_intercept=False,solver='liblinear',max_iter=2000,random_state=cfg['seed']).fit(xx,yy,sample_weight=ww)
    dest=path/'fixed';dest.mkdir(exist_ok=True)
    weight=ranker.coef_[0].astype(np.float32)
    np.savez(dest/'model.npz',mean=mean,scale=scale,weight=weight);dump(dest/'config.json',cfg)
    np.savez(path/'development.npz',scores=(dx@weight).reshape(-1,4))
    result=dict(seconds=time.perf_counter()-tick,iterations=int(ranker.n_iter_[0]),warnings=[str(w.message) for w in caught],
        inference_asset_bytes=sum(p.stat().st_size for p in (path/'space.npz',dest/'model.npz',dest/'config.json')))
    dump(path/'done.json',result);return result


def score_all(bank,arrays,path,heads=('baseline','long','kernel')):
    fusion=read(path/'fusion.json')
    base=baseline_scores(bank,arrays,path/'baseline')
    result={'baseline':base} if 'baseline' in heads else {}
    if 'long' in heads:
        model,vocab,cfg=load_model(path/'long');torch.set_num_threads(4)
        x,_=encode_bank(bank,arrays,vocab,cfg);s,_=predict_data(model,x,batch=cfg['batch'])
        result['long']=base+fusion['alpha']*normalized(s)
    if 'kernel' in heads:
        k=kernel_scores(bank,arrays,path/'kernel/fixed')
        result['kernel']=base+fusion['alpha']*normalized(k)
    return result


def predict(bank,arrays,checkpoint,head='long'):
    from pathlib import Path
    if not bank:return []
    assert head in ('baseline','long','kernel')
    path=Path(checkpoint);cfg=read(path/'fusion.json')
    s=score_all(bank,arrays,path,heads=(head,))[head]
    return round4_gates.apply_gate(bank,s,cfg['gate'],cfg['margin_limit']).astype(int).tolist()
