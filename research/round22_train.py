"""Every final expert is fitted on 4743 official training questions."""
from round22_common import ROOT,OUT,INCUMBENT,read,dump,sha,plan,load,subset
from features import metadata,prepare
from round3_features import build
from round3_train import text_fit,pair_data
from round3_joint import fit as joint_fit
from round3_selector import leaf_scores
from train_models import fit_mlp
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import argparse,time
import numpy as np


def fit_experts(b,a,y,d,da,path,ids):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'done.json').exists():
        assert read(path/'training_ids.json')==list(map(int,ids))
        return
    dump(path/'training_ids.json',list(map(int,ids)))
    tick=time.perf_counter();ss=[]
    for i in range(5):
        dest=path/'baseline'/f'member{i}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'eval.npz').exists():
            with np.load(dest/'eval.npz') as z:ss.append(z['scores'])
            continue
        cfg=read(INCUMBENT/'base'/f'member{i}'/'config.json')
        cfg.update(training_questions=len(b),training_partition='round22 eligible official TRAIN rows',round22_plan_sha=sha(OUT/'plan.json'))
        start=time.perf_counter();print('FIT',str(path.relative_to(OUT)),i,'n',len(b),flush=True)
        if cfg['family']=='mlp':
            dump(dest/'config.json',cfg)
            fit_mlp(build(b,a,cfg['features']),y,None,None,metadata(b)['valid'],cfg,dest,fixed_epochs=cfg['fixed_epochs'])
            s=leaf_scores(d,da,dest)
        elif cfg['family']=='joint':s=joint_fit(b,a,y,d,da,cfg,dest)
        else:s,_=text_fit(b,y,d,cfg['mode'],cfg['C'],dest)
        np.savez(dest/'eval.npz',scores=s);ss.append(s)
        print('FIT DONE',i,'seconds',round(time.perf_counter()-start,2),flush=True)
    kernel=path/'kernel';kernel.mkdir(exist_ok=True)
    if not (kernel/'eval.npz').exists():
        cfg=plan()['kernel'];rng=np.random.default_rng(cfg['seed']);start=time.perf_counter()
        pca=PCA(n_components=cfg['pca'],svd_solver='randomized',random_state=cfg['seed']).fit(a['embeddings'].reshape(-1,1024))
        space=dict(pca_mean=pca.mean_.astype(np.float32),components=pca.components_.astype(np.float32),
            pca_scale=np.sqrt(np.maximum(pca.explained_variance_,pca.explained_variance_[0]*.01)).astype(np.float32))
        def compact(bb,aa):
            p=prepare(bb,aa);e=((p['embedding']-space['pca_mean'])@space['components'].T)/space['pca_scale']
            av=e.mean(1,keepdims=True);small=np.concatenate([p['base'],p['numeric'][:,:,5:],p['checks2']],2)
            return np.concatenate([e,e-av,e*av,small],2).reshape(len(bb)*4,-1).astype(np.float32)
        x=compact(b,a);n=cfg['pca'];small=x.shape[-1]-3*n
        space.update(x_mean=x.mean(0),x_scale=x.std(0).clip(.001),block_scale=np.r_[np.full(3*n,1/np.sqrt(n)),np.full(small,1/np.sqrt(small))].astype(np.float32))
        def standard(x):return np.clip((x-space['x_mean'])/space['x_scale'],-8,8)*space['block_scale']
        x=standard(x);basis=x[rng.choice(len(x),cfg['anchors'],replace=False)]
        sample=x[rng.choice(len(x),min(1024,len(x)),replace=False)]
        distances=np.maximum((sample**2).sum(1)[:,None]+(basis**2).sum(1)[None,:]-2*sample@basis.T,0)
        space.update(anchors=basis,distance_median=np.array(np.median(distances),np.float32));np.savez(kernel/'space.npz',**space)
        dx=standard(compact(d,da))
        def kernelize(x):
            dist=np.maximum((x**2).sum(1)[:,None]+(basis**2).sum(1)[None,:]-2*x@basis.T,0)
            return np.exp(-cfg['gamma_factor']*dist/float(space['distance_median'])).astype(np.float32)
        x=kernelize(x);dx=kernelize(dx);mean=x.mean(0);scale=x.std(0).clip(.001)
        xx,yy,ww=pair_data(sparse.csr_matrix((x-mean)/scale),y)
        model=LogisticRegression(C=cfg['C'],fit_intercept=False,solver='liblinear',max_iter=2000,random_state=cfg['seed']).fit(xx,yy,sample_weight=ww)
        fixed=kernel/'fixed';fixed.mkdir(exist_ok=True)
        np.savez(fixed/'model.npz',mean=mean,scale=scale,weight=model.coef_[0].astype(np.float32));dump(fixed/'config.json',cfg)
        np.savez(kernel/'eval.npz',scores=(((dx-mean)/scale)@model.coef_[0]).reshape(-1,4))
        print('KERNEL DONE',round(time.perf_counter()-start,2),flush=True)
    with np.load(kernel/'eval.npz') as z:ss.append(z['scores'])
    np.savez(path/'eval_components.npz',scores=np.stack(ss,2))
    assets=[p for p in path.rglob('*') if p.name in ('config.json','model.npz','vocabulary.json','space.npz')]
    dump(path/'done.json',dict(n_fit=len(b),n_eval=len(d),seconds=time.perf_counter()-tick,
        inference_asset_bytes=sum(p.stat().st_size for p in assets),assets={str(p.relative_to(path)):sha(p) for p in assets}))
    print('EXPERTS COMPLETE',path.name,len(b),'seconds',round(time.perf_counter()-tick,2),flush=True)


def full():
    b,a,y,ids=load('train');v,va,vy,vids=load('validation')
    assert len(b)==4743 and len(v)==798
    fit_experts(b,a,y,v,va,OUT/'full',ids)
    np.savez(OUT/'validation_labels_alignment.npz',ids=vids,labels=vy)


def oof():
    from sklearn.model_selection import GroupKFold
    from round14_prepare import family
    b,a,y,ids=load('train');groups=[family(r['question']) for r in b]
    splitter=GroupKFold(n_splits=3,shuffle=True,random_state=plan()['seed'])
    components=np.zeros((len(b),4,6),float);seen=np.zeros(len(b),int)
    for fold,(fit,held) in enumerate(splitter.split(np.zeros(len(b)),groups=groups)):
        assert not {groups[i] for i in fit}&{groups[i] for i in held}
        path=OUT/'oof'/f'fold{fold}';path.mkdir(parents=True,exist_ok=True)
        dump(path/'partition.json',dict(fit_ids=ids[fit].tolist(),held_ids=ids[held].tolist(),grouped_by_normalized_question=True))
        tb,ta,ty=subset(b,a,y,fit);hb,ha,_=subset(b,a,y,held)
        fit_experts(tb,ta,ty,hb,ha,path,ids[fit])
        with np.load(path/'eval_components.npz') as z:components[held]=z['scores']
        seen[held]+=1
    assert (seen==1).all()
    np.savez(OUT/'oof.npz',ids=ids,labels=y,components=components)
    print('OOF COMPLETE',len(b),'training predictions; final experts independently fit all 4743',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['full','oof']);a=p.parse_args()
    assert not (OUT/'frozen.json').exists(), 'Keep frozen runs immutable; use a new output directory.'
    full() if a.stage=='full' else oof()
