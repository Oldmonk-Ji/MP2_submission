"""Regularized kernel ranking on TRAIN-fitted compact official representations."""
from round19_common import OUT, load_part, read_plan, dump, sha
from round19_select import evaluate_development,selection_key
from features import prepare
from round3_train import pair_data
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import json, time, warnings
import numpy as np


def compact_inputs(bank,arrays,space):
    p=prepare(bank,arrays);e=p['embedding']
    projected=((e-space['pca_mean'])@space['components'].T)/space['pca_scale']
    avg=projected.mean(1,keepdims=True)
    small=np.concatenate([p['base'],p['numeric'][:,:,5:],p['checks2']],2)
    x=np.concatenate([projected,projected-avg,projected*avg,small],2).reshape(len(bank)*4,-1)
    return x.astype(np.float32)


def standardize(x,space):return np.clip((x-space['x_mean'])/space['x_scale'],-8,8)*space['block_scale']


def fit_space(bank,arrays):
    cfg=read_plan()['kernel'];seed=read_plan()['seed'];rng=np.random.default_rng(seed)
    e=arrays['embeddings'].reshape(-1,1024)
    pca=PCA(n_components=cfg['pca'],svd_solver='randomized',random_state=seed).fit(e)
    space=dict(pca_mean=pca.mean_.astype(np.float32),components=pca.components_.astype(np.float32),
               pca_scale=np.sqrt(np.maximum(pca.explained_variance_,pca.explained_variance_[0]*.01)).astype(np.float32))
    x=compact_inputs(bank,arrays,space);n=cfg['pca'];small=x.shape[-1]-3*n
    space.update(x_mean=x.mean(0),x_scale=x.std(0).clip(.001),
                 block_scale=np.r_[np.full(3*n,1/np.sqrt(n)),np.full(small,1/np.sqrt(small))].astype(np.float32))
    x=standardize(x,space);basis=x[rng.choice(len(x),cfg['anchors'],replace=False)]
    sample=x[rng.choice(len(x),min(1024,len(x)),replace=False)]
    dist=np.maximum((sample**2).sum(1)[:,None]+(basis**2).sum(1)[None,:]-2*sample@basis.T,0)
    space.update(anchors=basis,distance_median=np.array(np.median(dist),np.float32))
    return space,x


def design(x,space,gamma):
    if gamma is None:return x
    basis=space['anchors'];dist=np.maximum((x**2).sum(1)[:,None]+(basis**2).sum(1)[None,:]-2*x@basis.T,0)
    return np.exp(-gamma*dist/float(space['distance_median'])).astype(np.float32)


def score(bank,arrays,path):
    from pathlib import Path
    path=Path(path);cfg=json.loads((path/'config.json').read_text())
    with np.load(path.parent/'space.npz',allow_pickle=False) as z:space={k:z[k] for k in z.files}
    x=standardize(compact_inputs(bank,arrays,space),space);x=design(x,space,cfg['gamma_factor'])
    with np.load(path/'model.npz',allow_pickle=False) as z:s=((x-z['mean'])/z['scale'])@z['weight']
    return np.asarray(s).reshape(-1,4)


def main():
    assert not (OUT/'frozen.json').exists();assert (OUT/'baseline/done.json').exists()
    root=OUT/'kernel';root.mkdir(exist_ok=True)
    assert not (root/'selection.json').exists()
    b,a,y,_=load_part('fit');db,da,dy,_=load_part('development');cfg=read_plan()['kernel'];records=[]
    tick=time.perf_counter();space,x=fit_space(b,a);np.savez(root/'space.npz',**space)
    dx=standardize(compact_inputs(db,da,space),space)
    print('KERNEL SPACE',x.shape,'seconds',round(time.perf_counter()-tick,2),flush=True)
    for gamma in [None]+cfg['gamma_factors']:
        k=design(x,space,gamma);dk=design(dx,space,gamma)
        mean=k.mean(0);scale=k.std(0).clip(.001);k=(k-mean)/scale;dk=(dk-mean)/scale
        xx,yy,ww=pair_data(sparse.csr_matrix(k),y)
        for C in cfg['linear_control_C'] if gamma is None else cfg['C']:
            name=f'linear_C{C}' if gamma is None else f'rbf_g{gamma}_C{C}'
            path=root/name;path.mkdir(exist_ok=True);start=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                model=LogisticRegression(C=C,fit_intercept=False,solver='liblinear',max_iter=2000,random_state=read_plan()['seed']).fit(xx,yy,sample_weight=ww)
            weight=model.coef_[0].astype(np.float32);s=(dk@weight).reshape(-1,4)
            np.savez(path/'model.npz',mean=mean,scale=scale,weight=weight)
            dump(path/'config.json',dict(gamma_factor=gamma,C=C,family='kernel',plan_sha=sha(OUT/'plan.json')))
            np.savez(path/'development.npz',scores=s)
            asset_bytes=(root/'space.npz').stat().st_size+sum((path/k).stat().st_size for k in ('config.json','model.npz'))
            record=dict(name=name,family='kernel' if gamma is not None else 'linear_control',path=str(path.relative_to(OUT)),
                        asset_bytes=asset_bytes,fit_seconds=time.perf_counter()-start,iterations=int(model.n_iter_[0]),
                        warnings=[str(w.message) for w in caught],**evaluate_development(s,db,dy,asset_bytes))
            dump(path/'selected.json',record);records.append(record)
            print('KERNEL',name,record['selected'],flush=True)
    chosen=max([r for r in records if r['family']=='kernel'],key=lambda r:selection_key(r['selected']))
    linear=max([r for r in records if r['family']=='linear_control'],key=lambda r:selection_key(r['selected']))
    dump(root/'selection.json',dict(selected=chosen,linear_control=linear,records=records,seconds=time.perf_counter()-tick))


if __name__=='__main__':main()
