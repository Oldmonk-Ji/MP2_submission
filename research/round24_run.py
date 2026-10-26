"""Predeclared controlled quadratic experiments; TEST blocked by data_access."""
from data_access import ROOT
from round22_common import load, read, dump, sha
from round24_selector import question_docs, transform, design
from round22_selector import normalized
from features import prepare, metadata
from round3_features import grounding
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
import datetime, functools, shutil, time
import round4_gates

OUT = ROOT/'experiments/round24'


def fit(x, y, offset, penalty):
    # Each question has unit total pair weight; all positive/negative pairs.
    xx, oo, ww = [], [], []
    for i in range(len(y)):
        pos, neg = np.flatnonzero(y[i]), np.flatnonzero(1-y[i])
        pairs = [(a,b) for a in pos for b in neg]
        for a,b in pairs:
            xx.append(x[i,a]-x[i,b]); oo.append(offset[i,a]-offset[i,b]); ww.append(1/len(pairs))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tx = torch.as_tensor(np.asarray(xx), device=device)
    to = torch.as_tensor(np.asarray(oo, np.float32), device=device)
    tw = torch.as_tensor(np.asarray(ww, np.float32), device=device); tw /= tw.sum()
    w = torch.zeros(tx.shape[1], device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1., max_iter=150, history_size=12, tolerance_grad=1e-7,
                           tolerance_change=1e-9, line_search_fn='strong_wolfe')
    calls = []
    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.softplus(-(tx@w+to))*tw).sum()+penalty*w.square().sum()/2
        loss.backward(); calls.append(float(loss.detach())); return loss
    opt.step(closure)
    result = w.detach().cpu().numpy().copy()
    return result, dict(objective=calls[-1],evaluations=len(calls),pairs=len(xx),device=device)


def main():
    torch.set_num_threads(4); torch.manual_seed(24024)
    OUT.mkdir(exist_ok=True)
    plan = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=24024,
        method='Convex pairwise quadratic ranker with explicit question TFIDF/SVD x candidate PCA interactions',
        kinds=['linear','quadratic','question','no_interaction'],penalties=[.0001,.001,.01],
        alphas=[.1,.25,.5,1.],pca=64,question_svd=32,question_vocabulary=3000,
        training_n=4743,validation_n=798,test_opened=False,offset='round22 grouped train OOF baseline',
        selection='complete validation, correct then fewer harms then smaller alpha',
        caveat='Fixed supplied embeddings can encode question information. quadratic ablates explicit question text only, not all question information.',
        excluded_training_ids=read(ROOT/'experiments/round22/plan.json')['excluded_training_ids'])
    assert not (OUT/'plan.json').exists(), 'Use a new directory; preserve completed runs.'
    dump(OUT/'plan.json', plan)
    b,a,y,ids = load('train'); v,va,vy,_ = load('validation')
    dump(OUT/'training_ids.json', ids.tolist())
    print('PREPARE', len(b), len(v), flush=True)
    tick=time.perf_counter()
    pca=PCA(n_components=64,svd_solver='randomized',random_state=24024).fit(a['embeddings'].reshape(-1,1024))
    vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=3,max_features=3000,dtype=np.float32)
    tq=vec.fit_transform(question_docs(b))
    svd=TruncatedSVD(n_components=32,random_state=24024).fit(tq)
    pp=prepare(b,a);small=np.concatenate([pp['base'],pp['numeric'][:,:,5:],pp['checks2'],grounding(b)],2)
    space=dict(pca_mean=pca.mean_,pca=pca.components_,pca_scale=np.sqrt(pca.explained_variance_).clip(.01),
        idf=vec.idf_,q_svd=svd.components_,q_scale=(tq@svd.components_.T).std(0).clip(.01),
        small_mean=small.mean((0,1)),small_scale=small.std((0,1)).clip(.01))
    space={k:np.asarray(z,np.float32) for k,z in space.items()}
    vocab={k:int(z) for k,z in vec.vocabulary_.items()}
    np.savez(OUT/'space.npz',**space);dump(OUT/'vocabulary.json',vocab)
    tparts=transform(b,a,space,vocab);vparts=transform(v,va,space,vocab)
    weights=np.asarray(read(ROOT/'experiments/round22/selected_bundle/readout.json')['base_weights'])
    with np.load(ROOT/'experiments/round22/oof.npz') as z:
        assert np.array_equal(z['ids'],ids)
        offset=normalized(z['components'][:,:,:5])@weights
    with np.load(ROOT/'experiments/round22/full/eval_components.npz') as z:
        base=normalized(z['scores'][:,:,:5])@weights
    with np.load(ROOT/'experiments/round22/validation_comparison.npz') as z:
        old=z['incumbent_choice'];control=z['refit_original_choice']
    oldok=vy[np.arange(len(v)),old].astype(bool);controlok=vy[np.arange(len(v)),control].astype(bool)
    hard=metadata(v)['hard'];records=[];all_scores={};all_choices={'control':control}
    original=round4_gates.trace;round4_gates.trace=functools.lru_cache(maxsize=8000)(original)
    print('PREPARED seconds',time.perf_counter()-tick,flush=True)
    for kind in plan['kinds']:
        x=design(*tparts,kind);vx=design(*vparts,kind)
        for penalty in plan['penalties']:
            name=f'{kind}_p{penalty}';path=OUT/'models'/name;path.mkdir(parents=True)
            start=time.perf_counter();weight,info=fit(x,y,offset,penalty)
            info.update(seconds=time.perf_counter()-start,features=x.shape[-1],training_n=len(b))
            np.savez(path/'model.npz',weight=weight);dump(path/'config.json',dict(kind=kind,penalty=penalty));dump(path/'training.json',info)
            delta=vx@weight;all_scores[name]=delta
            standalone=round4_gates.apply_gate(v,delta,'parse_and_repair',4)
            for alpha in plan['alphas']:
                choice=round4_gates.apply_gate(v,base+alpha*normalized(delta),'parse_and_repair',4)
                key=f'{name}_a{alpha}';all_choices[key]=choice;ok=vy[np.arange(len(v)),choice].astype(bool)
                rec=dict(name=key,model=name,alpha=alpha,correct=int(ok.sum()),n=len(v),
                    hard_correct=int(ok[hard].sum()),hard_n=int(hard.sum()),
                    repaired=int((ok&~oldok).sum()),damaged=int((~ok&oldok).sum()),net=int(ok.sum()-oldok.sum()),
                    repaired_control=int((ok&~controlok).sum()),damaged_control=int((~ok&controlok).sum()),
                    standalone_correct=int(vy[np.arange(len(v)),standalone].sum()))
                records.append(rec)
            print('FIT',name,info,'BEST',max(records[-4:],key=lambda z:z['correct']),flush=True)
            dump(OUT/'records.json',records)
    round4_gates.trace=original
    np.savez(OUT/'validation_scores.npz',**all_scores)
    np.savez(OUT/'validation_choices.npz',**all_choices)
    winner=max(records,key=lambda z:(z['correct'],-z['damaged'],-z['alpha']))
    dump(OUT/'selection.json',dict(selected=winner,control_correct=int(controlok.sum()),historical_best_correct=int(oldok.sum()),
        records=records,seconds=time.perf_counter()-tick,test_opened=False))
    print('WINNER',winner,flush=True)
    if winner['correct'] < int(controlok.sum()):return
    bundle=OUT/'selected_bundle';branch=bundle/'branch';branch.mkdir(parents=True)
    src=ROOT/'experiments/round22/selected_bundle/baseline'
    shutil.copytree(src,bundle/'baseline')
    # round3_selector ensemble configuration refers only to the five refit experts.
    cfg=read(ROOT/'experiments/round4/selected_bundle/base/config.json')
    dump(bundle/'baseline/config.json',cfg)
    for f in ['space.npz','vocabulary.json']:shutil.copyfile(OUT/f,branch/f)
    for f in ['model.npz','config.json']:shutil.copyfile(OUT/'models'/winner['model']/f,branch/f)
    dump(bundle/'fusion.json',dict(alpha=winner['alpha']))
    assets=[f for f in bundle.rglob('*') if f.is_file()]
    size=sum(f.stat().st_size for f in assets)
    dump(OUT/'asset_manifest.json',dict(bytes=size,limit=16*1024**2,eligible=size<=16*1024**2,
        hashes={str(f.relative_to(ROOT)):sha(f) for f in assets}))
    print('BUNDLE BYTES',size,flush=True)


if __name__=='__main__':main()
