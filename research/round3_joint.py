"""Joint learned n-gram and supplied-representation scorer, trained from scratch.

TF-IDF vocabulary, IDF, scalers, and neural weights see the fitting TRAIN subset
only. GPU training is optional; exported inference uses NumPy/SciPy/sklearn.
"""
from data_access import ROOT,load
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from features import prepare,metadata
from round3_features import build,lexical_documents
from train_models import SEED,metrics

OUT=ROOT/'experiments/round3'


def question_documents(bank):
    proxy=[dict(question=r['question'],candidates=[dict(text=r['question'])]) for r in bank]
    return lexical_documents(proxy)


def vectorizer(path=None):
    kwargs=dict(ngram_range=(1,2),sublinear_tf=True,dtype=np.float32)
    if path is None:return TfidfVectorizer(min_df=3,max_features=40000,**kwargs)
    v=TfidfVectorizer(vocabulary=json.loads((path/'vocabulary.json').read_text(encoding='utf-8')),**kwargs)
    with np.load(path/'model.npz') as m:v.idf_=m['idf']
    return v


def fit(bank,arrays,y,eval_bank,eval_arrays,cfg,path):
    import torch
    from torch import nn
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    rng=np.random.default_rng(cfg['seed'])
    vec=vectorizer();sx=vec.fit_transform(lexical_documents(bank,cfg['mode'])).tocsr()
    sq=vec.transform(question_documents(bank)).tocsr() if cfg.get('question') else None
    x=build(bank,arrays,cfg['features'])
    sc=StandardScaler().fit(x.reshape(-1,x.shape[-1]))
    xt=torch.as_tensor(((x-sc.mean_)/sc.scale_).astype(np.float32),device=device)
    yt=torch.as_tensor(y,device=device)
    class Joint(nn.Module):
        def __init__(self):
            super().__init__()
            self.lex=nn.Parameter(torch.randn(sx.shape[1],32)*.02)
            self.net=nn.Sequential(nn.Linear(x.shape[-1]+(128 if cfg.get('question') else 32),128),nn.ReLU(),nn.Dropout(.15),
                                   nn.Linear(128,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,dense,sp,qp=None):
            t=torch.sparse.mm(sp,self.lex)*4
            t=nn.functional.dropout(t,p=.3,training=self.training)
            if qp is not None:
                q=torch.sparse.mm(qp,self.lex).repeat_interleave(4,dim=0)*4
                t=torch.cat([t,q,t*q,t-q],1)
            return self.net(torch.cat([dense.reshape(-1,x.shape[-1]),t],1)).reshape(-1,4)
    model=Joint().to(device);ema=copy.deepcopy(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1)
    losses=[]
    for epoch in range(1,cfg['epochs']+1):
        model.train();loss_sum=0
        for ids in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/128))):
            rowids=(ids[:,None]*4+np.arange(4)).reshape(-1)
            coo=sx[rowids].tocoo()
            sp=torch.sparse_coo_tensor(torch.as_tensor(np.vstack([coo.row,coo.col]),device=device),
                torch.as_tensor(coo.data,device=device),coo.shape,device=device)
            qp=None
            if sq is not None:
                qc=sq[ids].tocoo()
                qp=torch.sparse_coo_tensor(torch.as_tensor(np.vstack([qc.row,qc.col]),device=device),torch.as_tensor(qc.data,device=device),qc.shape,device=device)
            s=model(xt[ids],sp,qp);target=yt[ids]
            loss=(torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(target<.5,-1e9),1)).mean()
            loss=loss+.25*nn.functional.binary_cross_entropy_with_logits(s,target)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,.02)
            loss_sum+=float(loss.detach())*len(ids)
        losses.append(loss_sum/len(y))
        if epoch%10==0:print(cfg['name'],'epoch',epoch,'loss',losses[-1],flush=True)
    model=ema.cpu().eval()
    weights=dict(mean=sc.mean_.astype(np.float32),scale=sc.scale_.astype(np.float32),idf=vec.idf_,
                 lexical=model.lex.detach().numpy(),layers=np.asarray(3))
    for i,layer in enumerate(m for m in model.net if isinstance(m,nn.Linear)):
        weights[f'w{i}']=layer.weight.detach().numpy().T.copy();weights[f'b{i}']=layer.bias.detach().numpy().copy()
    path.mkdir(parents=True,exist_ok=True)
    np.savez(path/'model.npz',**weights)
    (path/'vocabulary.json').write_text(json.dumps({k:int(v) for k,v in vec.vocabulary_.items()}),encoding='utf-8')
    (path/'config.json').write_text(json.dumps(cfg,indent=2))
    (path/'training.json').write_text(json.dumps(dict(loss=losses,device=device,
        gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,train_questions=len(bank)),indent=2))
    # Export is verified against the training implementation on TRAIN examples.
    coo=sx[:28].tocoo()
    sp=torch.sparse_coo_tensor(torch.as_tensor(np.vstack([coo.row,coo.col])),torch.as_tensor(coo.data),coo.shape)
    qp=None
    if sq is not None:
        qc=sq[:7].tocoo();qp=torch.sparse_coo_tensor(torch.as_tensor(np.vstack([qc.row,qc.col])),torch.as_tensor(qc.data),qc.shape)
    with torch.no_grad():reference=model(xt[:7].cpu(),sp,qp).numpy()
    exported=scores(bank[:7],{'embeddings':arrays['embeddings'][:7]},path)
    discrepancy=float(np.max(np.abs(reference-exported)))
    assert np.allclose(reference,exported,atol=2e-4,rtol=1e-4),discrepancy
    (path/'export_check.json').write_text(json.dumps(dict(max_absolute_error=discrepancy)))
    return scores(eval_bank,eval_arrays,path)


def scores(bank,arrays,path):
    path=Path(path);cfg=json.loads((path/'config.json').read_text())
    v=vectorizer(path);sx=v.transform(lexical_documents(bank,cfg['mode']))
    x=build(bank,arrays,cfg['features']).reshape(len(bank)*4,-1)
    with np.load(path/'model.npz') as m:
        text=sx@m['lexical']*4
        if cfg.get('question'):
            q=np.repeat(v.transform(question_documents(bank))@m['lexical']*4,4,axis=0)
            text=np.concatenate([text,q,text*q,text-q],1)
        h=np.concatenate([(x-m['mean'])/m['scale'],text],1)
        for i in range(3):
            h=h@m[f'w{i}']+m[f'b{i}']
            if i<2:h=np.maximum(h,0)
    return h.reshape(-1,4)


def run(stage):
    specs=[dict(name=f'joint_{mode}_{ep}',family='joint',features='base',
                mode='reasoning' if mode=='conditioned' else mode,question=mode=='conditioned',epochs=ep,seed=SEED)
           for mode in ('reasoning','answer_only','conditioned') for ep in (10,35,100)]
    # Registered before this branch examines any official validation labels.
    (OUT/'joint_plan.json').write_text(json.dumps(dict(configurations=specs,
        selection='one winner by train internal correct; fit winner and matched text ablation on full train',
        adaptation='10 epochs and explicit question interactions added after initial TRAIN internal overfitting; before official validation',
        fusion='within-question standardization; dense internal winner plus joint internal winner',
        fusion_weights=[0,.1,.25,.5,1],test_access=False),indent=2))
    b,a,y=load('train')
    if stage=='internal':
        split=np.load(OUT/'internal_split.npz');ids=split['fit'];held=split['holdout']
        tb=[b[i] for i in ids];hb=[b[i] for i in held]
        ta={'embeddings':a['embeddings'][ids]};ha={'embeddings':a['embeddings'][held]};hy=y[held]
        mm=metadata(hb);records=[]
        for cfg in specs:
            path=OUT/'internal'/cfg['name']
            if (path/'result.json').exists():records.append(json.loads((path/'result.json').read_text()));continue
            tick=time.perf_counter();s=fit(tb,ta,y[ids],hb,ha,cfg,path)
            rec=dict(name=cfg['name'],config=cfg,internal=metrics(s,hy,mm['valid'],mm['hard']),seconds=time.perf_counter()-tick)
            np.savez(path/'holdout_scores.npz',scores=s)
            (path/'result.json').write_text(json.dumps(rec,indent=2));records.append(rec);print(rec,flush=True)
        (OUT/'joint_selection.json').write_text(json.dumps(dict(records=records,best=max(records,key=lambda r:r['internal']['correct'])),indent=2))
    else:
        selection=json.loads((OUT/'joint_selection.json').read_text())
        best=selection['best']['config'];bv,av,yv=load('validation');mm=metadata(bv);records=[]
        for cfg in [c for c in specs if c['epochs']==best['epochs']]:
            path=OUT/'full'/cfg['name'];tick=time.perf_counter()
            s=scores(bv,av,path) if (path/'model.npz').exists() else fit(b,a,y,bv,av,cfg,path)
            rec=dict(name=cfg['name'],config=cfg,validation=metrics(s,yv,mm['valid'],mm['hard']),seconds=time.perf_counter()-tick)
            np.savez(path/'validation_scores.npz',scores=s)
            (path/'result.json').write_text(json.dumps(rec,indent=2));records.append(rec);print(rec,flush=True)
        (OUT/'joint_validation.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['internal','validation']);args=parser.parse_args();run(args.stage)
