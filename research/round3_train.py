"""Round 3: internal TRAIN selection, followed by a fixed validation comparison."""
from data_access import ROOT,load
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from features import prepare,metadata
from round3_features import build,grounding,lexical_documents
from selector import model_scores
from train_models import fit_mlp,metrics,SEED

OUT=ROOT/'experiments/round3'


def configs():
    out=[]
    for feature in ('base','ground','answer','answer_ground'):
        for epochs in (35,100):
            out.append(dict(name=f'{feature}_{epochs}',family='mlp',features=feature,loss='mixed',width=128,
                            lr=.0003,weight_decay=.1,dropout=.15,seed=SEED,ema=.98,fixed_epochs=epochs))
    return out


def pair_data(x,y):
    pos=[];neg=[];weights=[]
    for i,row in enumerate(y):
        p=np.flatnonzero(row);n=np.flatnonzero(row==0)
        for a in p:
            for b in n:
                pos.append(4*i+a);neg.append(4*i+b);weights.append(1/max(1,len(p)*len(n)))
    d=x[pos]-x[neg]
    return sparse.vstack([d,-d],format='csr'),np.r_[np.ones(len(pos)),np.zeros(len(pos))],np.tile(weights,2)


def text_fit(train_bank,train_y,eval_bank,mode,c,path):
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=50000,sublinear_tf=True,dtype=np.float32)
    x=vec.fit_transform(lexical_documents(train_bank,mode))
    xx,yy,ww=pair_data(x,train_y)
    model=LogisticRegression(C=c,solver='liblinear',fit_intercept=False,max_iter=400,random_state=SEED)
    model.fit(xx,yy,sample_weight=ww)
    scores=model.decision_function(vec.transform(lexical_documents(eval_bank,mode))).reshape(-1,4)
    path.mkdir(parents=True,exist_ok=True)
    (path/'vocabulary.json').write_text(json.dumps({k:int(v) for k,v in vec.vocabulary_.items()},ensure_ascii=False),encoding='utf-8')
    np.savez(path/'model.npz',idf=vec.idf_.astype(np.float32),weight=model.coef_[0].astype(np.float32))
    (path/'config.json').write_text(json.dumps(dict(family='text',mode=mode,C=c,ngram_range=[1,2],sublinear_tf=True),indent=2))
    return scores,len(vec.vocabulary_)


def text_scores(bank,path):
    cfg=json.loads((path/'config.json').read_text())
    vocab=json.loads((path/'vocabulary.json').read_text(encoding='utf-8'))
    with np.load(path/'model.npz') as m:
        vec=TfidfVectorizer(ngram_range=(1,2),vocabulary=vocab,sublinear_tf=True,dtype=np.float32)
        vec.idf_=m['idf']
        return np.asarray(vec.transform(lexical_documents(bank,cfg['mode']))@m['weight']).reshape(-1,4)


def train_internal():
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'protocol':'train-only 3804/951 split; all 8 dense and 8 lexical specifications fixed before validation',
          'seed':SEED,'dense':configs(),'text_C':[.1,1,10,100],'text_modes':['reasoning','answer_only'],
          'blend_weights':[0,.1,.25,.5,1],
          'selection':'best internal dense; best lexical C per mode; blend coefficient selected on internal train holdout',
          'test_access':False}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    b,a,y=load('train');meta=metadata(b)
    fit,hold=train_test_split(np.arange(len(b)),test_size=.2,random_state=SEED,stratify=y.sum(1))
    np.savez(OUT/'internal_split.npz',fit=fit,holdout=hold)
    p=prepare(b,a);g=grounding(b)
    records=[]
    for cfg in configs():
        path=OUT/'internal'/cfg['name'];path.mkdir(parents=True,exist_ok=True)
        if (path/'result.json').exists():
            records.append(json.loads((path/'result.json').read_text()));continue
        start=time.perf_counter();print('INTERNAL',cfg['name'],flush=True)
        x=build(b,a,cfg['features'],p,g)
        (path/'config.json').write_text(json.dumps(cfg,indent=2))
        fit_mlp(x[fit],y[fit],None,None,meta['valid'][fit],cfg,path,fixed_epochs=cfg['fixed_epochs'])
        s=model_scores(x[hold],path)
        np.savez(path/'holdout_scores.npz',scores=s)
        rec=dict(name=cfg['name'],kind='dense',internal=metrics(s,y[hold],meta['valid'][hold],meta['hard'][hold]),seconds=time.perf_counter()-start)
        (path/'result.json').write_text(json.dumps(rec,indent=2));records.append(rec);print(rec,flush=True)
        del x
    del p,g
    tb=[b[i] for i in fit];hb=[b[i] for i in hold]
    for mode in plan['text_modes']:
        for c in plan['text_C']:
            name=f'text_{mode}_{c}';path=OUT/'internal'/name
            if (path/'result.json').exists():
                records.append(json.loads((path/'result.json').read_text()));continue
            print('INTERNAL',name,flush=True);start=time.perf_counter()
            s,v=text_fit(tb,y[fit],hb,mode,c,path)
            np.savez(path/'holdout_scores.npz',scores=s)
            rec=dict(name=name,kind='text',mode=mode,C=c,vocabulary=v,internal=metrics(s,y[hold],meta['valid'][hold],meta['hard'][hold]),seconds=time.perf_counter()-start)
            (path/'result.json').write_text(json.dumps(rec,indent=2));records.append(rec);print(rec,flush=True)
    best=max([r for r in records if r['kind']=='dense'],key=lambda r:r['internal']['correct'])
    dense=np.load(OUT/'internal'/best['name']/'holdout_scores.npz')['scores']
    chosen={};blends=[]
    for mode in plan['text_modes']:
        tr=max([r for r in records if r.get('mode')==mode],key=lambda r:r['internal']['correct'])
        chosen[mode]=tr
        ts=np.load(OUT/'internal'/tr['name']/'holdout_scores.npz')['scores']
        # Training holdout selects a fixed scalar. No validation-fitted ensemble.
        for w in plan['blend_weights']:
            z=dense/(dense.std(1,keepdims=True)+1e-5)+w*ts/(ts.std(1,keepdims=True)+1e-5)
            blends.append(dict(mode=mode,weight=w,internal=metrics(z,y[hold],meta['valid'][hold],meta['hard'][hold])))
    selection=dict(dense=best,text=chosen,blend=max(blends,key=lambda r:r['internal']['correct']),blends=blends,records=records)
    (OUT/'internal_selection.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection['blend']),flush=True)


def evaluate():
    selection=json.loads((OUT/'internal_selection.json').read_text())
    b,a,y=load('train');bv,av,yv=load('validation');meta=metadata(b);mv=metadata(bv)
    p=prepare(b,a);pv=prepare(bv,av);g=grounding(b);gv=grounding(bv)
    # Four matched architecture ablations at the internally selected duration.
    epoch=int(selection['dense']['name'].rsplit('_',1)[1]);records=[];saved={}
    for cfg in [c for c in configs() if c['fixed_epochs']==epoch]:
        path=OUT/'full'/cfg['name'];path.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter();print('FULL',cfg['name'],flush=True)
        if not (path/'model.npz').exists():
            x=build(b,a,cfg['features'],p,g)
            (path/'config.json').write_text(json.dumps(cfg,indent=2))
            fit_mlp(x,y,None,None,meta['valid'],cfg,path,fixed_epochs=epoch);del x
        s=model_scores(build(bv,av,cfg['features'],pv,gv),path);saved[cfg['name']]=s
        np.savez(path/'validation_scores.npz',scores=s)
        rec=dict(name=cfg['name'],validation=metrics(s,yv,mv['valid'],mv['hard']),seconds=time.perf_counter()-start)
        (path/'result.json').write_text(json.dumps(rec,indent=2));records.append(rec);print(rec,flush=True)
    del p,g
    for mode,tr in selection['text'].items():
        path=OUT/'full'/tr['name'];start=time.perf_counter()
        if not (path/'model.npz').exists():s,_=text_fit(b,y,bv,mode,tr['C'],path)
        else:s=text_scores(bv,path)
        saved[tr['name']]=s;np.savez(path/'validation_scores.npz',scores=s)
        rec=dict(name=tr['name'],validation=metrics(s,yv,mv['valid'],mv['hard']),seconds=time.perf_counter()-start)
        records.append(rec);(path/'result.json').write_text(json.dumps(rec,indent=2));print(rec,flush=True)
    blend=selection['blend'];dense=saved[selection['dense']['name']];text=saved[selection['text'][blend['mode']]['name']]
    s=dense/(dense.std(1,keepdims=True)+1e-5)+blend['weight']*text/(text.std(1,keepdims=True)+1e-5)
    np.savez(OUT/'internal_selected_blend_validation.npz',scores=s)
    records.append(dict(name='internal_selected_blend',config=blend,validation=metrics(s,yv,mv['valid'],mv['hard'])))
    (OUT/'validation_comparison.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['internal','validation']);args=parser.parse_args()
    train_internal() if args.stage=='internal' else evaluate()
