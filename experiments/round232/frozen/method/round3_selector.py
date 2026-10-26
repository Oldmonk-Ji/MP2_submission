"""CPU-only round-3 inference, with no training-data or reference-file access."""
import json
from pathlib import Path
import numpy as np
from features import metadata
from round3_features import build,lexical_documents
from selector import model_scores


def leaf_scores(bank,arrays,path):
    cfg=json.loads((path/'config.json').read_text())
    if cfg['family']=='mlp':return model_scores(build(bank,arrays,cfg['features']),path)
    if cfg['family'] not in ('joint','text'):raise ValueError(cfg['family'])
    from sklearn.feature_extraction.text import TfidfVectorizer
    vocab=json.loads((path/'vocabulary.json').read_text(encoding='utf-8'))
    vec=TfidfVectorizer(vocabulary=vocab,ngram_range=(1,2),sublinear_tf=True,dtype=np.float32)
    with np.load(path/'model.npz',allow_pickle=False) as m:
        vec.idf_=m['idf'];sx=vec.transform(lexical_documents(bank,cfg['mode']))
        if cfg['family']=='text':return np.asarray(sx@m['weight']).reshape(-1,4)
        t=sx@m['lexical']*4
        if cfg.get('question'):
            qb=[dict(question=r['question'],candidates=[dict(text=r['question'])]) for r in bank]
            q=np.repeat(vec.transform(lexical_documents(qb))@m['lexical']*4,4,axis=0)
            t=np.concatenate([t,q,t*q,t-q],1)
        x=build(bank,arrays,cfg['features']).reshape(len(bank)*4,-1)
        h=np.concatenate([(x-m['mean'])/m['scale'],t],1)
        for i in range(int(m['layers'])):
            h=h@m[f'w{i}']+m[f'b{i}']
            if i<int(m['layers'])-1:h=np.maximum(h,0)
    return h.reshape(-1,4)


def scores(bank,arrays,checkpoint):
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text())
    if cfg['family']!='round3_ensemble':return leaf_scores(bank,arrays,path)
    result=np.zeros((len(bank),4),np.float64)
    for member in cfg['members']:
        if member['weight']==0:continue
        s=leaf_scores(bank,arrays,path/member['directory'])
        if cfg.get('normalization')=='within_question':s=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-5)
        result+=member['weight']*s
    return result


def predict(bank,arrays,checkpoint):
    if not bank:return []
    return np.where(metadata(bank)['valid'],scores(bank,arrays,checkpoint),-1e30).argmax(1).astype(int).tolist()


def predict_with_confidence(bank,arrays,checkpoint):
    from confidence import confidence_features
    if not bank:return [],[]
    x,choice=confidence_features(scores(bank,arrays,checkpoint),metadata(bank))
    with np.load(Path(checkpoint)/'confidence_calibrator.npz',allow_pickle=False) as c:
        z=x@c['weight']+c['bias']
    return choice.astype(int).tolist(),(1/(1+np.exp(-np.clip(z,-50,50)))).tolist()
