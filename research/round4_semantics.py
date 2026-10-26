"""Learn how source quantities affect the requested answer, from TRAIN derivations."""
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from round4_quantities import trace,quantities,quantity_documents,semantic_features


def fit(references,path,c=1.,balanced=False):
    docs=[];labels=[];weights=[];owners=[];parsed=0
    for i,r in enumerate(references):
        t=trace(r['question'],r['answer'],True)
        if not t['root_found'] or t['bad'] or t['collisions']:continue
        current=quantity_documents(r['question'],t['quantities']);parsed+=1
        for j,q in enumerate(t['quantities']):
            # Equal numeric values can refer to distinct entities. Do not use
            # their merged derivative as an unambiguous supervised relation.
            if q['occurrences']!=1 or q['value'] in (0,1):continue
            docs.append(current[j]);labels.append(int(t['classes'][j]));owners.append(i)
            weights.append(1/max(1,len(t['quantities'])))
    vec=TfidfVectorizer(min_df=2,max_features=14000,token_pattern=r'\S+',lowercase=False,dtype=np.float32)
    x=vec.fit_transform(docs)
    model=LogisticRegression(C=c,max_iter=700,solver='lbfgs',class_weight='balanced' if balanced else None).fit(x,labels,sample_weight=weights)
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    (path/'vocabulary.json').write_text(json.dumps({k:int(v) for k,v in vec.vocabulary_.items()}),encoding='utf-8')
    np.savez(path/'model.npz',idf=vec.idf_,weight=model.coef_.T.astype(np.float32),bias=model.intercept_.astype(np.float32))
    info=dict(C=c,balanced=balanced,reference_questions=len(references),parsed_clean_references=parsed,
              quantity_examples=len(labels),classes={str(k):int((np.asarray(labels)==k).sum()) for k in (0,1,2)},
              features=x.shape[1],labels='0 unused, 1 positive derivative, 2 negative derivative',
              training_source='provided training references only')
    (path/'config.json').write_text(json.dumps(info,indent=2))
    return info


class QuantityModel:
    def __init__(self,path):
        path=Path(path)
        v=json.loads((path/'vocabulary.json').read_text(encoding='utf-8'))
        self.vectorizer=TfidfVectorizer(vocabulary=v,token_pattern=r'\S+',lowercase=False,dtype=np.float32)
        with np.load(path/'model.npz',allow_pickle=False) as m:
            self.vectorizer.idf_=m['idf'];self.w=m['weight'];self.b=m['bias']
    def predict(self,bank):
        documents=[];lengths=[]
        for r in bank:
            qs=quantities(r['question']);documents+=quantity_documents(r['question'],qs);lengths.append(len(qs))
        if not documents:return [np.empty((0,3)) for _ in bank]
        z=self.vectorizer.transform(documents)@self.w+self.b
        z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
        result=[];offset=0
        for n in lengths:result.append(p[offset:offset+n]);offset+=n
        return result
    def features(self,bank):
        probs=self.predict(bank)
        return np.asarray([semantic_features(r['question'],r['candidates'],p) for r,p in zip(bank,probs)],np.float32)


def relation_evaluation(references,model):
    p=model.predict(references);y=[];probs=[]
    for r,pr in zip(references,p):
        t=trace(r['question'],r['answer'],True)
        if not t['root_found'] or t['bad'] or t['collisions']:continue
        for i,q in enumerate(t['quantities']):
            if q['occurrences']!=1 or q['value'] in (0,1):continue
            y.append(t['classes'][i]);probs.append(pr[i])
    y=np.asarray(y);probs=np.asarray(probs)
    return dict(n=len(y),accuracy=float((probs.argmax(1)==y).mean()),
                nll=float(-np.log(np.maximum(1e-8,probs[np.arange(len(y)),y])).mean()),
                by_class={str(k):dict(n=int((y==k).sum()),recall=float((probs.argmax(1)[y==k]==k).mean())) for k in (0,1,2)})
