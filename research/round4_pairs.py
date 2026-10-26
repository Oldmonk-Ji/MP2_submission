"""Question-conditioned verification of operations linking pairs of quantities.

Training targets are the lowest common ancestor operation in clean arithmetic
trees from TRAIN references. Repeated/ambiguous pair labels are excluded.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from round4_quantities import quantities,quantity_documents,trace,STOP

LABELS=['+','-','r-','*','/','r/']


def pairs(program):
    found=defaultdict(set);budget=[0]
    def walk(p):
        budget[0]+=1
        if budget[0]>300:raise ValueError('large program')
        if p[0]=='q':return {p[1]}
        if p[0]=='c':return set()
        a,b=walk(p[1]),walk(p[2])
        for i in a:
            for j in b:
                if i==j:continue
                lo,hi=sorted((i,j));op=p[0]
                if i>j and op in ('-','/'):op='r'+op
                found[(lo,hi)].add(LABELS.index(op))
        return a|b
    if program is None:return {}
    try:walk(program)
    except ValueError:return {}
    return {p:next(iter(v)) for p,v in found.items() if len(v)==1}


def pair_document(q,qs,i,j):
    a,b=qs[i],qs[j]
    base=quantity_documents(q,qs)
    words=['a_'+w for w in base[i].split() if not w.startswith('x_')]+['b_'+w for w in base[j].split() if not w.startswith('x_')]
    ca=[w for w in a['left'][-3:]+a['right'][:3] if w not in STOP]
    cb=[w for w in b['left'][-3:]+b['right'][:3] if w not in STOP]
    words+=['ab_'+x+'_'+y for x in ca for y in cb]
    import re
    between=re.findall(r'[a-z]+',q[a['end']:b['start']].lower())
    words+=['mid_'+w for w in between]
    words+=['near_'+str(min(j-i,4))]
    if a['right'] and b['right']:words+=['same_next_'+str(a['right'][0].rstrip('s')==b['right'][0].rstrip('s'))]
    return ' '.join(words)


def training_data(refs):
    docs=[];labels=[];weights=[]
    for r in refs:
        t=trace(r['question'],r['answer'],True)
        if not t['root_found'] or t['bad'] or t['collisions']:continue
        relations=pairs(t['program']);n=len(relations)
        for (i,j),label in relations.items():
            # Duplicate source numbers bind several entities to one slot.
            if any(t['quantities'][k]['occurrences']!=1 for k in (i,j)):continue
            docs.append(pair_document(r['question'],t['quantities'],i,j));labels.append(label);weights.append(1/max(1,n))
    return docs,np.asarray(labels),np.asarray(weights)


def fit(refs,path,c=10):
    docs,y,weights=training_data(refs)
    vec=TfidfVectorizer(min_df=2,max_features=12000,token_pattern=r'\S+',lowercase=False,dtype=np.float32)
    x=vec.fit_transform(docs);m=LogisticRegression(C=c,max_iter=800).fit(x,y,sample_weight=weights)
    assert m.classes_.tolist()==list(range(6))
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    (path/'vocabulary.json').write_text(json.dumps({k:int(v) for k,v in vec.vocabulary_.items()}),encoding='utf-8')
    prior=np.bincount(y,minlength=6)/len(y)
    np.savez(path/'model.npz',idf=vec.idf_,weight=m.coef_.T.astype(np.float32),bias=m.intercept_.astype(np.float32),prior=prior)
    info=dict(C=c,questions=len(refs),pairs=len(y),labels=LABELS,class_counts=np.bincount(y,minlength=6).tolist())
    (path/'config.json').write_text(json.dumps(info,indent=2));return info


class PairModel:
    def __init__(self,path):
        path=Path(path)
        self.vec=TfidfVectorizer(vocabulary=json.loads((path/'vocabulary.json').read_text(encoding='utf-8')),
                               token_pattern=r'\S+',lowercase=False,dtype=np.float32)
        with np.load(path/'model.npz',allow_pickle=False) as m:
            self.vec.idf_=m['idf'];self.w=m['weight'];self.b=m['bias'];self.prior=m['prior']
    def probabilities(self,documents):
        z=self.vec.transform(documents)@self.w+self.b;z-=z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
    def evaluate(self,refs):
        docs,y,_=training_data(refs);p=self.probabilities(docs)
        return dict(n=len(y),accuracy=float((p.argmax(1)==y).mean()),nll=float(-np.log(np.maximum(1e-8,p[np.arange(len(y)),y])).mean()))
    def features(self,bank):
        docs=[];specs=[];traces=[]
        for row in bank:
            q=row['question'];qs=quantities(q);tt=[trace(q,c['text']) for c in row['candidates']]
            relations=[pairs(t['program']) for t in tt];needed=sorted(set(pair for rr in relations for pair in rr))
            indices={pair:len(docs)+i for i,pair in enumerate(needed)}
            docs.extend(pair_document(q,qs,*pair) for pair in needed);specs.append((relations,indices));traces.append(tt)
        p=self.probabilities(docs) if docs else np.empty((0,6));out=[]
        for (relations,indices),tt in zip(specs,traces):
            for rr,t in zip(relations,tt):
                selected=[(p[indices[pair]],label) for pair,label in rr.items()]
                lp=np.asarray([np.log(max(probs[label],1e-6)) for probs,label in selected])
                lr=np.asarray([np.log(max(probs[label],1e-6)/self.prior[label]) for probs,label in selected])
                agreement=[probs.argmax()==label for probs,label in selected]
                features=[float(lp.mean()) if len(lp) else 0.,float(lr.mean()) if len(lr) else 0.,
                    float(lr.sum()),float(np.mean(agreement)) if selected else 0.,float(lp.min()) if len(lp) else 0.,
                    np.log1p(len(selected)),float(t['root_found']),np.log1p(t['collisions'])]
                out.append(features)
        return np.asarray(out,np.float32).reshape(len(bank),4,8)
