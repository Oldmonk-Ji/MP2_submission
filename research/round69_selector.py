"""Answer-equivalence stacking with explicit correlated-trace evidence."""
from pathlib import Path
import json,re,numpy as np
from features import metadata
from round68_selector import components
from round22_selector import normalized
from round4_gates import apply_gate
from round9_group_features import candidate_scores

def design(bank,arrays,ss,weights,g,diversity):
    z=normalized(ss);bs=z@np.asarray(weights);answers=metadata(bank)['answers'];rows=[];mapping=[];sizes=[];bases=[]
    for i,row in enumerate(bank):
        groups=[]
        for j,a in enumerate(answers[i]):
            if a is None:continue
            found=next((q for q in groups if answers[i][q[0]]==a),None)
            if found is None:groups.append([j])
            else:found.append(j)
        sizes.append(len(groups));prob=np.exp(z[i]-z[i].max(0));prob/=prob.sum(0)
        for group in groups:
            ix=np.array(group);others=np.array([j for j in range(4) if j not in group and answers[i][j] is not None],int)
            own=z[i,ix];alt=z[i,others] if len(others) else own
            wins=(own>=z[i].max(0)-1e-12).any(0).astype(float)
            a=g[i,ix,:12];b=g[i,others,:12] if len(others) else a
            f=np.r_[own.max(0),own.mean(0),own.min(0),own.std(0),own.max(0)-alt.max(0),prob[ix].sum(0),wins,
                len(ix)/4,len(groups)/4,len(others)/4,bs[i,ix].max(),bs[i,ix].mean(),bs[i,ix].max()-bs[i,others].max() if len(others) else 0,
                a.mean(0),a.min(0),a.max(0),a.mean(0)-b.mean(0)]
            if diversity:
                texts=[row['candidates'][j]['text'] for j in group];tokens=[set(re.findall(r'[a-z]+|\d+',t.lower())) for t in texts]
                e=np.asarray(arrays['embeddings'][i,ix],float);e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-9)
                cs=[];js=[]
                for j in range(len(ix)):
                    for k in range(j):cs.append(float(e[j]@e[k]));js.append(len(tokens[j]&tokens[k])/max(1,len(tokens[j]|tokens[k])))
                f=np.r_[f,len(set(texts))/len(texts),np.mean(cs) if cs else 0,max(cs,default=0),np.mean(js) if js else 0,max(js,default=0)]
            rows.append(f);mapping.append((i,group));bases.append(bs[i,ix].max())
    return np.asarray(rows,np.float32),np.asarray(bases),mapping,np.asarray(sizes),bs

def predict(bank,arrays,checkpoint):
    if not bank:return []
    import lightgbm as lgb
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());ss,g=components(bank,arrays,p/'baseline')
    x,base,mapping,sizes,bs=design(bank,arrays,ss,cfg['weights'],g,cfg['diversity'])
    if not len(x):return apply_gate(bank,bs,'parse_and_repair',4).astype(int).tolist()
    m=lgb.Booster(model_file=str(p/'meta.txt'));v=base+m.predict(x,num_threads=4)
    s=candidate_scores(v,mapping,bs,bank)
    return apply_gate(bank,s,'parse_and_repair',4).astype(int).tolist()
