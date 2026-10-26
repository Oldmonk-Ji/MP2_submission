"""Answer-equivalence groups and evidence fusion, using inputs only.

The official parser defines equality. No reference answers, IDs, learned entity
labels, or external model calls enter these features.
"""
import numpy as np
from features import metadata

def normalize(s):return (s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-5)

def build(bank,members,base,arithmetic=None,kind='groups'):
    meta=metadata(bank);z=np.stack([normalize(members[:,:,k]) for k in range(members.shape[2])],2)
    features=[];mapping=[];sizes=[];targets=[];baseline=[]
    for i,row in enumerate(bank):
        answers=meta['answers'][i]
        groups=[]
        if kind=='candidate':groups=[[j] for j in range(4) if answers[j] is not None]
        else:
            for j,a in enumerate(answers):
                if a is None:continue
                existing=next((g for g in groups if answers[g[0]]==a),None)
                if existing is None:groups.append([j])
                else:existing.append(j)
        sizes.append(len(groups))
        weights=np.exp(z[i]-z[i].max(0,keepdims=True));weights/=weights.sum(0,keepdims=True)
        wins=z[i].argmax(0)
        for g in groups:
            idx=np.array(g);others=np.array([j for j in range(4) if j not in g and answers[j] is not None],dtype=int)
            own=z[i,idx];alt=z[i,others] if len(others) else np.zeros((1,z.shape[2]))
            agg=[own.max(0),own.mean(0),own.min(0),own.std(0),own.max(0)-alt.max(0),weights[idx].sum(0),np.isin(wins,idx).astype(float)]
            f=list(np.concatenate(agg))
            f.extend([len(g)/4,len(groups)/4,len(others)/4,float(base[i,idx].max()),float(base[i,idx].mean()),float(base[i,idx].max()-base[i,others].max()) if len(others) else 0.])
            if arithmetic is not None:
                a=arithmetic[i,idx];b=arithmetic[i,others] if len(others) else np.zeros((1,a.shape[1]))
                f.extend(np.r_[a.mean(0),a.min(0),a.max(0),a.mean(0)-b.mean(0)])
            features.append(f);mapping.append((i,g));baseline.append(base[i,idx].max())
    return np.asarray(features,np.float32),np.asarray(baseline),mapping,np.asarray(sizes),meta

def group_labels(y,mapping):return np.asarray([max(y[i,j] for j in g) for i,g in mapping],dtype=int)

def choices(scores,mapping,n):
    result=np.zeros(n,dtype=int);best=np.full(n,-np.inf)
    for score,(i,g) in zip(scores,mapping):
        if score>best[i]:best[i]=score;result[i]=g[0]
    return result

def candidate_scores(group_scores,mapping,base,bank):
    """Pick a stable, strongest original trace inside each numeric-answer group.

    The arithmetic gate reads a candidate text, so arbitrary first-in-bank
    tie-breaking within an answer group would introduce order dependence.
    """
    result=np.full(base.shape,-1e30,dtype=np.float64)
    for value,(i,g) in zip(group_scores,mapping):
        winner=min(g,key=lambda j:(-float(base[i,j]),bank[i]['candidates'][j]['text']))
        result[i,g]=np.nextafter(float(value),-np.inf)
        result[i,winner]=value
    return result

def subset(x,base,target,mapping,sizes,question_ids):
    keep=np.zeros(len(sizes),bool);keep[question_ids]=True
    # Trivial all-same groups teach no ranking signal and are excluded from fit.
    mask=np.array([keep[i] and sizes[i]>1 for i,g in mapping])
    # Feature rows remain in bank order; group sizes must use the same order.
    return x[mask],base[mask],target[mask],sizes[keep & (sizes>1)]
