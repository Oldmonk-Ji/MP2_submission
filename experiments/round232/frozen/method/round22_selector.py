"""Input-only readout of the same six experts; no split IDs or reference access.

CPU inference reuses common feature extraction. All learned parameters reside in
the supplied bundle. The round20 experts themselves are unchanged.
"""
from pathlib import Path
import json
import re
import numpy as np
from scipy.special import softmax, logsumexp
from features import prepare, metadata
from round3_features import build, grounding, lexical_documents
from round4_gates import apply_gate

CUES = [r'\$|\bcost|\bprice|\bprofit', r'\beach\b|\bper\b',
        r'\bafter\b|\bbefore\b|\bnow\b|\bago\b|\blater\b', r'%|\bpercent',
        r'\bmore\b|\bless\b|\bfewer\b|\bdifference\b', r'\btotal\b|\ball\b|\btogether\b',
        r'\bleft\b|\bremain|\binitial', r'\bpack|\bbox|\bbag|\bdozen|\bpair',
        r'\bminute|\bhour|\bday|\bweek|\bmonth|\byear', r'\btwice\b|\bhalf\b|\btimes\b',
        r'\barea\b|\blength\b|\bwidth\b|\bsquare\b', r'\baverage\b|\bcheapest\b|\bat least\b']


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def normalized(s): return (s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-5)


def dense_forward(x, m):
    h=(x.reshape(-1,x.shape[-1])-m['mean'])/m['scale']
    for i in range(int(m['layers'])):
        h=h@m[f'w{i}']+m[f'b{i}']
        if i<int(m['layers'])-1: h=np.maximum(h,0)
    return h.reshape(-1,4)


def component_scores(bank, arrays, path):
    """Same equations as round3_selector/round19_kernel, sharing prepared data."""
    path=Path(path)
    p=prepare(bank,arrays); g=grounding(bank); dense={}; documents={}; ss=[]
    for j in range(5):
        member=path/'baseline'/f'member{j}'; cfg=read(member/'config.json')
        if cfg['family'] in ('mlp','joint'):
            kind=cfg['features']
            if kind not in dense:dense[kind]=build(bank,arrays,kind,prepared=p,ground=g)
            x=dense[kind]
        with np.load(member/'model.npz',allow_pickle=False) as m:
            if cfg['family']=='mlp':s=dense_forward(x,m)
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                if cfg['mode'] not in documents:documents[cfg['mode']]=lexical_documents(bank,cfg['mode'])
                vec=TfidfVectorizer(vocabulary=read(member/'vocabulary.json'),ngram_range=(1,2),sublinear_tf=True,dtype=np.float32)
                vec.idf_=m['idf']; sx=vec.transform(documents[cfg['mode']])
                if cfg['family']=='text':s=np.asarray(sx@m['weight']).reshape(-1,4)
                else:
                    t=sx@m['lexical']*4
                    if cfg.get('question'):
                        qb=[dict(question=r['question'],candidates=[dict(text=r['question'])]) for r in bank]
                        q=np.repeat(vec.transform(lexical_documents(qb))@m['lexical']*4,4,axis=0)
                        t=np.concatenate([t,q,t*q,t-q],1)
                    h=np.concatenate([(x.reshape(-1,x.shape[-1])-m['mean'])/m['scale'],t],1)
                    for k in range(int(m['layers'])):
                        h=h@m[f'w{k}']+m[f'b{k}']
                        if k<int(m['layers'])-1:h=np.maximum(h,0)
                    s=h.reshape(-1,4)
            ss.append(s)
    with np.load(path/'kernel/space.npz',allow_pickle=False) as z:space={k:z[k] for k in z.files}
    projected=((p['embedding']-space['pca_mean'])@space['components'].T)/space['pca_scale']
    avg=projected.mean(1,keepdims=True)
    small=np.concatenate([p['base'],p['numeric'][:,:,5:],p['checks2']],2)
    kx=np.concatenate([projected,projected-avg,projected*avg,small],2).reshape(len(bank)*4,-1).astype(np.float32)
    kx=np.clip((kx-space['x_mean'])/space['x_scale'],-8,8)*space['block_scale']
    cfg=read(path/'kernel/fixed/config.json'); basis=space['anchors']
    dist=np.maximum((kx**2).sum(1)[:,None]+(basis**2).sum(1)[None,:]-2*kx@basis.T,0)
    kx=np.exp(-cfg['gamma_factor']*dist/float(space['distance_median'])).astype(np.float32)
    with np.load(path/'kernel/fixed/model.npz',allow_pickle=False) as m:
        ks=((kx-m['mean'])/m['scale'])@m['weight']
    ss.append(ks.reshape(-1,4))
    return np.stack(ss,2), dict(prepared=p,ground=g)


def base_scores(components, base_weights, kernel_alpha=.1):
    z=normalized(components)
    return z[:,:,:5]@np.asarray(base_weights)+kernel_alpha*z[:,:,5]


def side_features(bank, extra=None):
    cues=np.array([[bool(re.search(c,r['question'].lower())) for c in CUES] for r in bank],np.float32)
    # Arithmetic/grounding are evidence, not hard truth labels or semantic proofs.
    if extra is None:
        from features import checked_arithmetic
        check=np.array([[checked_arithmetic(c['text']) for c in r['candidates']] for r in bank],np.float32)
        g=grounding(bank)
    else:check=extra['prepared']['checks2'];g=extra['ground']
    evidence=np.concatenate([check,g[:,:,:12]],2)
    return cues,evidence


def linear_design(z,cues,evidence,kind):
    if kind=='linear':return z
    interaction=(z[:,:,:,None]*cues[:,None,None,:]).reshape(len(z),4,-1)
    return np.concatenate([z,interaction,evidence,evidence-evidence.mean(1,keepdims=True)],2)


def group_inputs(bank):
    meta=metadata(bank); groups=np.full((len(bank),4),-1,np.int64); masks=np.zeros((len(bank),4),bool)
    for i,answers in enumerate(meta['answers']):
        lookup={}
        for j,a in enumerate(answers):
            if a is None:continue
            if a not in lookup:lookup[a]=len(lookup)
            groups[i,j]=lookup[a];masks[i,lookup[a]]=True
    return groups,masks,meta['valid']


def probability_expert_logits(z, groups, valid, mode, temperatures):
    v=z/np.asarray(temperatures)[None,None,:]
    if mode=='prob_candidate':return np.where(valid[:,:,None],v,-1e9)
    result=np.full_like(v,-1e9,dtype=float)
    for k in range(4):
        keep=groups==k
        if mode=='prob_group_mean':
            a=logsumexp(np.where(keep[:,:,None],v,-1e9),axis=1)-np.log(np.maximum(keep.sum(1),1))[:,None]
        else:a=np.where(keep[:,:,None],v,-1e9).max(1)
        result[:,k]=np.where(keep.any(1)[:,None],a,-1e9)
    return result


def adjustment_scores(bank, components, config, cues=None, evidence=None):
    z=normalized(components); family=config['family']
    if family in ('linear','conditional'):
        if family=='conditional' and cues is None:cues,evidence=side_features(bank)
        x=linear_design(z,cues,evidence,family)
        return ((x-np.asarray(config['mean']))/np.asarray(config['scale']))@np.asarray(config['weights'])
    groups,group_mask,valid=group_inputs(bank)
    logits=probability_expert_logits(z,groups,valid,family,config['temperatures'])
    probs=(softmax(logits,axis=1)*np.asarray(config['weights'])[None,None,:]).sum(2)
    lp=np.log(np.maximum(probs,1e-12))
    if family=='prob_candidate':return lp
    return np.take_along_axis(lp,np.maximum(groups,0),axis=1)


def fused_scores(bank, components, cfg, cues=None, evidence=None):
    base=base_scores(components,cfg['base_weights'],cfg['kernel_alpha'])
    if cfg['alpha']==0:return base
    adj=adjustment_scores(bank,components,cfg['readout'],cues,evidence)
    valid=metadata(bank)['valid']
    # Keep invalid-answer placeholders out of standard-deviation calculations.
    for i in range(len(adj)):
        if valid[i].any():adj[i,~valid[i]]=adj[i,valid[i]].mean()
    return base+cfg['alpha']*normalized(adj)


def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=read(path/'readout.json')
    components,extra=component_scores(bank,arrays,path)
    cues,evidence=side_features(bank,extra) if cfg['readout']['family']=='conditional' else (None,None)
    scores=fused_scores(bank,components,cfg,cues,evidence)
    return apply_gate(bank,scores,cfg['gate'],cfg['gate_margin']).astype(int).tolist()
