"""Round 19: predeclared, question-grouped, TRAIN-only method development."""
from data_access import ROOT, CODE, read_jsonl, parse_answer
from pathlib import Path
import os, sys, json, hashlib
import numpy as np

OUT=ROOT/'experiments/round19'
VALIDATION_ALLOWED=False


def no_validation(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0]).replace('\\','/').lower()
        if not VALIDATION_ALLOWED and p.endswith(('gsm8k_validation.jsonl','gsm8k_validation.npz')):
            raise PermissionError('Round19 validation stays closed until train-side gate and full-fit freeze.')
sys.addaudithook(no_validation)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def read_plan():return json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
def normalized(s):return (s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-5)


def load_part(name,embeddings=True):
    assert name in ('fit','development','audit','all_train')
    if name=='audit':assert (OUT/'frozen.json').exists(), 'Freeze before unsealing audit.'
    ids=np.arange(4755) if name=='all_train' else np.asarray(read_plan()['partitions'][name],int)
    if name=='all_train':assert json.loads((OUT/'audit_result.json').read_text())['train_side_gate_passed']
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl');refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    rows=[bank[i] for i in ids];labels=[]
    for i in ids:
        assert bank[i]['id']==refs[i]['id'] and bank[i]['question']==refs[i]['question']
        answer=parse_answer(refs[i]['answer']);assert answer is not None
        labels.append([parse_answer(c['text'])==answer for c in bank[i]['candidates']])
    y=np.asarray(labels,np.float32);assert (y.sum(1)>0).all()
    a={}
    if embeddings:
        with np.load(CODE/'data/candidates/gsm8k_train.npz',allow_pickle=False) as z:
            e=z['embeddings'];assert e.shape==(len(bank)*4,1024)
            a={'embeddings':e.reshape(len(bank),4,1024)[ids].copy()}
    return rows,a,y,ids


def paired(choice,base,y,hard):
    a=y[np.arange(len(y)),choice].astype(bool);b=y[np.arange(len(y)),base].astype(bool);change=choice!=base
    return dict(n=len(y),correct=int(a.sum()),accuracy=float(a.mean()),hard_n=int(hard.sum()),hard_correct=int(a[hard].sum()),
                repaired=int((a&~b).sum()),damaged=int((~a&b).sum()),net=int(a.sum()-b.sum()),changed=int(change.sum()),
                wrong_to_wrong=int((change&~a&~b).sum()),right_to_right=int((change&a&b).sum()))


def prepare():
    from round14_prepare import family
    from round15_prepare import exclusions
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'plan.json').exists()
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl')
    old=json.loads((ROOT/'experiments/round18/plan.json').read_text());seen=set(old['partitions']['audit'])|exclusions()
    for r in (15,17):seen|={int(i) for i in json.loads((ROOT/f'experiments/round{r}/plan.json').read_text())['groups']}
    seen|=set(json.loads((ROOT/'experiments/round16/plan.json').read_text())['fresh_audit_parents'])
    groups={}
    for i,row in enumerate(bank):groups.setdefault(digest(family(row['question'])),[]).append(i)
    audited=set();audit=[]
    for k in sorted(groups,key=lambda k:digest('round19-audit-20261015:'+k)):
        if set(groups[k])&seen:continue
        audit+=groups[k];audited.add(k)
        if len(audit)>=950:break
    dev=[];devg=set()
    for k in sorted((k for k in groups if k not in audited),key=lambda k:digest('round19-dev:'+k)):
        dev+=groups[k];devg.add(k)
        if len(dev)>=650:break
    fit=[i for k,v in groups.items() if k not in audited|devg for i in v]
    # Strictly separate the already examined round18 audit from all new evaluation.
    old_audit=set(old['partitions']['audit'])
    moved=[i for i in dev if i in old_audit]
    if moved:
        dev=[i for i in dev if i not in old_audit];fit+=moved
        need=650-len(dev)
        candidates=sorted((i for i in fit if i not in old_audit),key=lambda i:digest('round19-dev-fill:'+str(i)))
        # Whole normalized-template groups are moved together.
        chosen=set();key_by_i={i:k for k,v in groups.items() for i in v}
        for i in candidates:
            k=key_by_i[i]
            if k in chosen:continue
            members=[j for j in groups[k] if j in fit]
            if set(members)&old_audit:continue
            chosen.add(k);dev+=members;fit=[j for j in fit if j not in members]
            if len(dev)>=650:break
    configs=[dict(name='legacy_short',width=32,vocab_limit=4096,question_tokens=96,candidate_tokens=160,
                  dropout=.15,weight_decay=.01,lr=.001,ema=0.,smooth=0.,mode='joint'),
             dict(name='regularized_short',width=16,vocab_limit=2048,question_tokens=160,candidate_tokens=160,
                  dropout=.35,weight_decay=.15,lr=.0005,ema=.95,smooth=.05,mode='joint'),
             dict(name='regularized_long',width=16,vocab_limit=2048,question_tokens=160,candidate_tokens=512,
                  dropout=.35,weight_decay=.15,lr=.0005,ema=.95,smooth=.05,mode='joint'),
             dict(name='regularized_independent',width=16,vocab_limit=2048,question_tokens=160,candidate_tokens=512,
                  dropout=.35,weight_decay=.15,lr=.0005,ema=.95,smooth=.05,mode='independent')]
    for c in configs:c.update(embedding_projection=16 if c['width']==16 else 32,batch=16,epochs=32,patience=6,seed=19019)
    p=dict(seed=19019,partitions=dict(fit=sorted(fit),development=sorted(dev),audit=sorted(audit)),
           group_hashes={str(i):k for k,v in groups.items() for i in v},
           historical_audit_excluded_from_new_development_and_audit=True,
           caveat='Fresh per-round fitting isolation, conditional on all historical TRAIN development. Not untouched external evidence. Consumed round18 audit used as training data only.',
           baseline='Refit the same five members on fit; fixed C=0.1 stack trained on development; same arithmetic gate.',
           neural=configs,neural_checkpoint='minimum development unsmoothed pair+0.25 group loss; not tiny fused-accuracy fluctuations',
           kernel=dict(pca=96,anchors=256,gamma_factors=[.3,1.,3.],C=[.01,.1,1.,10.],
                       features='PCA embedding, within-question differences and products, bounded input-only numeric/style features',
                       linear_control_C=[.01,.1,1.,10.]),
           alphas=[.1,.25,.5,1.],
           primary_candidates=['regularized_short','regularized_long','regularized_independent','kernel'],
           selection='Best development fused correct; ties fewer harms, fewer changed choices, smaller alpha, smaller inference assets. One primary frozen before audit.',
           gate=dict(min_net=5,paired_one_sided_p_max=.05,require_positive_bootstrap_lower=True,require_dev_gain=True),
           primary_tests=1,bootstrap_replicates=20000,reference_step_supervision=False,synthetic_data=False,
           validation_policy='Only after train-side gate and full-fit freeze; single fixed comparison. No TEST access.',test_access=False,
           target_validation_correct_for_95=759)
    for a in p['partitions']:
        for b in p['partitions']:
            if a!=b:assert not {p['group_hashes'][str(i)] for i in p['partitions'][a]}&{p['group_hashes'][str(i)] for i in p['partitions'][b]}
    assert len(set(fit+dev+audit))==4755 and not old_audit&set(dev+audit)
    dump(OUT/'plan.json',p);print(json.dumps({k:len(v) for k,v in p['partitions'].items()}))


if __name__=='__main__':prepare()
