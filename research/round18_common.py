"""Train-bank-only grouped comparison. No reference reasoning supervision."""
from data_access import ROOT,CODE,read_jsonl,parse_answer
from pathlib import Path
import sys,os,json,hashlib
import numpy as np
OUT=ROOT/'experiments/round18'

def no_validation(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0]).replace('\\','/').lower()
        if p.endswith(('gsm8k_validation.jsonl','gsm8k_validation.npz')):
            raise PermissionError('Round18 must establish train-side evidence before validation.')
sys.addaudithook(no_validation)

def read_plan():return json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
def inputs():return read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl')
def load_part(name,embeddings=True):
    plan=read_plan()
    if name=='audit':assert (OUT/'frozen.json').exists(),'Audit is sealed until all selections freeze.'
    ids=np.array(plan['partitions'][name],dtype=int);bank=inputs();rows=[bank[i] for i in ids]
    # Only final numbers are exposed to model fitting; reference prose discarded.
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    for i in ids:
        assert bank[i]['id']==refs[i]['id'] and bank[i]['question']==refs[i]['question']
        assert parse_answer(refs[i]['answer']) is not None
    labels=np.asarray([[parse_answer(c['text'])==parse_answer(refs[i]['answer']) for c in bank[i]['candidates']] for i in ids],np.float32)
    assert np.all(labels.sum(1)>0)
    arrays={}
    if embeddings:
        with np.load(CODE/'data/candidates/gsm8k_train.npz',allow_pickle=False) as z:
            e=z['embeddings']
            assert e.shape==(len(bank)*4,1024)
            arrays={'embeddings':e.reshape(len(bank),4,1024)[ids].copy()}
            assert np.isfinite(arrays['embeddings']).all()
    return rows,arrays,labels,ids

def normalized(s):return (s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-5)
def paired(choice,base,y,hard):
    a=y[np.arange(len(y)),choice].astype(bool);b=y[np.arange(len(y)),base].astype(bool)
    change=choice!=base
    return dict(n=len(y),correct=int(a.sum()),accuracy=float(a.mean()),hard_n=int(hard.sum()),hard_correct=int(a[hard].sum()),
                repaired=int((a&~b).sum()),damaged=int((~a&b).sum()),net=int(a.sum()-b.sum()),
                changed=int(change.sum()),wrong_to_wrong=int((change&~a&~b).sum()),right_to_right=int((change&a&b).sum()))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare():
    from round14_prepare import family
    from round15_prepare import exclusions
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'plan.json').exists()
    bank=inputs();seen=exclusions()
    for rnd in (15,17):seen|={int(i) for i in json.loads((ROOT/f'experiments/round{rnd}/plan.json').read_text(encoding='utf-8'))['groups']}
    seen|=set(json.loads((ROOT/'experiments/round16/plan.json').read_text())['fresh_audit_parents'])
    groups={}
    for i,r in enumerate(bank):groups.setdefault(sha_text(family(r['question'])),[]).append(i)
    excluded={k for k,v in groups.items() if set(v)&seen}
    ordered=sorted(groups,key=lambda k:sha_text('r18-audit-20261015:'+k));audit=[];audit_groups=set()
    for k in ordered:
        if k in excluded:continue
        audit+=groups[k];audit_groups.add(k)
        if len(audit)>=950:break
    remaining=[k for k in groups if k not in audit_groups]
    development=[];dev_groups=set()
    for k in sorted(remaining,key=lambda k:sha_text('r18-development:'+k)):
        development+=groups[k];dev_groups.add(k)
        if len(development)>=700:break
    fit=[i for k,v in groups.items() if k not in audit_groups|dev_groups for i in v]
    plan=dict(seed=18018,partitions=dict(fit=sorted(fit),development=sorted(development),audit=sorted(audit)),
        group_by='normalized question template; all four candidates kept together',group_hashes={str(i):k for k,v in groups.items() for i in v},
        audit_excludes_previously_manually_annotated_questions=True,
        caveat='Fresh fitting isolation conditional on historically designed architectures; existing project exploration saw TRAIN data. Not an untouched external benchmark.',
        baseline='Same five member recipes; all weights/vocabulary/scalers fit only on fit partition; stack coefficients fit only on development. Fixed historical arithmetic gate.',
        comparator=dict(vocab_limit=4096,width=32,question_tokens=96,candidate_tokens=160,batch=32,epochs=24,patience=8,
            lr=.001,weight_decay=.01,dropout=.15,embedding_projection=32,initialization='random; supplied fixed embeddings are allowed inputs',
            modes=['joint','independent'],alpha=[.1,.25,.5,1.],selection='development best fused correct, tie fewer changes, smaller alpha, smaller pair loss; same fit/development for both modes'),
        audit_primary='development-selected joint fusion; independent ablation and standalone joint are secondary fixed diagnostics',
        gate=dict(min_net=5,paired_one_sided_p_max=.05,require_positive_bootstrap_lower=True,require_dev_gain=True),
        reference_step_supervision=False,synthetic_data=False,validation_allowed=False,test_access=False)
    (OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({k:len(v) for k,v in plan['partitions'].items()}))
def sha_text(text):return hashlib.sha256(text.encode()).hexdigest()
if __name__=='__main__':prepare()
