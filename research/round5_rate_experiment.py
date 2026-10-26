from data_access import ROOT,CODE,load,read_jsonl,parse_answer
import argparse,json,math,hashlib
from fractions import Fraction
import numpy as np
from round5_rate_solver import proposals
from round5_experiment import OUT,save,choose,evaluate
from features import metadata

def train():
    b,_,y=load('train');s=np.load(ROOT/'experiments/round3/stack_scores.npz')['meta_oof'];old=choose(b,s);new,proofs=proposals(b,s,old);refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');audit=[]
    for i,p in enumerate(proofs):
        if p is not None:audit.append(dict(index=i,question=b[i]['question'],reference=refs[i]['answer'],proof=p,reference_match=math.isclose(p['value'],float(Fraction(parse_answer(refs[i]['answer']))),rel_tol=1e-6,abs_tol=1e-6)))
    save('rate_train_audit.json',audit)
    hold=np.load(ROOT/'experiments/round4/internal_split.npz')['holdout'];m=metadata(b);gap=s[np.arange(len(b)),old]-s[np.arange(len(b)),new]
    variants=[dict(limit=None,metrics=evaluate(old[hold],y[hold],old[hold],m['hard'][hold]))]
    for limit in [.5,1,2,4,8]:
        ch=np.where(gap<=limit,new,old);variants.append(dict(limit=limit,metrics=evaluate(ch[hold],y[hold],old[hold],m['hard'][hold])))
    best=max(variants,key=lambda r:r['metrics']['correct'])
    record=dict(coverage=len(audit),reference_matches=sum(r['reference_match'] for r in audit),variants=variants,best=best,source_sha256=hashlib.sha256((ROOT/'research/round5_rate_solver.py').read_bytes()).hexdigest())
    save('rate_train_selection.json',record);print(json.dumps(record),flush=True)
def validation():
    assert (OUT/'manual_quality_review.json').exists()
    r=json.loads((OUT/'rate_train_selection.json').read_text());assert r['source_sha256']==hashlib.sha256((ROOT/'research/round5_rate_solver.py').read_bytes()).hexdigest()
    b,_,y=load('validation');s=np.load(ROOT/'experiments/round3/stack_scores.npz')['validation'];old=choose(b,s);new,proofs=proposals(b,s,old);limit=r['best']['limit'];gap=s[np.arange(len(b)),old]-s[np.arange(len(b)),new];ch=old if limit is None else np.where(gap<=limit,new,old)
    rec=dict(train_selected_limit=limit,coverage=sum(p is not None for p in proofs),metrics=evaluate(ch,y,old,metadata(b)['hard']),changed=[dict(index=int(i),before=int(old[i]),after=int(ch[i]),correct=int(y[i,ch[i]]),proof=proofs[i]) for i in np.flatnonzero(ch!=old)])
    save('rate_validation_comparison.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','validation']);a=p.parse_args();train() if a.phase=='train' else validation()
