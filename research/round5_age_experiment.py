from data_access import ROOT,CODE,load,read_jsonl,parse_answer
import argparse,json,hashlib,math
from fractions import Fraction
import numpy as np
from round5_age_graph import solve
from round5_experiment import OUT,save,choose,evaluate
from features import metadata

def run(phase):
    if phase=='validation':
        frozen=json.loads((OUT/'age_train_selection.json').read_text());assert frozen['manual_review_complete'];assert frozen['hash']==hashlib.sha256((ROOT/'research/round5_age_graph.py').read_bytes()).hexdigest()
    b,_,y=load(phase);refs=read_jsonl(CODE/'data'/f'gsm8k_{phase}.jsonl');s=np.load(ROOT/'experiments/round3/stack_scores.npz')['meta_oof' if phase=='train' else 'validation'];old=choose(b,s);new=old.copy();audit=[]
    for i,row in enumerate(b):
        proof=solve(row['question'])
        if proof is None:continue
        good=math.isclose(proof['value'],float(Fraction(parse_answer(refs[i]['answer']))),rel_tol=1e-6,abs_tol=1e-6)
        audit.append(dict(index=i,question=row['question'],reference=refs[i]['answer'],proof=proof,reference_match=good))
        choices=[j for j,c in enumerate(row['candidates']) if parse_answer(c['text']) is not None and math.isclose(float(Fraction(parse_answer(c['text']))),proof['value'],rel_tol=1e-6,abs_tol=1e-6)]
        if choices:new[i]=max(choices,key=lambda j:s[i,j])
    save(f'age_{phase}_audit.json',audit);m=metadata(b);gap=s[np.arange(len(b)),old]-s[np.arange(len(b)),new]
    if phase=='train':
        hold=np.load(ROOT/'experiments/round4/internal_split.npz')['holdout'];variants=[]
        for lim in [None,.5,1,2,4,8]:
            ch=old if lim is None else np.where(gap<=lim,new,old);variants.append(dict(limit=lim,metrics=evaluate(ch[hold],y[hold],old[hold],m['hard'][hold])))
        best=max(variants,key=lambda a:a['metrics']['correct']);r=dict(coverage=len(audit),reference_matches=sum(a['reference_match'] for a in audit),variants=variants,best=best,manual_review_complete=False,hash=hashlib.sha256((ROOT/'research/round5_age_graph.py').read_bytes()).hexdigest());save('age_train_selection.json',r)
    else:
        lim=frozen['best']['limit'];ch=old if lim is None else np.where(gap<=lim,new,old)
        r=dict(coverage=len(audit),reference_matches=sum(a['reference_match'] for a in audit),train_selected_limit=lim,metrics=evaluate(ch,y,old,m['hard']),changed=[dict(index=int(i),before=int(old[i]),after=int(ch[i]),correct=int(y[i,ch[i]])) for i in np.flatnonzero(old!=ch)]);save('age_validation_comparison.json',r)
    print(json.dumps(r),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','validation']);run(p.parse_args().phase)
