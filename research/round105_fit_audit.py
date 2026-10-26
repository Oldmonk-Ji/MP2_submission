"""TRAIN-fit audit before replacing ambiguous positive roots by reference evidence."""
from round22_common import ROOT,load,read,dump,sha
from round101_templates import canonical,execute
from round29_programs import quantities
from features import parse_answer
from fractions import Fraction as F
from collections import Counter,defaultdict
import numpy as np

if __name__=='__main__':
    out=ROOT/'experiments/round105';out.mkdir();b,a,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)};packets=read(ROOT/'experiments/round102/train_programs.json')
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:folds=z['folds']
    excluded={r['train_index'] for r in read(ROOT/'experiments/round56/held_source_review_before_reference.json')};sources=[dict(r,source='strict_reference') for r in read(ROOT/'experiments/round63/accepted_reference_programs.json')]+[dict(r,source='reviewed') for r in read(ROOT/'experiments/round51/reviewed_programs.json')];accepted=defaultdict(dict);counts=Counter();rejections=[]
    for r in sources:
        idx=r['train_index'];i=index[idx]
        if idx in excluded or folds[i]==2:continue
        q=b[i]['question'];truth={F(parse_answer(b[i]['candidates'][j]['text'])) for j in np.flatnonzero(y[i])};values=[F(x['value']) for x in quantities(q)]
        try:
            p=canonical(r['program']);value=execute(p,values);okay=value in truth
        except (ValueError,ZeroDivisionError,IndexError):okay=False
        counts['source_programs']+=1
        if not okay:
            counts['rejected_incompatible_or_unsupported']+=1
            if len(rejections)<10:rejections.append(dict(train_index=idx,question=q,source=r['source'],program=r['program']))
            continue
        accepted[idx][repr(p)]=r['program'];counts['accepted_programs']+=1
    examples=[]
    for i in np.flatnonzero(folds!=2):
        if y[i].min()==y[i].max():continue
        counts['mixed_fit_parents']+=1;idx=int(ids[i]);pos=np.flatnonzero(y[i]);ambiguous=any(len(packets[i][j])>1 for j in pos)
        counts['ambiguous_positive_parents']+=ambiguous
        if idx in accepted:
            counts['reference_covered_mixed_parents']+=1;counts['reference_covered_ambiguous_positive_parents']+=ambiguous
            candidates=set()
            for j in pos:
                for p in packets[i][j]:
                    try:candidates.add(repr(canonical(p)))
                    except ValueError:pass
            shared=bool(candidates&set(accepted[idx]));counts['reference_template_present_in_correct_candidate_roots']+=shared
            if ambiguous and len(examples)<8:examples.append(dict(train_index=idx,question=b[i]['question'],positive_root_counts=[len(packets[i][j]) for j in pos],reference_template_matches=shared,reference_templates=list(accepted[idx])))
    report=dict(counts=dict(counts),reference_parents=len(accepted),examples=examples,rejected_examples=rejections,fit_only=True,test_opened=False)
    dump(out/'fit_audit.json',report);dump(out/'audit_freeze.json',dict(source_sha256=sha(__file__),audit_sha256=sha(out/'fit_audit.json'),predeclared_plan_sha256=sha(ROOT/'experiments/next_after_round104_predeclared.json'),test_opened=False));print(dict(counts),flush=True)
