"""Conservative explicit-consistency gates; no question-specific templates."""
import json,re
import numpy as np
from features import parse_answer,metadata
from round4_quantities import trace
from round4_evidence import program_value,unit_checks,close


def proposals(bank,base_scores,kind):
    meta=metadata(bank);selected=np.where(meta['valid'],base_scores,-1e30).argmax(1)
    targets=selected.copy();reasons=[]
    for i,(row,chosen) in enumerate(zip(bank,selected)):
        candidates=row['candidates'];current=candidates[chosen]['text'];answer=meta['answers'][i][chosen]
        proposal=None;reason=None
        if kind in ('parse_alignment','combined','parse_and_repair'):
            boxes=re.findall(r'\\boxed\{([^{}]+)\}',current)
            displayed=parse_answer('#### '+boxes[-1]) if boxes else None
            if displayed is not None and displayed!=answer:
                eligible=[j for j,a in enumerate(meta['answers'][i]) if a==displayed]
                if eligible:proposal=max(eligible,key=lambda j:base_scores[i,j]);reason='official/displayed answer mismatch'
        if proposal is None and kind in ('units','combined'):
            good,bad=unit_checks(current)
            if bad:
                eligible=[j for j,c in enumerate(candidates) if meta['valid'][i,j] and unit_checks(c['text'])[0]>0 and unit_checks(c['text'])[1]==0]
                if eligible:proposal=max(eligible,key=lambda j:base_scores[i,j]);reason='explicit unit equality contradiction'
        if proposal is None and kind in ('arithmetic_repair','combined','parse_and_repair'):
            t=trace(row['question'],current,repair_subexpressions=True)
            if t['root_found'] and t['bad'] and t['collisions']==0:
                try:v=program_value(t['program'],t['quantities'])
                except (ValueError,OverflowError):v=None
                if v is not None and not close(v,t['answer']):
                    eligible=[]
                    for j,c in enumerate(candidates):
                        a=parse_answer(c['text'])
                        if a is None:continue
                        from fractions import Fraction
                        if not close(v,float(Fraction(a))):continue
                        other=trace(row['question'],c['text'],repair_subexpressions=True)
                        if other['root_found'] and other['bad']==0:
                            try:other_v=program_value(other['program'],other['quantities'])
                            except (ValueError,OverflowError):continue
                            if close(v,other_v):eligible.append(j)
                    if eligible:proposal=max(eligible,key=lambda j:base_scores[i,j]);reason='recomputed dependency chain agrees with another consistent candidate'
        if proposal is not None:targets[i]=proposal
        reasons.append(reason)
    return selected,targets,reasons


def apply_gate(bank,scores,kind,margin_limit):
    old,new,_=proposals(bank,scores,kind)
    gap=scores[np.arange(len(bank)),old]-scores[np.arange(len(bank)),new]
    allowed=(old!=new)&(gap<=margin_limit)
    return np.where(allowed,new,old)


def run():
    from data_access import ROOT,load
    from round4_train import OUT
    b,_,y=load('train');old=np.load(ROOT/'experiments/round3/stack_scores.npz')
    base=old['meta_oof'];split=np.load(OUT/'internal_split.npz');hold=split['holdout']
    kinds=['parse_alignment','units','arithmetic_repair','combined'];limits=[0,.5,1,2,4,8,1e6]
    (OUT/'gate_plan.json').write_text(json.dumps(dict(kinds=kinds,margin_limits=limits,
        selection='TRAIN internal OOF holdout correct; tie uses smaller margin limit',
        implementation_note='strict repair parser excludes Markdown list minus; old semantic feature experiments remain unchanged',test_access=False),indent=2))
    selected=[]
    for kind in kinds:
        current,proposed,reasons=proposals(b,base,kind)
        gap=base[np.arange(len(b)),current]-base[np.arange(len(b)),proposed];records=[]
        for limit in limits:
            choice=np.where((current!=proposed)&(gap<=limit),proposed,current);correct=y[np.arange(len(y)),choice]
            records.append(dict(limit=limit,correct=int(correct[hold].sum()),n=len(hold),changes=int((choice[hold]!=current[hold]).sum())))
        best=max(records,key=lambda r:r['correct']);selected.append(dict(kind=kind,variants=records,best=best));print('GATE INTERNAL',selected[-1],flush=True)
    (OUT/'gate_internal_selection.json').write_text(json.dumps(selected,indent=2))
    bv,_,yv=load('validation');mv=metadata(bv);result=[]
    for sel in selected:
        choice=apply_gate(bv,old['validation'],sel['kind'],sel['best']['limit']);correct=yv[np.arange(len(yv)),choice]
        current,proposed,reasons=proposals(bv,old['validation'],sel['kind']);oldcorrect=yv[np.arange(len(yv)),current]
        rec=dict(kind=sel['kind'],margin_limit=sel['best']['limit'],correct=int(correct.sum()),n=len(yv),
                 accuracy=float(100*correct.mean()),hard_correct=int(correct[mv['hard']].sum()),hard_n=int(mv['hard'].sum()),
                 fixed_old_errors=int(((correct==1)&(oldcorrect==0)).sum()),lost_old_correct=int(((correct==0)&(oldcorrect==1)).sum()),
                 changed=[dict(index=int(i),before=int(current[i]),after=int(choice[i]),reason=reasons[i],correct=int(correct[i])) for i in np.flatnonzero(current!=choice)])
        path=OUT/'gates'/sel['kind'];path.mkdir(parents=True,exist_ok=True)
        np.savez(path/'validation_choices.npz',selected=choice);(path/'result.json').write_text(json.dumps(rec,indent=2));result.append(rec);print('GATE VALIDATION',rec,flush=True)
    (OUT/'gate_comparison.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':run()
