"""Calibrate rare-rule thresholds on all TRAIN OOF events, retaining earlier audit."""
from data_access import ROOT,load
import json
import numpy as np
from round4_train import OUT
from round4_gates import proposals,apply_gate
from features import metadata


def main():
    kinds=['parse_alignment','arithmetic_repair','parse_and_repair','units'];limits=[0,.5,1,2,4,8,1e6]
    (OUT/'rare_gate_plan.json').write_text(json.dumps(dict(kinds=kinds,margin_limits=limits,
        rationale='951-question internal holdout contained zero active parsing/repair cases; fit rare-rule thresholds on all TRAIN OOF events',
        selection='maximize TRAIN OOF correct, ties prefer smaller margin limit; official validation not used to fit thresholds',
        caveat='OOF threshold fit performance is training performance, not an independent accuracy estimate',test_access=False),indent=2))
    b,_,y=load('train');base=np.load(ROOT/'experiments/round3/stack_scores.npz');s=base['meta_oof'];records=[]
    for kind in kinds:
        current,proposed,reasons=proposals(b,s,kind);gap=s[np.arange(len(b)),current]-s[np.arange(len(b)),proposed]
        oldcorrect=y[np.arange(len(y)),current];variants=[]
        for limit in limits:
            choice=np.where((current!=proposed)&(gap<=limit),proposed,current);correct=y[np.arange(len(y)),choice]
            variants.append(dict(limit=limit,correct=int(correct.sum()),n=len(y),changes=int((choice!=current).sum()),
                fixed=int(((correct==1)&(oldcorrect==0)).sum()),lost=int(((correct==0)&(oldcorrect==1)).sum())))
        best=max(variants,key=lambda r:r['correct']);records.append(dict(kind=kind,variants=variants,best=best));print(records[-1],flush=True)
    chosen=max(records,key=lambda r:r['best']['correct'])
    (OUT/'rare_gate_selection.json').write_text(json.dumps(dict(records=records,chosen=chosen),indent=2))
    bv,_,yv=load('validation');mv=metadata(bv);result=[]
    for record in records:
        kind=record['kind'];limit=record['best']['limit'];choice=apply_gate(bv,base['validation'],kind,limit)
        correct=yv[np.arange(len(yv)),choice];current,_,reasons=proposals(bv,base['validation'],kind);oldcorrect=yv[np.arange(len(yv)),current]
        rec=dict(kind=kind,margin_limit=limit,train_selected=kind==chosen['kind'],correct=int(correct.sum()),n=len(yv),
            accuracy=float(100*correct.mean()),hard_correct=int(correct[mv['hard']].sum()),hard_n=int(mv['hard'].sum()),
            fixed_old_errors=int(((correct==1)&(oldcorrect==0)).sum()),lost_old_correct=int(((correct==0)&(oldcorrect==1)).sum()),
            changed=[dict(index=int(i),before=int(current[i]),after=int(choice[i]),reason=reasons[i],correct=int(correct[i])) for i in np.flatnonzero(current!=choice)])
        dest=OUT/'rare_gates'/kind;dest.mkdir(parents=True,exist_ok=True)
        np.savez(dest/'validation_choices.npz',selected=choice);(dest/'config.json').write_text(json.dumps(dict(kind=kind,margin_limit=limit),indent=2))
        (dest/'result.json').write_text(json.dumps(rec,indent=2));result.append(rec);print('VALIDATION',rec,flush=True)
    (OUT/'rare_gate_comparison.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
