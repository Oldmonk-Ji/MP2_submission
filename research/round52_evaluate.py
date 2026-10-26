"""Prespecified feature validation: full VAL selection before final TEST freeze."""
from data_access import ROOT,load,parse_answer
from round22_common import read,dump,sha
from round52_selector import infer,select
from round52_local import focus,csr
import round48_evaluate,round46_evaluate
from round46_train import metrics
from round25_selector import scores
from round4_gates import apply_gate
import numpy as np,shutil

OUT=ROOT/'experiments/round52'

if __name__=='__main__':
    round48_evaluate.OUT=OUT;round48_evaluate.focus=focus;round48_evaluate.csr=csr;round48_evaluate.held_relations()
    round46_evaluate.OUT=OUT;round46_evaluate.infer=infer;round46_evaluate.select=select;round46_evaluate.main()
    b,a,y=load('validation');bs=scores(b,a,ROOT/'experiments/round25/selected_bundle');base=apply_gate(b,bs,'parse_and_repair',4)
    paths=[ROOT/'experiments/round50/full'/f'exact_s{s}' for s in (48048,48049)]
    s,mask,matches=infer(b,paths,False);c=select(b,bs,s,mask,matches)
    conflict=np.asarray([len({parse_answer(v['text']) for v in r['candidates']}-{None})>=2 for r in b])
    record=metrics(y,c,base,conflict);record.update(arm='coverage_only',structural=False,weights='frozen round50 exact')
    bundle=OUT/'bundle_coverage_only';shutil.copytree(ROOT/'experiments/round50/bundle_exact',bundle)
    cfg=read(bundle/'config.json');cfg.update(arm='coverage_only',structural=False);dump(bundle/'config.json',cfg)
    record['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    p=read(ROOT/'experiments/final_evaluation_20261021/packaging_validation_predictions.json');p['selected']=c.tolist();p.pop('prediction_seconds',None)
    dump(OUT/'validation_coverage_only_predictions.json',p);dump(OUT/'coverage_only_ablation.json',record)
    # Its independent official score is reproduced by the final feature workflow.
    records=read(OUT/'validation_result.json')+[record];best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'feature_selection.json',dict(records=records,best_new=best,selected='round25' if best['correct']<=752 else best['arm'],
        test_target=best['arm'],test_target_fixed_before_test=True,test_used_for_selection=False))
    print('FEATURE_SELECTION',read(OUT/'feature_selection.json'),flush=True)
