"""One matched feature ablation, selected using validation before any TEST."""
from data_access import ROOT,CODE
from round22_common import read,dump
from round52_selector import infer,select
import round46_evaluate
import numpy as np,subprocess,sys
from round48_local import focus,csr,node_scores
from round44_programs import edits,valid
from collections import Counter

if __name__=='__main__':
    out=ROOT/'experiments/round53';round46_evaluate.OUT=out
    round46_evaluate.infer=lambda bank,paths:infer(bank,paths,False)
    round46_evaluate.select=select;round46_evaluate.main()
    cfg=read(out/'bundle_sft/config.json');cfg['structural']=False;dump(out/'bundle_sft/config.json',cfg)
    pairs=[]
    for r in read(out/'held_source_review_before_reference.json'):
        counts=Counter();seen=set()
        for kind,p,d in edits(r['program'],r['quantities']):
            a=valid(p,r['quantities']);key=repr(p)
            if a is None or a==r['source_answer'] or key in seen or counts[kind]>=4:continue
            counts[kind]+=1;seen.add(key);pairs.append(dict(question=r['question'],positive=r['program'],negative=p,decision=d))
    ff=[focus(p) for p in pairs];xp=csr([p[0] for p in ff]);xn=csr([p[1] for p in ff]);m=[]
    for seed in (48048,48049):
        with np.load(out/'diagnostic'/f'sft_s{seed}'/'model.npz') as w:m.append(node_scores(xp,w)-node_scores(xn,w))
    dump(out/'held_relation_results.json',dict(n=len(pairs),correct=int((np.mean(m,axis=0)>0).sum()),arm='lexical_sft',test_opened=False))
    selection=read(ROOT/'experiments/round52/feature_selection.json');record=read(out/'validation_result.json')[0]
    coverage=ROOT/'experiments/round52'
    run=subprocess.run([sys.executable,str(CODE/'score_predictions.py'),'--split','validation','--predictions',str(coverage/'validation_coverage_only_predictions.json'),'--output',str(coverage/'validation_coverage_only_official_score.json')],capture_output=True,text=True)
    assert run.returncode==0,(run.stdout,run.stderr)
    assert read(coverage/'validation_coverage_only_official_score.json')['correct']==read(coverage/'coverage_only_ablation.json')['correct']
    record['asset_bytes']=sum(p.stat().st_size for p in (out/'bundle_sft').rglob('*') if p.is_file())
    record=dict(record,arm='lexical_sft',bundle='experiments/round53/bundle_sft')
    records=[dict(r,bundle='experiments/round52/bundle_'+r['arm']) for r in selection['records']]+[record]
    best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    result=dict(records=records,best_new=best,selected='round25' if best['correct']<=752 else best['arm'],test_target=best['arm'],test_target_fixed_before_test=True,test_used_for_selection=False)
    dump(ROOT/'experiments/round52/final_feature_selection.json',result);print('FINAL_FEATURE_SELECTION',result,flush=True)
