"""One fixed new program feature on the validation-selected frozen policy."""
from data_access import ROOT,CODE,load,parse_answer
from round22_common import read,dump,sha
from round54_selector import infer,select
from round25_selector import scores
from round4_gates import apply_gate
from round46_train import metrics
import numpy as np,shutil,subprocess,sys

OUT=ROOT/'experiments/round54'

if __name__=='__main__':
    assert not (OUT/'final_feature_selection.json').exists()
    for p,h in read(OUT/'plan.json')['source_hashes'].items():assert sha(ROOT/p)==h,p
    previous=read(ROOT/'experiments/round52/final_feature_selection.json');best=previous['best_new'];source=ROOT/best['bundle'];cfg=read(source/'config.json')
    dump(OUT/'frozen_weight_choice.json',dict(source=best,hashes={str(p.relative_to(ROOT)):sha(p) for p in source.rglob('*') if p.is_file()},selection_uses='VAL only before this ablation'))
    b,a,y=load('validation');bs=scores(b,a,source/'baseline');base=apply_gate(b,bs,'parse_and_repair',4)
    s,mask,matches=infer(b,[source/p for p in cfg['policies']],cfg.get('structural',True));c=select(b,bs,s,mask,matches,cfg['mode'])
    conflict=np.asarray([len({parse_answer(v['text']) for v in r['candidates']}-{None})>=2 for r in b]);rec=metrics(y,c,base,conflict)
    bundle=OUT/'bundle_trace_completion';shutil.copytree(source,bundle);cfg['trace_completion']=True;dump(bundle/'config.json',cfg)
    rec.update(arm='trace_completion',bundle=str(bundle.relative_to(ROOT)),asset_bytes=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()),frozen_weight_source=best['arm'])
    p=read(ROOT/'experiments/final_evaluation_20261021/packaging_validation_predictions.json');p['selected']=c.tolist();p.pop('prediction_seconds',None)
    pred=OUT/'validation_trace_completion_predictions.json';dump(pred,p)
    run=subprocess.run([sys.executable,str(CODE/'score_predictions.py'),'--split','validation','--predictions',str(pred),'--output',str(OUT/'validation_official_score.json')],capture_output=True,text=True)
    assert run.returncode==0,(run.stdout,run.stderr);assert read(OUT/'validation_official_score.json')['correct']==rec['correct'];dump(OUT/'validation_result.json',rec)
    records=previous['records']+[rec];winner=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    result=dict(records=records,best_new=winner,selected='round25' if winner['correct']<=752 else winner['arm'],test_target_fixed_before_test=True,test_used_for_selection=False)
    dump(OUT/'final_feature_selection.json',result);print('FINAL_FEATURE_SELECTION',result,flush=True)
