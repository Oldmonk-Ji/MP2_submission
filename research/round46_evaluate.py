"""One frozen validation evaluation per prespecified arm, no TEST access."""
from data_access import ROOT,CODE,load,parse_answer
from round22_common import read,dump,sha
from round46_selector import infer,select
from round25_selector import scores
from round4_gates import apply_gate
from round46_train import metrics
import numpy as np,shutil,subprocess,sys,time

OUT=ROOT/'experiments/round46'

def main():
    assert not (OUT/'validation_result.json').exists()
    plan=read(OUT/'plan.json')
    for p,h in plan['source_hashes'].items():assert sha(ROOT/p)==h,p
    b,a,y=load('validation');bs=scores(b,a,ROOT/'experiments/round25/selected_bundle');base=apply_gate(b,bs,'parse_and_repair',4)
    assert int(y[np.arange(len(y)),base].sum())==752
    conflict=np.asarray([len({parse_answer(c['text']) for c in r['candidates']}-{None})>=2 for r in b]);assert conflict.sum()==355
    result=[]
    for arm in plan['arms']:
        start=time.perf_counter();paths=[OUT/'full'/f'{arm}_s{seed}' for seed in plan['seeds']]
        s,mask,matches=infer(b,paths);c=select(b,bs,s,mask,matches)
        np.savez(OUT/f'validation_{arm}_scores.npz',scores=s,mask=mask,matches=matches,choices=c)
        p=read(ROOT/'experiments/final_evaluation_20261021/packaging_validation_predictions.json')
        p['selected']=c.astype(int).tolist();p.pop('prediction_seconds',None)
        pred=OUT/f'validation_{arm}_predictions.json';dump(pred,p)
        scorefile=OUT/f'validation_{arm}_official_score.json'
        command=[sys.executable,str(CODE/'score_predictions.py'),'--split','validation','--predictions',str(pred),'--output',str(scorefile)]
        run=subprocess.run(command,cwd=CODE,capture_output=True,text=True,encoding='utf-8')
        (OUT/f'validation_{arm}_official_stdout.txt').write_text(run.stdout,encoding='utf-8')
        (OUT/f'validation_{arm}_official_stderr.txt').write_text(run.stderr,encoding='utf-8')
        assert run.returncode==0,(run.stdout,run.stderr)
        rec=metrics(y,c,base,conflict);rec.update(arm=arm,relation_inference_seconds=time.perf_counter()-start)
        assert read(scorefile)['correct']==rec['correct'] and read(scorefile)['n']==798
        bundle=OUT/f'bundle_{arm}';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
        for j,path in enumerate(paths):
            dst=bundle/f'policy{j}';dst.mkdir();shutil.copy2(path/'model.npz',dst/'model.npz')
        dump(bundle/'config.json',dict(policies=['policy0','policy1'],mode='fixed_residual',arm=arm))
        rec['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());rec['asset_MiB']=rec['asset_bytes']/1024**2
        assert rec['asset_bytes']<=16*1024**2
        result.append(rec);print('VALIDATION',rec,flush=True)
    dump(OUT/'validation_result.json',result)
    best=max(result,key=lambda r:(r['correct'],-r['harms']))
    dump(OUT/'selection.json',dict(incumbent_correct=752,selected='round25' if best['correct']<=752 else best['arm'],best_new=best,
         selection_uses='complete official validation only; reference-machine timing still required',test_opened=False))

if __name__=='__main__':main()
