"""One frozen validation comparison after positive train-only confirmation."""
from data_access import ROOT,CODE,load
import json,hashlib,shutil,time,subprocess
import numpy as np
from features import metadata
from round9_train import metric
from round9_selector import scores,predict
from round4_gates import apply_gate
from reliability import calibration_stats
OUT=ROOT/'experiments/round12'

def main():
    decision=json.loads((OUT/'decision.json').read_text());assert decision['summary']['group']['gate_passed'] or (OUT/'before_order_fix/note.json').exists()
    for filename in ['round9','round12']:
        for name,h in json.loads((ROOT/'experiments'/filename/'frozen.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,(name,'changed since freeze')
    original=ROOT/'experiments/round4/selected_bundle';gate=json.loads((original/'config.json').read_text());variants={}
    for name,kind in [('answer_group','groups'),('candidate_ablation','candidate')]:
        path=OUT/name;path.mkdir(exist_ok=True);shutil.copytree(original/'base',path/'base',dirs_exist_ok=True);shutil.copy2(ROOT/'experiments/round9'/f'{kind}.txt',path/'correction.txt')
        cfg=dict(family='answer_group_correction',kind=kind,alpha=.25,gate_kind=gate['gate_kind'],margin_limit=gate['margin_limit'])
        (path/'config.json').write_text(json.dumps(cfg,indent=2));variants[name]=path
    (OUT/'validation_plan.json').write_text(json.dumps(dict(variants=list(variants),selection='Keep new group method only if strictly more validation correct than old bundle and within limits. Candidate variant is a fixed ablation, not a new hyperparameter search.',test_access=False),indent=2))
    b,a,y=load('validation');meta=metadata(b);old=np.load(ROOT/'experiments/round4/selected_predictions.npz')['selected'];oldok=y[np.arange(len(y)),old]>0
    results={};saved={}
    for name,path in variants.items():
        tick=time.perf_counter();s=scores(b,a,path);c=apply_gate(b,s,gate['gate_kind'],gate['margin_limit']);elapsed=time.perf_counter()-tick;saved[name]=(s,c)
        rec=metric(c,y,meta['hard'],old);rec['inprocess_seconds']=elapsed;rec['asset_bytes']=sum(p.stat().st_size for p in path.rglob('*') if p.is_file());rec['changed_candidate_indices']=np.flatnonzero(c!=old).tolist();rec['changed_indices']=[i for i in range(len(b)) if meta['answers'][i][c[i]]!=meta['answers'][i][old[i]]];results[name]=rec
        np.savez(OUT/f'{name}_validation.npz',scores=s,selected=c);print(name,json.dumps(rec),flush=True)
    choice=saved['answer_group'][1]
    # Contract checks target both new answer choices and duplicate answer groups.
    ids=list(range(12))+results['answer_group']['changed_indices'];ids=list(dict.fromkeys(ids));bb=[dict(b[i],id='unseen-id-'+str(k)) for k,i in enumerate(ids)];aa={'embeddings':a['embeddings'][ids]}
    expected=choice[ids].tolist();assert predict(bb,aa,variants['answer_group'])==expected
    perm=np.array([2,0,3,1]);bp=[dict(r,candidates=[dict(r['candidates'][int(j)],index=k) for k,j in enumerate(perm)]) for r in bb]
    pp=predict(bp,{'embeddings':aa['embeddings'][:,perm]},variants['answer_group'])
    original_answers=[meta['answers'][i][c] for i,c in zip(ids,expected)]
    assert [metadata(bp)['answers'][k][c] for k,c in enumerate(pp)]==original_answers,'Candidate-order dependence'
    assert predict([],{'embeddings':np.empty((0,4,1024))},variants['answer_group'])==[]
    for n in (1,3):assert predict(bb[:n],{'embeddings':aa['embeddings'][:n]},variants['answer_group'])==expected[:n]
    # Official prediction + scoring processes, validation only, fresh startup.
    python=ROOT/'.venv/Scripts/python.exe';pred=OUT/'official-validation-predictions.json';score=OUT/'official-validation-score.json';tick=time.perf_counter()
    subprocess.run([str(python),str(CODE/'predict.py'),'--method',str(ROOT/'research/round9_selector.py'),'--checkpoint',str(variants['answer_group']),'--split','validation','--output',str(pred)],capture_output=True,check=True)
    subprocess.run([str(python),str(CODE/'score_predictions.py'),'--split','validation','--predictions',str(pred),'--output',str(score)],capture_output=True,check=True)
    seconds=time.perf_counter()-tick;official=json.loads(score.read_text());assert official['correct']==results['answer_group']['correct'];assert json.loads(pred.read_text())['selected']==choice.tolist()
    selected=decision['summary']['group']['gate_passed'] and results['answer_group']['correct']>int(oldok.sum()) and results['answer_group']['asset_bytes']<16*2**20 and seconds<30
    ok=y[np.arange(len(y)),choice]>0;delta=ok.astype(float)-oldok;rng=np.random.default_rng(20261014);boot=[delta[rng.integers(0,len(delta),len(delta))].mean()*100 for _ in range(5000)]
    report=dict(baseline=metric(old,y,meta['hard']),variants=results,official_validation=official,fresh_predict_and_score_seconds=seconds,local_hardware='Intel i7-11800H, CPU, four threads; reference CPU eligibility not established',paired_gain_95ci_pp=np.quantile(boot,[.025,.975]).tolist(),selection_caveat='Validation reused across prior rounds; interval does not correct adaptive selection. Train meta-crossfit not fully nested.',new_method_selected=bool(selected),selected_path=str(variants['answer_group'] if selected else original),selector=str(ROOT/'research/round9_selector.py' if selected else ROOT/'research/round4_selector.py'),test_access=False,contract_checks='passed: cached/full inference parity, unknown IDs, variable N, empty bank, numerical-answer invariance under candidate permutation',target_correct_for_99=791)
    (OUT/'validation_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
