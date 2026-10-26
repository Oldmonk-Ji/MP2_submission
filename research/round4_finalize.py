"""Freeze the trained gate, calibrate on TRAIN only, verify official validation."""
from data_access import ROOT,CODE,load
import hashlib,json,shutil,subprocess,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from features import metadata
from reliability import calibration_stats
from round4_gates import apply_gate
from round4_selector import confidence_inputs,predict,predict_with_confidence
from round4_train import OUT
from train_models import SEED


def main():
    selected=json.loads((OUT/'rare_gate_selection.json').read_text())['chosen']
    cfg=dict(family='constraint_gate',base='base',gate_kind=selected['kind'],margin_limit=selected['best']['limit'])
    bundle=OUT/'selected_bundle';bundle.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'experiments/round3/stack_bundle',bundle/'base',dirs_exist_ok=True)
    (bundle/'config.json').write_text(json.dumps(cfg,indent=2))
    compact=OUT/'compact_bundle';compact.mkdir(exist_ok=True)
    for filename in ('model.npz','config.json'):shutil.copy2(OUT/'full/parser_35'/filename,compact/filename)
    b,_,y=load('train');bv,av,yv=load('validation');mt=metadata(b);mv=metadata(bv)
    old=np.load(ROOT/'experiments/round3/stack_scores.npz');st=old['meta_oof'];sv=old['validation']
    ct=apply_gate(b,st,cfg['gate_kind'],cfg['margin_limit']);cv=apply_gate(bv,sv,cfg['gate_kind'],cfg['margin_limit'])
    xt=confidence_inputs(st,mt,ct);xv=confidence_inputs(sv,mv,cv)
    yt=y[np.arange(len(y)),ct];correct=yv[np.arange(len(yv)),cv]
    sc=StandardScaler().fit(xt);cal=LogisticRegression(C=.1,max_iter=1000,random_state=SEED).fit(sc.transform(xt),yt)
    w=cal.coef_[0]/sc.scale_;bias=cal.intercept_[0]-sc.mean_@w
    np.savez(bundle/'confidence_calibrator.npz',weight=w,bias=bias)
    conf=cal.predict_proba(sc.transform(xv))[:,1]
    previous=np.where(mv['valid'],sv,-1e30).argmax(1);previous_correct=yv[np.arange(len(yv)),previous]
    rng=np.random.default_rng(SEED);boot=[]
    for _ in range(5000):
        ids=rng.integers(0,len(yv),len(yv));boot.append(float((correct[ids]-previous_correct[ids]).mean()))
    info=dict(test_access=False,selected_config=cfg,validation=dict(correct=int(correct.sum()),n=len(yv),accuracy=float(100*correct.mean()),
        hard_correct=int(correct[mv['hard']].sum()),hard_n=int(mv['hard'].sum()),hard_accuracy=float(100*correct[mv['hard']].mean())),
        paired=dict(fixed_old_errors=int(((correct==1)&(previous_correct==0)).sum()),lost_old_correct=int(((correct==0)&(previous_correct==1)).sum()),
                    gain_95ci_pp=(100*np.quantile(boot,[.025,.975])).tolist()),
        calibrated=calibration_stats(conf,correct),conflict_calibrated=calibration_stats(conf[mv['hard']],correct[mv['hard']]),
        calibration_note='Fit only on TRAIN OOF base predictions with TRAIN-fitted gate thresholds; not fully nested.',
        selection_note='Validation has been reused for method development; paired intervals do not correct adaptive selection.',
        semantic_claim='Accuracy gain is arithmetic/format consistency, not demonstrated semantic improvement.')
    ids=[0,17,52,65,224,733,739];bb=[dict(bv[i],id='unseen-'+str(i)) for i in ids];aa={'embeddings':av['embeddings'][ids]}
    choices,confidence=predict_with_confidence(bb,aa,bundle)
    assert choices==cv[ids].tolist() and np.allclose(confidence,conf[ids],atol=1e-5)
    permutation=np.array([2,0,3,1]);pp=[dict(r,candidates=[dict(r['candidates'][int(j)],index=k) for k,j in enumerate(permutation)]) for r in bb]
    pc=predict(pp,{'embeddings':aa['embeddings'][:,permutation]},bundle)
    assert permutation[np.asarray(pc)].tolist()==choices
    for n in (1,3,7):assert predict(bb[:n],{'embeddings':aa['embeddings'][:n]},bundle)==choices[:n]
    assert predict([],{'embeddings':np.empty((0,4,1024))},bundle)==[]
    compact_choices=predict(bv,av,compact)
    expected=np.where(mv['valid'],np.load(OUT/'full/parser_35/validation_scores.npz')['scores'],-1e30).argmax(1)
    assert compact_choices==expected.tolist()
    info['contract_checks']='passed: exact cached scores/choices, independent calibration output, unknown IDs, variable N, candidate permutation'
    python=ROOT/'.venv/Scripts/python.exe';timings=[]
    for k in range(3):
        pred=OUT/f'official-validation-predictions-{k}.json';score=OUT/f'official-validation-score-{k}.json';tick=time.perf_counter()
        subprocess.run([str(python),str(CODE/'predict.py'),'--method',str(ROOT/'research/round4_selector.py'),
            '--checkpoint',str(bundle),'--split','validation','--output',str(pred)],check=True,capture_output=True)
        subprocess.run([str(python),str(CODE/'score_predictions.py'),'--split','validation','--predictions',str(pred),
            '--output',str(score)],check=True,capture_output=True)
        timings.append(time.perf_counter()-tick)
        official=json.loads(score.read_text());assert official['correct']==int(correct.sum())
        assert json.loads(pred.read_text())['selected']==cv.tolist()
    assets={str(f.relative_to(bundle)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in bundle.rglob('*') if f.is_file()}
    info['asset_bytes']=sum(v['bytes'] for v in assets.values());assert info['asset_bytes']<16*2**20
    info['local_fresh_prediction_plus_score_seconds']=timings;info['official_validation']=official
    info['compact_asset_bytes']=sum(f.stat().st_size for f in compact.iterdir() if f.is_file())
    info['additional_correct_needed_for_99']=791-int(correct.sum())
    (OUT/'asset_manifest.json').write_text(json.dumps(assets,indent=2))
    (OUT/'results.json').write_text(json.dumps(info,indent=2))
    np.savez(OUT/'selected_predictions.npz',selected=cv,correct=correct,confidence=conf,train_selected=ct,train_confidence_inputs=xt)
    selection=dict(name='constraint_gate',path=str(bundle),selector=str(ROOT/'research/round4_selector.py'),
        correct=int(correct.sum()),n=len(yv),accuracy=100*float(correct.mean()),target_correct=791,
        additional_correct_needed=791-int(correct.sum()),test_access=False)
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    print(json.dumps(dict(selected=selection,assets=info['asset_bytes'],seconds=timings,paired=info['paired']),indent=2))


if __name__=='__main__':main()
