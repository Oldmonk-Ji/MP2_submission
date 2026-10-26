"""Independent official validation reproduction and production invariants."""
from data_access import ROOT,CODE
from round22_common import load,read,dump,sha
from features import metadata
import copy,importlib,os,subprocess,sys,time
import numpy as np


def main(number):
    out=ROOT/f'experiments/round{number}';bundle=out/'selected_bundle';selected=read(out/'selection.json')['selected']
    assert bundle.exists() and selected['correct']>752
    env=dict(os.environ)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='4'
    prediction=out/'official_validation_predictions.json';score=out/'official_validation_score.json';timing=[]
    for args in [[str(CODE/'predict.py'),'--method',str(ROOT/f'research/round{number}_selector.py'),'--checkpoint',str(bundle),'--split','validation','--output',str(prediction)],
                 [str(CODE/'score_predictions.py'),'--split','validation','--predictions',str(prediction),'--output',str(score)]]:
        tick=time.perf_counter();cp=subprocess.run([sys.executable,'-X','utf8']+args,env=env,capture_output=True,text=True,encoding='utf-8')
        timing.append(time.perf_counter()-tick);assert cp.returncode==0,(cp.stdout,cp.stderr);print(cp.stdout.strip(),flush=True)
    official=read(score);assert official['correct']==selected['correct'] and official['n']==798
    choice=np.asarray(read(prediction)['selected'])
    with np.load(out/'validation_choices.npz') as z:assert np.array_equal(z[selected['name']],choice)
    bn=read(ROOT/'experiments/round25/selection.json')['selected']['name']
    with np.load(ROOT/'experiments/round25/validation_choices.npz') as z:old=z[bn]
    b,a,y,_=load('validation');module=importlib.import_module(f'round{number}_selector');rng=np.random.default_rng(29035)
    ids=np.unique(np.r_[rng.choice(len(b),12,replace=False),np.flatnonzero(choice!=old)])
    bb=[copy.deepcopy(b[i]) for i in ids];aa={'embeddings':a['embeddings'][ids]};expected=choice[ids]
    direct=np.asarray(module.predict(bb,aa,bundle));assert np.array_equal(direct,expected)
    pieces=[]
    for start in range(0,len(bb),5):pieces+=module.predict(bb[start:start+5],{'embeddings':aa['embeddings'][start:start+5]},bundle)
    assert np.array_equal(pieces,expected)
    changed=copy.deepcopy(bb)
    for i,row in enumerate(changed):row['id']=f'unknown-semantic-check-{i}'
    assert module.predict(changed,aa,bundle)==direct.tolist()
    permutation=np.asarray([rng.permutation(4) for _ in bb]);pb=copy.deepcopy(bb);pe=np.take_along_axis(aa['embeddings'],permutation[:,:,None],axis=1)
    for i,row in enumerate(pb):row['candidates']=[dict(bb[i]['candidates'][j],index=k) for k,j in enumerate(permutation[i])]
    pc=np.asarray(module.predict(pb,{'embeddings':pe},bundle));restored=permutation[np.arange(len(bb)),pc];answers=metadata(bb)['answers']
    assert all(answers[i][int(direct[i])]==answers[i][int(restored[i])] for i in range(len(bb)))
    assert module.predict([],{'embeddings':np.empty((0,4,1024),np.float32)},bundle)==[]
    same=copy.deepcopy(bb[:2]);se=aa['embeddings'][:2].copy()
    for i,row in enumerate(same):
        j=int(direct[i]);row['candidates']=[dict(bb[i]['candidates'][j],index=k) for k in range(4)];se[i]=aa['embeddings'][i,j]
    sc=module.predict(same,{'embeddings':se},bundle);assert len(sc)==2 and all(0<=c<4 for c in sc)
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets);assert size<=16*1024**2
    report=dict(official_prediction_matches_training_cache=True,full_validation_n=798,test_opened=False,
        checked_cases=len(ids),batch_invariant=True,unknown_id_invariant=True,candidate_permutation_answer_invariant=True,
        empty_input=True,identical_candidates=True,asset_bytes=size,asset_MiB=size/1024**2,
        local_cpu='Intel Core i7-11800H',threads=4,prediction_process_seconds=timing[0],scoring_process_seconds=timing[1],
        total_fresh_process_seconds=sum(timing),reference_machine_eligibility_not_certified=True)
    dump(out/'verification.json',report)
    ok=y[np.arange(len(b)),choice].astype(bool);oldok=y[np.arange(len(b)),old].astype(bool);changes=[];errors=[]
    for i,row in enumerate(b):
        rec=dict(validation_index=i,question=row['question'],chosen_index=int(choice[i]),previous_index=int(old[i]),correct=bool(ok[i]),
            previous_correct=bool(oldok[i]),candidate_labels=y[i].astype(int).tolist(),candidate_texts=[c['text'] for c in row['candidates']])
        if ok[i]!=oldok[i]:rec['change']='repaired' if ok[i] else 'damaged';changes.append(rec)
        if not ok[i]:errors.append(rec)
    import json
    for name,rows in [('validation_correctness_changes',changes),('validation_remaining_errors',errors)]:
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    print('VERIFIED',report,flush=True)


if __name__=='__main__':main(int(sys.argv[1]))
