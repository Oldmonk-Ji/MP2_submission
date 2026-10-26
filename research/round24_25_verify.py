"""Official reproduction, input invariants, immutable historical assets."""
from data_access import ROOT,CODE
from round22_common import load,read,dump,sha
import argparse,copy,importlib,json,os,subprocess,sys,time
import numpy as np
from features import metadata


def main(round_number):
    out=ROOT/f'experiments/round{round_number}';bundle=out/'selected_bundle'
    selected=read(out/'selection.json')['selected']
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
    prediction=out/'official_validation_predictions.json';score=out/'official_validation_score.json';timing=[]
    for args in [[str(CODE/'predict.py'),'--method',str(ROOT/f'research/round{round_number}_selector.py'),
                 '--checkpoint',str(bundle),'--split','validation','--output',str(prediction)],
                 [str(CODE/'score_predictions.py'),'--split','validation','--predictions',str(prediction),'--output',str(score)]]:
        tick=time.perf_counter();cp=subprocess.run([sys.executable,'-X','utf8']+args,env=env,capture_output=True,text=True,encoding='utf-8')
        timing.append(time.perf_counter()-tick);assert cp.returncode==0,(cp.stdout,cp.stderr);print(cp.stdout.strip(),flush=True)
    official=read(score);assert official['correct']==selected['correct'] and official['n']==798
    choice=np.asarray(read(prediction)['selected'])
    with np.load(out/'validation_choices.npz') as z:assert np.array_equal(z[selected['name']],choice)
    module=importlib.import_module(f'round{round_number}_selector')
    bank,arrays,y,_=load('validation');rng=np.random.default_rng(24025)
    # Include actual changed cases as well as a fixed random subset.
    with np.load(ROOT/'experiments/round22/validation_comparison.npz') as z:old=z['incumbent_choice']
    ids=np.unique(np.r_[rng.choice(len(bank),12,replace=False),np.flatnonzero(choice!=old)[:12]])
    bb=[copy.deepcopy(bank[i]) for i in ids];aa={'embeddings':arrays['embeddings'][ids]}
    expected=choice[ids]
    direct=np.asarray(module.predict(bb,aa,bundle));assert np.array_equal(direct,expected)
    pieces=[]
    for start in range(0,len(bb),5):pieces+=module.predict(bb[start:start+5],{'embeddings':aa['embeddings'][start:start+5]},bundle)
    assert np.array_equal(pieces,expected)
    changed=copy.deepcopy(bb)
    for i,row in enumerate(changed):row['id']=f'never-seen-{i+995031}'
    assert module.predict(changed,aa,bundle)==direct.tolist()
    permutation=np.array([rng.permutation(4) for _ in bb]);pb=copy.deepcopy(bb)
    pe=np.take_along_axis(aa['embeddings'],permutation[:,:,None],axis=1)
    for i,row in enumerate(pb):
        row['candidates']=[dict(bb[i]['candidates'][j],index=k) for k,j in enumerate(permutation[i])]
    pc=np.asarray(module.predict(pb,{'embeddings':pe},bundle));restored=permutation[np.arange(len(bb)),pc]
    answers=metadata(bb)['answers']
    assert all(answers[i][int(direct[i])]==answers[i][int(restored[i])] for i in range(len(bb)))
    assert module.predict([],{'embeddings':np.empty((0,4,1024),np.float32)},bundle)==[]
    same=copy.deepcopy(bb[:2]);se=aa['embeddings'][:2].copy()
    for i,row in enumerate(same):
        j=int(direct[i]);row['candidates']=[dict(bb[i]['candidates'][j],index=k) for k in range(4)]
        se[i]=aa['embeddings'][i,j]
    sc=module.predict(same,{'embeddings':se},bundle);assert len(sc)==2 and all(0<=c<4 for c in sc)
    historical={}
    for r in (22,23):
        manifest=read(ROOT/f'experiments/round{r}/completion_manifest.json')
        assert all(sha(ROOT/k)==v for k,v in manifest['hashes'].items())
        historical[str(r)]=True
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets)
    assert size<=16*1024**2
    report=dict(official_prediction_matches_training_cache=True,full_validation_n=798,test_opened=False,
        random_and_changed_cases_checked=len(ids),batch_invariant=True,unknown_id_invariant=True,
        candidate_permutation_answer_invariant=True,empty_input=True,identical_candidates=True,
        historical_manifests_unchanged=historical,asset_bytes=size,asset_MiB=size/1024**2,
        local_cpu='Intel Core i7-11800H',threads=4,prediction_process_seconds=timing[0],scoring_process_seconds=timing[1],
        total_fresh_process_seconds=sum(timing),reference_machine_eligibility_not_certified=True)
    dump(out/'verification.json',report)
    changes=[];errors=[];ok=y[np.arange(len(bank)),choice].astype(bool);oldok=y[np.arange(len(bank)),old].astype(bool)
    for i,row in enumerate(bank):
        rr=dict(validation_index=i,question=row['question'],chosen_index=int(choice[i]),previous_index=int(old[i]),
            correct=bool(ok[i]),previous_correct=bool(oldok[i]),candidate_texts=[c['text'] for c in row['candidates']])
        if ok[i]!=oldok[i]:rr['change']='repaired' if ok[i] else 'damaged';changes.append(rr)
        if not ok[i]:errors.append(rr)
    for name,rows in [('validation_correctness_changes',changes),('validation_remaining_errors',errors)]:
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    sources=[ROOT/f'research/round{round_number}_selector.py',ROOT/f'research/round{round_number}_run.py',Path(__file__)]
    if round_number==24:sources.append(ROOT/'research/round25_selector.py')
    dump(out/'frozen.json',dict(test_opened=False,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets+sources}))
    print('VERIFIED',report,flush=True)


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('round',type=int,choices=[24,25]);a=p.parse_args();main(a.round)
