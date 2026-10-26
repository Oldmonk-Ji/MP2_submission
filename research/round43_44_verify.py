"""Independent scoring, fresh-process interface checks, immutable-history audit."""
from data_access import ROOT,CODE
from round22_common import read,dump,load,sha
from features import metadata
import copy,os,subprocess,sys,time
import numpy as np

def process(args):
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
    tick=time.perf_counter()
    p=subprocess.run([sys.executable,'-X','utf8']+[str(a) for a in args],env=env,capture_output=True,text=True,encoding='utf-8')
    elapsed=time.perf_counter()-tick
    assert p.returncode==0,(p.stdout,p.stderr)
    print(p.stdout.strip(),flush=True)
    return elapsed

def history():
    cache={};manifests=[ROOT/f'experiments/round{i}/completion_manifest.json' for i in range(22,38)]
    manifests.append(ROOT/'experiments/round38_42_completion_manifest.json')
    for manifest in manifests:
        for name,h in read(manifest)['hashes'].items():
            if name not in cache:cache[name]=sha(ROOT/name)
            assert cache[name]==h,(str(manifest),name)
    assert sha(ROOT/'research/round43_dpo.py')==read(ROOT/'experiments/round43/plan.json')['source_hash']
    for name,h in read(ROOT/'experiments/round44/training_freeze.json')['hashes'].items():assert sha(ROOT/name)==h,name
    result=dict(prior_manifests=len(manifests),distinct_prior_files=len(cache),all_unchanged=True,round43_44_training_freezes_unchanged=True,test_opened=False)
    dump(ROOT/'experiments/round44/history_integrity.json',result);print(result,flush=True)

def verify():
    history()
    for number in (43,44):
        out=ROOT/f'experiments/round{number}';sel=read(out/'selection.json')
        for row in sel['records']:
            label=row.get('name',row.get('arm'))
            if label=='control':continue
            pred=out/f'validation_{label}_predictions.json';dest=out/f'official_{label}_score.json'
            process([CODE/'score_predictions.py','--split','validation','--predictions',pred,'--output',dest])
            score=read(dest);assert score['correct']==row['correct'] and score['n']==798
    out=ROOT/'experiments/round44';sel=read(out/'selection.json');bundle=ROOT/sel['evaluation_bundle'];name=sel['evaluation_bundle_name']
    expected_record=next(r for r in sel['records'] if r['name']==name)
    pred=out/'official_validation_predictions.json';score=out/'official_validation_score.json'
    timing=[process([CODE/'predict.py','--method',ROOT/'research/round44_selector.py','--checkpoint',bundle,'--split','validation','--output',pred]),
            process([CODE/'score_predictions.py','--split','validation','--predictions',pred,'--output',score])]
    actual=read(score);assert actual['correct']==expected_record['correct'] and actual['n']==798
    choice=np.asarray(read(pred)['selected'])
    with np.load(out/'validation_choices.npz') as z:
        assert np.array_equal(choice,z[name]);base=z['control']
    b,a,y,_=load('validation');from round44_selector import predict
    rng=np.random.default_rng(44044);ids=np.unique(np.r_[rng.choice(len(b),12,replace=False),np.flatnonzero(choice!=base)])
    bb=[copy.deepcopy(b[i]) for i in ids];aa={'embeddings':a['embeddings'][ids]};expected=choice[ids]
    direct=np.asarray(predict(bb,aa,bundle));assert np.array_equal(direct,expected)
    pieces=[]
    for start in range(0,len(bb),5):pieces+=predict(bb[start:start+5],{'embeddings':aa['embeddings'][start:start+5]},bundle)
    assert np.array_equal(pieces,expected)
    changed=copy.deepcopy(bb)
    for i,r in enumerate(changed):r['id']=f'unknown-relation-check-{i}'
    assert predict(changed,aa,bundle)==direct.tolist()
    permutation=np.asarray([rng.permutation(4) for _ in bb]);pb=copy.deepcopy(bb);pe=np.take_along_axis(aa['embeddings'],permutation[:,:,None],axis=1)
    for i,r in enumerate(pb):r['candidates']=[dict(bb[i]['candidates'][j],index=k) for k,j in enumerate(permutation[i])]
    pc=np.asarray(predict(pb,{'embeddings':pe},bundle));restored=permutation[np.arange(len(bb)),pc];answers=metadata(bb)['answers']
    assert all(answers[i][int(direct[i])]==answers[i][int(restored[i])] for i in range(len(bb)))
    assert predict([],{'embeddings':np.empty((0,4,1024),np.float32)},bundle)==[]
    same=copy.deepcopy(bb[:2]);se=aa['embeddings'][:2].copy()
    for i,r in enumerate(same):
        j=int(direct[i]);r['candidates']=[dict(bb[i]['candidates'][j],index=k) for k in range(4)];se[i]=aa['embeddings'][i,j]
    sc=predict(same,{'embeddings':se},bundle);assert len(sc)==2 and all(0<=c<4 for c in sc)
    size=sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file());assert size<=16*1024**2
    report=dict(official_prediction_matches_cache=True,full_validation_n=798,test_opened=False,evaluated_name=name,
        checked_cases=len(ids),batch_invariant=True,unknown_id_invariant=True,candidate_permutation_answer_invariant=True,
        empty_input=True,identical_candidates=True,asset_bytes=size,asset_MiB=size/1024**2,threads=4,
        local_cpu='Intel Core i7-11800H',prediction_process_seconds=timing[0],scoring_process_seconds=timing[1],
        total_fresh_process_seconds=sum(timing),local_under_30_seconds=sum(timing)<=30,reference_machine_eligibility_not_certified=True)
    dump(out/'verification.json',report)
    print('VERIFIED',report,flush=True)

if __name__=='__main__':globals()[sys.argv[1] if len(sys.argv)>1 else 'verify']()
