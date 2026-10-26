"""Independent validation reproduction, interface checks and immutable history."""
from data_access import ROOT,CODE,load
from round22_common import read,dump,sha
from round43_44_verify import process
from round46_selector import predict
from features import metadata
import numpy as np,copy,sys,os

OUT=ROOT/'experiments/round47'

def main():
    cache={}
    manifests=[ROOT/f'experiments/round{i}/completion_manifest.json' for i in range(22,38)]
    manifests += [ROOT/'experiments/round38_42_completion_manifest.json',ROOT/'experiments/round43_44_completion_manifest.json',ROOT/'experiments/round45/completion_manifest.json']
    for path in manifests:
        for p,h in read(path)['hashes'].items():
            if p not in cache:cache[p]=sha(ROOT/p)
            assert cache[p]==h,p
    for p,h in read(OUT/'plan.json')['source_hashes'].items():assert sha(ROOT/p)==h,p
    for p,h in read(OUT/'prepared_freeze.json')['hashes'].items():assert sha(ROOT/p)==h,p
    dump(OUT/'history_integrity.json',dict(manifests=len(manifests),distinct_files=len(cache),unchanged=True,test_opened=False))
    with np.load(OUT/'train_targets.npz') as z:ids=z['ids'];folds=z['folds']
    new_held={r['train_index'] for r in read(OUT/'held_source_review_before_reference.json')}
    for stage in ('diagnostic','full'):
        expected=ids[folds!=2].tolist() if stage=='diagnostic' else ids.tolist()
        for path in (OUT/stage).glob('*/training.json'):
            r=read(path);assert r['training_ids']==expected,path
            if 'constraint_parents' in r:
                assert not set(r['constraint_parents'])&new_held,path
                if stage=='diagnostic':assert set(r['constraint_parents'])<=set(expected),path
    best=read(OUT/'selection.json')['best_new'];arm=best['arm'];bundle=OUT/f'bundle_{arm}'
    pred=OUT/'fresh_validation_predictions.json';score=OUT/'fresh_validation_score.json'
    times=[process([CODE/'predict.py','--method',ROOT/'research/round46_selector.py','--checkpoint',bundle,'--split','validation','--output',pred]),
           process([CODE/'score_predictions.py','--split','validation','--predictions',pred,'--output',score])]
    assert read(score)['correct']==best['correct'] and read(score)['n']==798
    chosen=np.asarray(read(pred)['selected']);assert chosen.tolist()==read(OUT/f'validation_{arm}_predictions.json')['selected']
    b,a,y=load('validation');base=np.asarray(read(ROOT/'experiments/final_evaluation_20261021/packaging_validation_predictions.json')['selected'])
    rng=np.random.default_rng(47047);ix=np.unique(np.r_[rng.choice(len(b),12,replace=False),np.flatnonzero(chosen!=base)])
    bb=[copy.deepcopy(b[i]) for i in ix];aa={'embeddings':a['embeddings'][ix]}
    def forbid_refs(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0]).replace('\\','/').lower()
            if any(p.endswith('/data/gsm8k_'+s+'.jsonl') for s in ('train','validation','test')):raise PermissionError('Inference cannot open reference answers')
    sys.addaudithook(forbid_refs)
    direct=np.asarray(predict(bb,aa,bundle));assert np.array_equal(direct,chosen[ix])
    parts=[]
    for start in range(0,len(bb),5):parts+=predict(bb[start:start+5],{'embeddings':aa['embeddings'][start:start+5]},bundle)
    assert parts==direct.tolist()
    changed=copy.deepcopy(bb)
    for i,r in enumerate(changed):r['id']=f'unseen-graph-{i}'
    assert predict(changed,aa,bundle)==direct.tolist()
    perm=np.asarray([rng.permutation(4) for _ in bb]);pb=copy.deepcopy(bb);pe=np.take_along_axis(aa['embeddings'],perm[:,:,None],axis=1)
    for i,r in enumerate(pb):r['candidates']=[dict(bb[i]['candidates'][j],index=k) for k,j in enumerate(perm[i])]
    pc=np.asarray(predict(pb,{'embeddings':pe},bundle));mapped=perm[np.arange(len(bb)),pc];answers=metadata(bb)['answers']
    assert all(answers[i][int(c)]==answers[i][int(mapped[i])] for i,c in enumerate(direct))
    assert predict([],{'embeddings':np.empty((0,4,1024),np.float32)},bundle)==[]
    same=copy.deepcopy(bb[:2]);emb=aa['embeddings'][:2].copy()
    for i,r in enumerate(same):
        c=int(direct[i]);r['candidates']=[dict(bb[i]['candidates'][c],index=j) for j in range(4)];emb[i]=aa['embeddings'][i,c]
    sc=predict(same,{'embeddings':emb},bundle);assert all(0<=c<4 for c in sc)
    size=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert size<=16*1024**2
    report=dict(arm=arm,correct=best['correct'],n=798,checked_cases=len(ix),fresh_cache_equal=True,
        no_reference_reads_during_interface_checks=True,batch_invariant=True,unknown_id_invariant=True,candidate_permutation_answer_invariant=True,
        empty_input=True,identical_candidates=True,asset_bytes=size,asset_MiB=size/1024**2,
        prediction_seconds=times[0],scoring_seconds=times[1],total_fresh_process_seconds=sum(times),
        timing_bank='complete798validation, not467test',local_cpu='Intel Core i7-11800H',threads=4,reference_machine_certified=False,test_opened=False)
    dump(OUT/'verification.json',report);print(report,flush=True)

if __name__=='__main__':main()
