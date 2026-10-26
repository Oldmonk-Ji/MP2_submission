"""Fresh official validation runs and deployment contract checks."""
from data_access import ROOT,CODE
from round22_common import read,dump,load,sha
from round43_44_verify import process
from round45_selector import predict
from features import metadata
import copy,numpy as np

OUT=ROOT/'experiments/round45'
def main():
    cache={};manifests=[ROOT/f'experiments/round{i}/completion_manifest.json' for i in range(22,38)]
    manifests += [ROOT/'experiments/round38_42_completion_manifest.json',ROOT/'experiments/round43_44_completion_manifest.json']
    for path in manifests:
        for p,h in read(path)['hashes'].items():
            if p not in cache:cache[p]=sha(ROOT/p)
            assert cache[p]==h,p
    for p,h in read(OUT/'plan.json')['source_and_input_hashes'].items():assert sha(ROOT/p)==h,p
    partition=read(OUT/'diagnostic_partition.json');allids=read(OUT/'training_ids.json')
    allowed_parents=set(partition['fit_constraint_parents'])
    for stage in ('diagnostic','full'):
        expected=partition['fit_ids'] if stage=='diagnostic' else allids
        for p in (OUT/stage).glob('*/training.json'):
            info=read(p);assert info['training_ids']==expected,str(p)
            if stage=='diagnostic':assert set(info['constraint_parents'])<=allowed_parents,str(p)
    dump(OUT/'history_integrity.json',dict(prior_manifests=len(manifests),distinct_prior_files=len(cache),unchanged=True,test_opened=False))
    selection=read(OUT/'selection.json')
    for row in selection['records']:
        if row['name']=='control':continue
        pred=OUT/f'validation_{row["name"]}_predictions.json';dest=OUT/f'official_{row["name"]}_score.json'
        process([CODE/'score_predictions.py','--split','validation','--predictions',pred,'--output',dest])
        r=read(dest);assert r['correct']==row['correct'] and r['n']==798
    reports=[]
    for item in selection['evaluation_bundles']:
        level=item['level'];bundle=ROOT/item['bundle'];name=item['name'];pred=OUT/f'official_{level}_predictions.json';score=OUT/f'official_{level}_score.json'
        timing=[process([CODE/'predict.py','--method',ROOT/'research/round45_selector.py','--checkpoint',bundle,'--split','validation','--output',pred]),
                process([CODE/'score_predictions.py','--split','validation','--predictions',pred,'--output',score])]
        assert read(score)['correct']==item['correct'];choice=np.asarray(read(pred)['selected'])
        with np.load(OUT/'validation_choices.npz') as z:assert np.array_equal(choice,z[name]);base=z['control']
        b,a,y,_=load('validation');rng=np.random.default_rng(45045);ids=np.unique(np.r_[rng.choice(len(b),10,replace=False),np.flatnonzero(choice!=base)])
        bb=[copy.deepcopy(b[i]) for i in ids];aa={'embeddings':a['embeddings'][ids]};expected=choice[ids]
        direct=np.asarray(predict(bb,aa,bundle));assert np.array_equal(direct,expected)
        pieces=[]
        for start in range(0,len(bb),5):pieces+=predict(bb[start:start+5],{'embeddings':aa['embeddings'][start:start+5]},bundle)
        assert np.array_equal(pieces,expected)
        changed=copy.deepcopy(bb)
        for i,r in enumerate(changed):r['id']=f'unknown-rlvr-{i}'
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
        size=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert size<=16*1024**2
        result=dict(level=level,name=name,correct=item['correct'],n=798,official_cache_match=True,checked_cases=len(ids),
            batch_invariant=True,unknown_id_invariant=True,candidate_permutation_answer_invariant=True,empty_input=True,identical_candidates=True,
            asset_bytes=size,asset_MiB=size/1024**2,local_cpu='Intel Core i7-11800H',threads=4,
            prediction_seconds=timing[0],scoring_seconds=timing[1],total_fresh_process_seconds=sum(timing),local_under30=sum(timing)<=30,
            reference_machine_timing_certified=False,test_opened=False)
        reports.append(result);print('VERIFIED',result,flush=True)
    dump(OUT/'verification.json',reports)

if __name__=='__main__':main()
