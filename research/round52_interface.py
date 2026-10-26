"""Input interface tests on VAL only, including all changed predictions."""
from data_access import ROOT,load
from round22_common import read,dump
from features import parse_answer
from pathlib import Path
import argparse,sys,importlib.util,copy,numpy as np

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method-dir',required=True);p.add_argument('--bundle',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    bank,arrays,y=load('validation');chosen=np.asarray(read(args.predictions)['selected']);base=np.asarray(read(ROOT/'experiments/final_evaluation_20261021/packaging_validation_predictions.json')['selected'])
    rng=np.random.default_rng(52052);ix=np.unique(np.r_[rng.choice(len(bank),12,replace=False),np.flatnonzero(chosen!=base)])
    bb=[copy.deepcopy(bank[i]) for i in ix];aa={'embeddings':arrays['embeddings'][ix]}
    sys.path.insert(0,args.method_dir);spec=importlib.util.spec_from_file_location('frozen_entry',Path(args.method_dir)/'entrypoint.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);predict=mod.predict
    c=predict(bb,aa,args.bundle);assert c==chosen[ix].tolist()
    parts=[]
    for start in range(0,len(bb),5):parts+=predict(bb[start:start+5],{'embeddings':aa['embeddings'][start:start+5]},args.bundle)
    assert parts==c
    renamed=copy.deepcopy(bb)
    for i,r in enumerate(renamed):r['id']='unseen-feature-'+str(i)
    assert predict(renamed,aa,args.bundle)==c
    perms=np.asarray([rng.permutation(4) for _ in bb]);pb=copy.deepcopy(bb);pe=np.take_along_axis(aa['embeddings'],perms[:,:,None],axis=1)
    for i,r in enumerate(pb):r['candidates']=[dict(bb[i]['candidates'][j],index=k) for k,j in enumerate(perms[i])]
    pc=predict(pb,{'embeddings':pe},args.bundle)
    assert all(parse_answer(bb[i]['candidates'][c[i]]['text'])==parse_answer(bb[i]['candidates'][perms[i,pc[i]]]['text']) for i in range(len(bb)))
    assert predict([],{'embeddings':np.empty((0,4,1024),np.float32)},args.bundle)==[]
    same=copy.deepcopy(bb[:2]);em=aa['embeddings'][:2].copy()
    for i,r in enumerate(same):r['candidates']=[dict(bb[i]['candidates'][c[i]],index=j) for j in range(4)];em[i]=aa['embeddings'][i,c[i]]
    assert all(0<=x<4 for x in predict(same,{'embeddings':em},args.bundle))
    result=dict(cases=len(ix),all_passed=True,fresh_predictions_equal=True,batch_invariant=True,unseen_ids=True,candidate_permutation_answer_invariant=True,empty=True,identical_candidates=True,reference_reads_blocked=True,test_opened=False)
    dump(args.output,result);print(result,flush=True)
