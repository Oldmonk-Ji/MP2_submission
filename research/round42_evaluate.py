from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump,sha,load
from round42_relations import solve
from round40_evaluate import apply,metrics
from round25_selector import scores
from round4_gates import apply_gate
import numpy as np,time

OUT=ROOT/'experiments/round42'
def main():
    p=read(OUT/'freeze.json')
    for f,h in p['source_hashes'].items():assert sha(ROOT/f)==h
    b,a,y,ids=load('train');xx=read(OUT/'train_extractions.json');assert [r['train_index'] for r in xx]==ids.tolist()
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:base=z['choices'];s=z['scores'];fold=z['folds']
    new,proofs=apply(b,base,s,[r['extraction'] for r in xx]);discovery=set(p['discovery_ids'])
    held=np.asarray([k for k,i in enumerate(ids) if fold[k]!=0 and int(i) not in discovery])
    train=dict(all=metrics(b,y,base,new,proofs),uninspected_held=metrics([b[k] for k in held],y[held],base[held],new[held],[proofs[k] for k in held]),
       caveat='All includes discovery cases; only uninspected_held avoids current-rule discovery. No fresh supported proof in that subset.',test_opened=False)
    dump(OUT/'train_effect.json',train);print('TRAIN',train,flush=True)
    b,a,y,ids=load('validation');start=time.perf_counter();s=scores(b,a,ROOT/'experiments/round25/selected_bundle');base=apply_gate(b,s,'parse_and_repair',4)
    assert int(y[np.arange(len(y)),base].sum())==752
    pp=[]
    for r in b:
        try:pp.append(solve(r['question']))
        except (ValueError,ZeroDivisionError):pp.append(dict(status='unknown',answer=None,issues=['unsupported_numeric_literal']))
    new,proofs=apply(b,base,s,pp);result=metrics(b,y,base,new,proofs);result.update(test_opened=False,local_seconds=time.perf_counter()-start,
       selection='round42' if result['correct']>752 else 'retain_round25',
       semantic_audit_caveat='v41 audit failures became v42 discovery; no newly covered uninspected TRAIN proofs. Diagnostic validation does not certify broad semantic precision.')
    dump(OUT/'validation_result.json',result);np.savez(OUT/'validation_predictions.npz',baseline=base,choices=new,scores=s,labels=y)
    dump(OUT/'validation_proofs.json',[dict(validation_index=k,extraction=p) for k,p in enumerate(proofs) if p['answer'] is not None])
    print('VALIDATION',result,flush=True)

if __name__=='__main__':main()
