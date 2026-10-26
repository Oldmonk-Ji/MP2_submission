from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump,sha,load
from round40_budget import compile
from round25_selector import scores
from round4_gates import apply_gate
import numpy as np,time

OUT=ROOT/'experiments/round40'
def apply(bank,base,score,extractions=None):
    out=base.copy();proofs=[]
    for i,r in enumerate(bank):
        try:p=extractions[i] if extractions is not None else compile(r['question'])
        except (ValueError,ZeroDivisionError):p=dict(status='unknown',answer=None,issues=['unsupported_numeric_literal'])
        proofs.append(p)
        if p['answer'] is None:continue
        eligible=[j for j,c in enumerate(r['candidates']) if parse_answer(c['text'])==p['answer']]
        if eligible:out[i]=max(eligible,key=lambda j:score[i,j])
    return out,proofs

def metrics(bank,y,base,new,proofs):
    bc=y[np.arange(len(y)),base].astype(bool);nc=y[np.arange(len(y)),new].astype(bool)
    conflict=np.asarray([len({parse_answer(c['text']) for c in r['candidates']}-{None})>1 for r in bank])
    return dict(n=len(y),base_correct=int(bc.sum()),correct=int(nc.sum()),accuracy=float(nc.mean()*100),
        repairs=int((~bc&nc).sum()),harms=int((bc&~nc).sum()),proved=sum(p['answer'] is not None for p in proofs),
        answer_changes=sum(parse_answer(r['candidates'][a]['text'])!=parse_answer(r['candidates'][b]['text']) for r,a,b in zip(bank,base,new)),
        conflict_n=int(conflict.sum()),conflict_correct=int(nc[conflict].sum()))

def main():
    for p,h in read(OUT/'freeze.json')['source_hashes'].items():assert sha(ROOT/p)==h
    b,a,y,ids=load('train');xx=read(OUT/'train_extractions.json');assert [r['train_index'] for r in xx]==ids.tolist()
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:base=z['choices'];s=z['scores'];fold=z['folds']
    new,proofs=apply(b,base,s,[r['extraction'] for r in xx]);discovery=set(read(OUT/'freeze.json')['discovery_ids'])
    held=np.asarray([k for k,i in enumerate(ids) if fold[k]!=0 and int(i) not in discovery])
    train=dict(all=metrics(b,y,base,new,proofs),uninspected_held=metrics([b[k] for k in held],y[held],base[held],new[held],[proofs[k] for k in held]),
        caveat='All includes discovery cases; only uninspected_held avoids current-rule discovery. Historical base recipe explored before.',test_opened=False)
    dump(OUT/'train_effect.json',train);print('TRAIN',train,flush=True)
    # Diagnostic evaluation is allowed despite low coverage; no claim of a
    # fresh precision estimate when the fresh held packet is empty.
    b,a,y,ids=load('validation');start=time.perf_counter();s=scores(b,a,ROOT/'experiments/round25/selected_bundle');base=apply_gate(b,s,'parse_and_repair',4)
    assert int(y[np.arange(len(y)),base].sum())==752
    new,proofs=apply(b,base,s);result=metrics(b,y,base,new,proofs);result.update(test_opened=False,local_seconds=time.perf_counter()-start,
        selection='round40' if result['correct']>752 else 'retain_round25',
        semantic_audit_caveat='No newly covered uninspected TRAIN-fold proofs; this is a low-coverage diagnostic, not independently established semantic precision.')
    dump(OUT/'validation_result.json',result);np.savez(OUT/'validation_predictions.npz',baseline=base,choices=new,scores=s,labels=y)
    dump(OUT/'validation_proofs.json',[dict(validation_index=k,extraction=p) for k,p in enumerate(proofs) if p['answer'] is not None])
    print('VALIDATION',result,flush=True)

if __name__=='__main__':main()
