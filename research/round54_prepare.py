"""Second fit-only parser freeze; report gains AND losses at fixed64 budget."""
from data_access import ROOT
from round22_common import read,dump,sha,load
from round54_programs import proposals,checks,value
import numpy as np,json,time,datetime

OUT=ROOT/'experiments/round54'

if __name__=='__main__':
    OUT.mkdir();b,_,y,ids=load('train')
    with np.load(ROOT/'experiments/round51/train_targets.npz') as z:
        folds=z['folds'];base=z['base'];prior=z['rewards'].any(1)
    ok=y[np.arange(len(y)),base].astype(bool);fitids=ids[(folds!=2)&~ok&~prior].tolist()
    sources=[ROOT/'research'/n for n in ('round54_programs.py','round54_prepare.py','round54_selector.py','round54_evaluate.py')]
    dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fit_discovery_ids=fitids,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},test_opened=False,held_text_used_for_rules=False,
        hypothesis='Repair markdown/symbol substitution and complete final target operation from existing intermediates.',
        inference='Up to16 new connected roots then old proposals within64. Report lost coverage too.',
        weights='Frozen best NEW method from round52/53 VAL; no new training or parameter search.',validation='One prespecified coverage variant, complete798 then final feature selection before TEST.'))
    dump(OUT/'parser_checks.json',checks());oracle=np.zeros(len(y),bool);records=[];start=time.perf_counter()
    with (OUT/'train_proposals.jsonl').open('w',encoding='utf-8') as f:
        for i,r in enumerate(b):
            p=proposals(r);f.write(json.dumps(dict(train_index=int(ids[i]),**p),ensure_ascii=False)+'\n')
            for h in p['hypotheses']:
                assert str(value(h['program'],p['quantities']))==h['answer']
                oracle[i]|=any(y[i,j] for j in h['matches'])
            if i%600==0:print('TRACE_COMPLETION',i,'seconds',time.perf_counter()-start,flush=True)
    for name,ix in [('all',np.ones(len(y),bool)),('held',folds==2),('held_baseline_errors',(folds==2)&~ok),('fit_baseline_errors',(folds!=2)&~ok)]:
        records.append(dict(group=name,n=int(ix.sum()),previous=int(prior[ix].sum()),correct_program=int(oracle[ix].sum()),gained=int((ix&oracle&~prior).sum()),lost=int((ix&~oracle&prior).sum())))
    np.savez(OUT/'coverage.npz',oracle=oracle,ids=ids,folds=folds)
    dump(OUT/'preparation_result.json',dict(records=records,seconds=time.perf_counter()-start,test_opened=False));dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()}));print(records,flush=True)
