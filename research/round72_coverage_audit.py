"""TRAIN-fit oracle diagnostic of source-quantity omission, not deployable scoring."""
from round22_common import ROOT,load,read,dump,sha
from round51_programs import candidate_roots
from round44_programs import locations
from round46_graph import graph
import numpy as np
OUT=ROOT/'experiments/round72'
if __name__=='__main__':
    OUT.mkdir();dump(OUT/'plan.json',dict(scope='fit-only reference-derived quantity-usage oracle; no inference metric',test_opened=False,source_hash=sha(__file__)))
    b,_,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)}
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:base=z['base'];fold=z['folds']
    records=[]
    for r in read(ROOT/'experiments/round63/accepted_reference_programs.json'):
        i=index[r['train_index']]
        if fold[i]==2:continue
        required={n[1] for _,n in locations(r['program']) if n[0]=='N'}
        if not required:continue
        q=b[i]['question'];g=graph(q);cover=[];available=[]
        for c in b[i]['candidates']:
            roots=candidate_roots(q,c['text'],g)
            usage=[{n[1] for _,n in locations(p) if n[0]=='N'} for p,k in roots]
            cover.append(max([len(s&required)/len(required) for s in usage],default=-1));available.append(len(roots))
        correct=[cover[j] for j in range(4) if y[i,j]];wrong=[cover[j] for j in range(4) if not y[i,j]]
        if not wrong:continue
        records.append(dict(train_index=int(ids[i]),required=sorted(required),candidate_coverage=cover,program_counts=available,labels=y[i].tolist(),
            baseline_wrong=bool(y[i,base[i]]==0),correct_strictly_dominates=max(correct)>max(wrong),wrong_strictly_dominates=max(wrong)>max(correct),
            all_candidates_parsed=all(n>0 for n in available)))
    stats={}
    for name,rows in [('all_mixed_fit',records),('baseline_errors',[r for r in records if r['baseline_wrong']]),('all_parsed',[r for r in records if r['all_candidates_parsed']])]:
        stats[name]=dict(n=len(rows),oracle_correct_dominates=sum(r['correct_strictly_dominates'] for r in rows),oracle_wrong_dominates=sum(r['wrong_strictly_dominates'] for r in rows))
    dump(OUT/'fit_oracle_records.json',records);dump(OUT/'fit_oracle_summary.json',dict(results=stats,reference_used=True,not_deployable=True,test_opened=False,held_used=False));print(stats,flush=True)
