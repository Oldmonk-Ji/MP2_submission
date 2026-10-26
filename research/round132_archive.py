from round22_common import ROOT,load,read,dump,sha
from round122_features import EXCLUDE
from round103_features import STOP
from collections import Counter
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,re
if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('train')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    counts=Counter();affected=target=0
    for row,fold in zip(b,f):
        if fold==2:continue
        q=row['question'];bad=({m.group().lower() for m in re.finditer(r'\b[A-Z][a-z]+\b',q)}-EXCLUDE)&STOP
        if bad:
            affected+=1;counts.update(bad);tail=re.split(r'[.?!]',q.rstrip('? .'))[-1].lower();target+=bool(bad&set(re.findall('[a-z]+',tail)))
    dump(ROOT/'experiments/round128/fit_entity_audit.json',dict(fit_only=True,affected_questions=affected,affected_targets=target,words=counts,test_opened=False))
    x=sparse.load_npz(ROOT/'experiments/round103/train_local.npz');rev=sparse.load_npz(ROOT/'experiments/round131/train_reversed_features.npz');reports=[]
    for part,foldset in [('fit_nested',(0,1)),('outer_held',(2,))]:
        for arm in ('original','augmented'):
            changed=flips=n=correct=0
            for fold in foldset:
                stage=f'nested{fold}' if fold!=2 else 'oof2';path=ROOT/f'experiments/round103/local/{stage}.npz' if arm=='original' else ROOT/f'experiments/round131/{stage}.npz'
                ix=np.flatnonzero(f==fold);rr=(ix[:,None]*4+np.arange(4)).ravel()
                assert not set(ids[ix])&set(read(path.with_suffix('.json'))['training_ids'])
                with np.load(path) as w:weight=w['weight']
                cc=np.asarray(x[rr]@weight).reshape(-1,4).argmax(1);rc=np.asarray(rev[rr]@weight).reshape(-1,4).argmax(1)
                ok=y[ix][np.arange(len(ix)),cc];rok=y[ix][np.arange(len(ix)),rc];changed+=int((cc!=rc).sum());flips+=int((ok!=rok).sum());n+=len(ix);correct+=int(ok.sum())
            reports.append(dict(part=part,arm=arm,n=n,changed_choices=changed,correctness_flips=flips,original_view_correct=correct))
    dump(ROOT/'experiments/round131/invariance_comparison.json',dict(reports=reports,test_opened=False))
    records=[]
    for n in (125,127,129,132):
        rows=read(ROOT/f'experiments/round{n}/validation_records.json');assert rows[0]['correct']==759
        r=rows[1];assert r['correct']<759;held=read(ROOT/f"experiments/round{n}/{r['arm']}_held.json")
        records.append(dict(round=n,**r,held_correct=held['correct'],held_n=held['n']))
    for src,h in read(ROOT/'experiments/round126/plan.json')['source_hashes'].items():assert sha(ROOT/src)==h,src
    for stage in ('oof0','oof1','oof2','nested0','nested1','full'):assert read(ROOT/f'experiments/round126/{stage}.json')['optimizer_success']
    summary=dict(results=records,current_recommendation='round104 local',validation_correct=759,validation_n=798,previous_frozen_test_correct=391,test_n=467,goal_required_test_correct=402,goal_active=True,goal_complete=False,test_opened_in_this_phase=False,checks=read(ROOT/'experiments/round132/feature_checks.json'),entity_fit_audit=read(ROOT/'experiments/round128/fit_entity_audit.json'),invariance=reports,local_cpu_screen_seconds=33,official_reference_cpu_seconds=30,official_assets_bytes=16*1024**2)
    dump(ROOT/'experiments/round124_132_summary.json',summary)
    for n in range(124,133):
        d=ROOT/f'experiments/round{n}';assert not (d/'completion_manifest.json').exists();files=[p for p in d.rglob('*') if p.is_file()]+list((ROOT/'research').glob(f'round{n}_*.py'))
        dump(d/'completion_manifest.json',dict(completed=True,test_used_for_tuning=False,hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}))
    dump(ROOT/'experiments/next_after_round132.json',dict(goal_active=True,current_recommendation='round104 local',goal_required_test_correct=402,validation_threshold_strictly_greater_than=759,task='Condition source-quantity relation classification directly on supplied semantic embeddings.',evidence='Current six-way relation classifiers use lexical/heuristic pair features only. Repaired consensus labels in120: old953 versus new973 correct of1830 held relations. Semantic context is currently appended only at final candidate meta stage, not used to predict the quantity-pair operation.',fixed_first_ablation='Use the same5276 repaired consensus labels, same6-class C1 logistic classifier, same parent-balanced mass, same six train partitions. Add question-level mean of the4 supplied candidate embeddings as a context block; compare lexical-only120. No external weights/data, no new neural architecture, no parameter sweep.',normalization='Fit context centering and a single block-RMS scale on allowed TRAIN parents per stage. Match context block mean squared norm to lexical pair block mean squared norm on that stage training pairs; record statistics. No held/VAL examples contribute to scaling. Keep final preprocessing in the bundle.',inference='Question mean is candidate-permutation invariant. New linear context contribution is shared across pairs from the same question; pair lexical features still distinguish the operands and direction. Preserve original root extraction and deterministic executor.',validation='Reproduce104 control759; also report archived repaired-label-only121757. A new formal TEST evaluation requires a strict full-VAL win, frozen assets, complete interface checks and fresh CPU timing.',next_if_context_evidence_positive='Only after the additive context ablation, consider a bounded low-rank interaction between pair features and semantic context if the diagnostic evidence justifies it.',local_cpu_screen_seconds=33,official_reference_cpu_seconds=30,official_assets_bytes=16777216,test_error_analysis_forbidden=True))
    print(dict(results=records,invariance=reports,goal_active=True))
