"""Freeze first, then run one TRAIN-side audit. Never read validation/test."""
from round18_common import OUT, ROOT, CODE, read_plan, load_part, paired, normalized, sha, no_validation
import argparse, datetime, functools, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
from data_access import forbid_test
from features import metadata
from round3_selector import scores as baseline_scores
from round18_model import load_model, encode_bank, predict_data, LEFT, RIGHT
import round4_gates
round4_gates.trace=functools.lru_cache(maxsize=8192)(round4_gates.trace)


def dump(name, content):
    (OUT/name).write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding='utf-8')


def incumbent_integrity():
    manifest=json.loads((ROOT/'experiments/round4/asset_manifest.json').read_text())
    root=ROOT/'experiments/round4/selected_bundle'
    assert all(sha(root/p)==v['sha256'] for p,v in manifest.items())
    return dict(files=len(manifest),all_original_sha256_unchanged=True)


def checks():
    plan=read_plan(); parts=plan['partitions']; group=plan['group_hashes']
    assert len(set(sum(parts.values(),[])))==sum(map(len,parts.values()))==4755
    for a in parts:
        for b in parts:
            if a!=b:assert not {group[str(i)] for i in parts[a]}&{group[str(i)] for i in parts[b]}
    # Invoke the guard itself without opening or touching any forbidden file.
    for guard,name in [(no_validation,'gsm8k_validation.jsonl'),(forbid_test,'gsm8k_test.npz')]:
        try:guard('open',(name,))
        except PermissionError:pass
        else:raise AssertionError('File access guard failed')
    b,a,y,ids=load_part('development'); torch.set_num_threads(4)
    bs=baseline_scores(b,a,OUT/'baseline')
    with np.load(OUT/'baseline/development.npz') as z:
        base_error=float(np.max(np.abs(bs-z['scores'])))
        assert np.allclose(bs,z['scores'],atol=2e-4,rtol=1e-4)
        assert np.array_equal(round4_gates.apply_gate(b,bs,'parse_and_repair',4),z['choices'])
    records={}
    for mode in ('joint','independent'):
        model,vocab,cfg=load_model(OUT/mode)
        data,_=encode_bank(b[:16],{'embeddings':a['embeddings'][:16]},vocab,cfg)
        clean=[dict(question=r['question'],candidates=[dict(text=c['text']) for c in r['candidates']]) for r in b[:16]]
        clean_data,_=encode_bank(clean,{'embeddings':a['embeddings'][:16]},vocab,cfg)
        assert all(np.array_equal(data[k],clean_data[k]) for k in data)
        s,m=predict_data(model,data)
        with np.load(OUT/mode/'development.npz') as z:
            export_error=float(np.max(np.abs(s-z['scores'][:16])))
            assert np.allclose(s,z['scores'][:16],atol=3e-4,rtol=2e-4)
        perm=[2,0,3,1]
        pd={k:(v[:,perm] if k in ('c','cn','e') else v) for k,v in data.items()}
        ps,pm=predict_data(model,pd)
        perm_error=float(np.max(np.abs(ps-s[:,perm])))
        assert np.allclose(ps,s[:,perm],atol=3e-4,rtol=2e-4)
        duplicate={k:v.copy() for k,v in data.items()}
        for k in ('c','cn','e'):duplicate[k][:,1]=duplicate[k][:,0]
        _,dupm=predict_data(model,duplicate)
        assert np.max(np.abs(dupm[:,0]))<1e-5
        # Utility differences are transitive in the independent control;
        # a joint comparator may have cycles, hence no scalar-only shortcut.
        cycle=float(np.max(np.abs(m[:,0]+m[:,3]-m[:,1])))
        if mode=='independent':assert cycle<1e-4
        assert np.isfinite(s).all() and np.isfinite(m).all()
        records[mode]=dict(cpu_export_error=export_error,permutation_error=perm_error,
                           identical_pair_error=float(np.max(np.abs(dupm[:,0]))),
                           max_pair_cycle=cycle,identifiers_removed_without_change=True)
    result=dict(split_question_and_template_disjoint=True,guard_function_checks=True,
                guard_checks_do_not_open_files=True,baseline_export_error=base_error,
                models=records,incumbent=incumbent_integrity(),passed=True)
    dump('checks.json',result);return result


def freeze():
    assert not (OUT/'frozen.json').exists()
    checked=checks(); plan=read_plan()
    chosen={mode:json.loads((OUT/mode/'selected.json').read_text())['best'] for mode in ('joint','independent')}
    paths=[OUT/'plan.json',OUT/'architecture.json',OUT/'vocabulary.json',OUT/'checks.json']
    paths += [p for p in (OUT/'baseline').rglob('*') if p.is_file()]
    for mode in chosen:paths += [p for p in (OUT/mode).iterdir() if p.is_file()]
    paths += [ROOT/'research'/f for f in ('round18_common.py','round18_baseline.py','round18_model.py','round18_train.py','round18_evaluate.py')]
    # Freeze the existing implementations reused for all baseline/gate outputs.
    paths += list((ROOT/'research').glob('*.py'))
    paths += [CODE/'common.py',ROOT/'starter/method.py']
    paths += [CODE/'data/candidates/gsm8k_train.jsonl',CODE/'data/candidates/gsm8k_train.npz',CODE/'data/gsm8k_train.jsonl']
    paths=sorted(set(paths))
    asset_base=json.loads((OUT/'baseline/done.json').read_text())['inference_asset_bytes']
    assets_joint=sum((OUT/p).stat().st_size for p in ['joint/model.npz','joint/config.json','vocabulary.json'])
    record=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                selected={mode:dict(epoch=r['epoch'],alpha=r['selected']['alpha'],development=r['selected']) for mode,r in chosen.items()},
                primary='joint fusion versus refit baseline',secondary=['independent fusion','standalone joint','joint versus independent'],
                audit_policy='One fixed audit evaluation. No tuning or retraining against this audit afterwards.',
                statistics='Question-template cluster percentile bootstrap, seed 18018, 20000 replicates; exact one-sided paired binomial test.',
                gate=plan['gate'],baseline_inference_bytes=asset_base,joint_extra_bytes=assets_joint,
                primary_inference_bytes_without_packaging=asset_base+assets_joint,
                within_16MiB_asset_limit_before_packaging=asset_base+assets_joint<16*1024**2,
                hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
                checks_passed=checked['passed'],validation_opened=False,test_opened=False)
    dump('frozen.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='hashes'},ensure_ascii=False),flush=True)


def verify_frozen():
    freeze=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/k)==v for k,v in freeze['hashes'].items()), 'Frozen implementation or model changed.'
    return freeze


def paired_stats(choice, base, y, hard, ids):
    result=paired(choice,base,y,hard)
    discord=result['repaired']+result['damaged']
    result['paired_one_sided_p']=float(binomtest(result['repaired'],discord,.5,alternative='greater').pvalue) if discord else 1.
    diff=(y[np.arange(len(y)),choice]-y[np.arange(len(y)),base]).astype(float)
    plan=read_plan();groups={}
    for j,i in enumerate(ids):groups.setdefault(plan['group_hashes'][str(i)],[]).append(j)
    sums=np.array([diff[v].sum() for v in groups.values()]);sizes=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(plan['seed']);boot=[]
    for _ in range(20):
        ix=rng.integers(len(groups),size=(1000,len(groups)))
        boot.extend((100*sums[ix].sum(1)/sizes[ix].sum(1)).tolist())
    result['accuracy_gain_percentage_points']=100*diff.mean()
    result['cluster_bootstrap_95_ci_percentage_points']=np.quantile(boot,[.025,.975]).tolist()
    result['bootstrap_groups']=len(groups)
    return result


def pair_metrics(margin,y):
    diff=y[:,LEFT]-y[:,RIGHT];keep=diff!=0;prod=margin[keep]*diff[keep]
    return dict(n=int(keep.sum()),correct=int((prod>0).sum()),ties=int((prod==0).sum()),
                accuracy=float((prod>0).mean()))


def audit():
    assert not (OUT/'audit_result.json').exists(),'Audit already consumed; read saved outputs.'
    frozen=verify_frozen();torch.set_num_threads(4)
    b,a,y,ids=load_part('audit');meta=metadata(b)
    tick=time.perf_counter();bs=baseline_scores(b,a,OUT/'baseline')
    bc=round4_gates.apply_gate(b,bs,'parse_and_repair',4);base_seconds=time.perf_counter()-tick
    results={'baseline':paired_stats(bc,bc,y,meta['hard'],ids)};predictions={'baseline':bc}
    timing={'baseline_local_warm_seconds':base_seconds};coverages={};pair_quality={}
    saved={'ids':ids,'labels':y,'baseline_scores':bs,'baseline_choice':bc}
    for mode in ('joint','independent'):
        tick=time.perf_counter();model,vocab,cfg=load_model(OUT/mode)
        x,coverage=encode_bank(b,a,vocab,cfg);s,m=predict_data(model,x,batch=cfg['batch'])
        timing[mode+'_load_encode_predict_local_warm_seconds']=time.perf_counter()-tick
        alpha=frozen['selected'][mode]['alpha']
        choice=round4_gates.apply_gate(b,bs+alpha*normalized(s),'parse_and_repair',4)
        single=round4_gates.apply_gate(b,s,'parse_and_repair',4)
        predictions[mode]=choice;predictions[mode+'_standalone']=single
        results[mode]=paired_stats(choice,bc,y,meta['hard'],ids)
        results[mode+'_standalone']=paired_stats(single,bc,y,meta['hard'],ids)
        pair_quality[mode]=pair_metrics(m,y);coverages[mode]=coverage
        saved.update({mode+'_scores':s,mode+'_margins':m,mode+'_choice':choice,mode+'_standalone_choice':single})
    results['joint_vs_independent']=paired_stats(predictions['joint'],predictions['independent'],y,meta['hard'],ids)
    p=results['joint'];gate=frozen['gate']
    gate_checks=dict(min_net=p['net']>=gate['min_net'],one_sided_p=p['paired_one_sided_p']<=gate['paired_one_sided_p_max'],
        positive_bootstrap_lower=p['cluster_bootstrap_95_ci_percentage_points'][0]>0,
        development_gain=frozen['selected']['joint']['development']['net']>0)
    changes=[];decisions=[]
    for i,row in enumerate(b):
        base=int(bc[i]);new=int(predictions['joint'][i])
        rec=dict(train_index=int(ids[i]),id=row['id'],baseline=base,joint=new,independent=int(predictions['independent'][i]),
                 baseline_correct=bool(y[i,base]),joint_correct=bool(y[i,new]),
                 answers=meta['answers'][i],labels=y[i].astype(int).tolist(),
                 baseline_scores=bs[i].tolist(),joint_scores=saved['joint_scores'][i].tolist())
        decisions.append(rec)
        if base!=new:
            rec=dict(rec,question=row['question'],before_text=row['candidates'][base]['text'],after_text=row['candidates'][new]['text'])
            rec['change_type']='repaired' if y[i,new] and not y[i,base] else 'damaged' if y[i,base] and not y[i,new] else 'right_to_right' if y[i,new] else 'wrong_to_wrong'
            changes.append(rec)
    np.savez(OUT/'audit_predictions.npz',**saved)
    for name,rows in [('audit_decisions.jsonl',decisions),('audit_changes.jsonl',changes)]:
        (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    result=dict(partition='sealed TRAIN audit; not official validation',n=len(b),results=results,
                gate_checks=gate_checks,train_side_gate_passed=all(gate_checks.values()),
                pair_quality=pair_quality,coverage=coverages,timing=timing,
                timing_limit='Local warm CPU 4-thread diagnostic. Not a cold official hardware qualification.',
                incumbent=incumbent_integrity(),frozen_sha=sha(OUT/'frozen.json'),
                validation_opened=False,test_opened=False,reference_steps_used=False,synthetic_used=False,
                audit_consumed=True,reuse_for_model_selection_prohibited=True)
    dump('audit_result.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','audit']);arg=parser.parse_args()
    freeze() if arg.stage=='freeze' else audit()


if __name__=='__main__':main()
