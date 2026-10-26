"""Freeze the deletion experiment, then compare all arms on an unchanged TRAIN audit."""
from round20_common import ROOT,CODE,OUT,plan,load_part,read,dump,sha,paired,no_validation
from round20_models import score_all,predict
from data_access import forbid_test
from features import metadata
from scipy.stats import binomtest
from collections import Counter
import round4_gates
import argparse,datetime,time
import numpy as np
import torch

NAMES=('full','evidence_only','quality_05','quality_10','random_matched')
HEADS=('baseline','long','kernel')


def integrity():
    manifest=read(ROOT/'experiments/round4/asset_manifest.json')
    assert all(sha(ROOT/'experiments/round4/selected_bundle'/k)==v['sha256'] for k,v in manifest.items())
    for n in (18,19):
        frozen=read(ROOT/f'experiments/round{n}/frozen.json')
        assert all(sha(ROOT/k)==v for k,v in frozen['hashes'].items()),f'round{n} changed'
    return dict(incumbent_unchanged=True,prior_round18_19_frozen_unchanged=True,original_train_data_unchanged=True)


def checks():
    p=plan();parts=p['partitions'];groups=p['group_hashes'];fit=set(parts['fit'])
    assert len(set(sum(parts.values(),[])))==sum(map(len,parts.values()))==4755
    for a in parts:
        for b in parts:
            if a!=b:assert not {groups[str(i)] for i in parts[a]}&{groups[str(i)] for i in parts[b]}
    old=set()
    for n in (18,19):old.update(read(ROOT/f'experiments/round{n}/plan.json')['partitions']['audit'])
    assert not old&set(parts['development']+parts['audit'])
    seen=[]
    for f in range(3):
        fold=read(OUT/f'quality/fold{f}_partition.json');a=set(fold['fit_ids']);b=set(fold['hold_ids'])
        assert not a&b and a|b==fit
        assert not {groups[str(i)] for i in a}&{groups[str(i)] for i in b}
        seen+=fold['hold_ids']
    assert sorted(seen)==sorted(fit)
    review=read(OUT/'quality/adjudication.json');protected={r['train_index'] for r in review if r['decision']=='keep'}
    evidence={r['train_index'] for r in review if r['decision']=='exclude'}
    for name in NAMES:
        mask=read(OUT/f'quality/{name}.json');a=set(mask['fit_ids']);b=set(mask['removed_ids'])
        assert a|b==fit and not a&b and len(b)<=int(.1*len(fit))
        assert read(OUT/f'runs/{name}/training_subset.json')['fit_ids']==sorted(a)
        if name not in ('full','random_matched'):
            assert not protected&b and evidence<=b
    selected=read(OUT/'selection.json')['primary']['name'];rank={r['train_index']:r for r in read(OUT/'quality/ranking.json')}
    def strata(name):return Counter((rank[i]['n_positive'],rank[i]['hard']) for i in read(OUT/f'quality/{name}.json')['removed_ids'])
    assert strata(selected)==strata('random_matched')
    for guard,name in ((no_validation,'gsm8k_validation.jsonl'),(forbid_test,'gsm8k_test.npz')):
        try:guard('open',(name,))
        except PermissionError:pass
        else:raise AssertionError('Access guard failure')
    b,a,_,_=load_part('development');b=b[:12];a={'embeddings':a['embeddings'][:12]}
    perm=[2,0,3,1];pb=[dict(question=r['question'],candidates=[dict(text=r['candidates'][i]['text'],index=j) for j,i in enumerate(perm)]) for r in b]
    pa={'embeddings':a['embeddings'][:,perm]};model_checks={};torch.set_num_threads(4)
    for name in NAMES:
        path=OUT/'runs'/name;s=score_all(b,a,path);ps=score_all(pb,pa,path);result={}
        with np.load(path/'development.npz') as z:
            for head in HEADS:
                err=float(np.max(np.abs(s[head]-z[head+'_scores'][:12])));pe=float(np.max(np.abs(ps[head]-s[head][:,perm])))
                assert np.allclose(s[head],z[head+'_scores'][:12],atol=4e-4,rtol=2e-4),(name,head,err)
                assert np.allclose(ps[head],s[head][:,perm],atol=4e-4,rtol=2e-4),(name,head,pe)
                assert np.isfinite(s[head]).all()
                result[head]=dict(export_error=err,permutation_error=pe)
            assert np.array_equal(predict(b,a,path,head='long'),z['long_choice'][:12])
        sizes=read(path/'done.json')['total_inference_bytes']
        assert sizes['long']<=16*1024**2 and sizes['kernel']<=16*1024**2
        model_checks[name]=dict(heads=result,inference_bytes=sizes)
    result=dict(passed=True,question_template_isolation=True,oof_predictions_never_in_fit=True,
        deletion_cap=True,reviewed_clean_hard_cases_protected=True,random_strata_matched=True,
        development_and_audit_not_filtered=True,guards_checked_without_opening_forbidden_files=True,
        models=model_checks,**integrity())
    dump(OUT/'checks.json',result);return result


def freeze():
    assert not (OUT/'frozen.json').exists();assert all((OUT/f'runs/{n}/done.json').exists() for n in NAMES)
    # Make inference self-contained; it must not read training split/ID tables
    # just to obtain a scalar fusion setting. This also counts its asset bytes.
    for name in NAMES:
        path=OUT/'runs'/name
        dump(path/'fusion.json',dict(alpha=plan()['alpha'],gate='parse_and_repair',margin_limit=4))
        r=read(path/'done.json');models=r['models'];size=(path/'fusion.json').stat().st_size
        r['total_inference_bytes']={h:models['baseline']['inference_asset_bytes']+models[h]['inference_asset_bytes']+size for h in ('long','kernel')}
        dump(path/'done.json',r)
    checked=checks();sel=read(OUT/'selection.json')
    protocol=dict(primary=sel['primary']['name'],primary_head='long',selection=sel,
        all_models_and_masks_frozen_before_audit=True,checks_passed=checked['passed'],
        audit_policy='One pass; all 950 questions retained; no post-audit hyperparameter, mask, checkpoint or primary switching.',
        statistics='Primary paired test and 20000 template bootstrap replicates; all other comparisons descriptive.',
        retention='Retain all positive and negative candidates. Significance is reported, not a deletion rule.',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    paths=list((ROOT/'research').glob('*.py'))+[CODE/'common.py']
    paths+=[p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.log']
    paths+=[CODE/'data/candidates/gsm8k_train.jsonl',CODE/'data/candidates/gsm8k_train.npz',CODE/'data/gsm8k_train.jsonl']
    protocol['hashes']={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    dump(OUT/'frozen.json',protocol)
    print('FROZEN primary',protocol['primary'],'all',len(NAMES),'runs',flush=True)


def stats(c,base,y,hard,ids):
    result=paired(c,base,y,hard);n=result['repaired']+result['damaged']
    result['paired_one_sided_p']=float(binomtest(result['repaired'],n,.5,alternative='greater').pvalue) if n else 1.
    diff=y[np.arange(len(y)),c]-y[np.arange(len(y)),base];g={};p=plan()
    for j,i in enumerate(ids):g.setdefault(p['group_hashes'][str(i)],[]).append(j)
    sums=np.asarray([diff[ii].sum() for ii in g.values()]);sizes=np.asarray([len(ii) for ii in g.values()])
    rng=np.random.default_rng(p['seed']);boot=[]
    for _ in range(20):
        ii=rng.integers(len(g),size=(1000,len(g)));boot.extend((100*sums[ii].sum(1)/sizes[ii].sum(1)).tolist())
    result['gain_percentage_points']=float(100*diff.mean())
    result['bootstrap_95_ci_percentage_points']=np.quantile(boot,[.025,.975]).tolist()
    return result


def audit():
    assert not (OUT/'audit_result.json').exists()
    f=read(OUT/'frozen.json');assert all(sha(ROOT/k)==v for k,v in f['hashes'].items())
    b,a,y,ids=load_part('audit');meta=metadata(b);torch.set_num_threads(4)
    saved=dict(ids=ids,labels=y);timing={};all_choices={}
    for name in NAMES:
        start=time.perf_counter();ss=score_all(b,a,OUT/'runs'/name);all_choices[name]={}
        for head,s in ss.items():
            c=round4_gates.apply_gate(b,s,'parse_and_repair',4);all_choices[name][head]=c
            saved[name+'_'+head+'_scores']=s;saved[name+'_'+head+'_choice']=c
        timing[name]=time.perf_counter()-start
        print('AUDIT',name,{h:int(y[np.arange(len(y)),c].sum()) for h,c in all_choices[name].items()},flush=True)
    results={}
    for name in NAMES:
        results[name]={h:stats(all_choices[name][h],all_choices['full'][h],y,meta['hard'],ids) for h in HEADS}
    primary=f['primary'];pr=results[primary]['long'];dev=f['selection']['primary']
    against_random=stats(all_choices[primary]['long'],all_choices['random_matched']['long'],y,meta['hard'],ids)
    changes=[];old=all_choices['full']['long']
    for name in NAMES:
        if name=='full':continue
        c=all_choices[name]['long']
        for j in np.flatnonzero(c!=old):
            before,after=bool(y[j,old[j]]),bool(y[j,c[j]])
            changes.append(dict(variant=name,train_index=int(ids[j]),question=b[j]['question'],
                before_index=int(old[j]),after_index=int(c[j]),before_correct=before,after_correct=after,
                kind='repaired' if after and not before else 'damaged' if before and not after else 'right_to_right' if after else 'wrong_to_wrong',
                answers=meta['answers'][j],before_text=b[j]['candidates'][old[j]]['text'],after_text=b[j]['candidates'][c[j]]['text']))
    (OUT/'audit_changes.jsonl').write_text(''.join(__import__('json').dumps(r,ensure_ascii=False)+'\n' for r in changes),encoding='utf-8')
    np.savez(OUT/'audit_predictions.npz',**saved)
    result=dict(n=len(b),partition='TRAIN audit, not official validation',primary=primary,results=results,
        primary_vs_matched_random=against_random,positive_development_and_audit_gain=dev['net']>0 and pr['net']>0,
        positive_quality_gain_over_random=against_random['net']>0,
        candidate_retained=True,incumbent_replaced=False,validation_opened=False,test_opened=False,
        local_cpu_warm_seconds_all_three_heads=timing,timing_not_official_cold_qualification=True,
        no_audit_filtering=True,no_post_audit_selection=True,**integrity())
    dump(OUT/'audit_result.json',result)
    print('PRIMARY',primary,pr,'VERSUS RANDOM',against_random,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','audit']);a=p.parse_args()
    freeze() if a.stage=='freeze' else audit()
