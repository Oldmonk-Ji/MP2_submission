"""Freeze one development-selected primary, then evaluate once on TRAIN audit."""
from round19_common import OUT,ROOT,CODE,read_plan,load_part,dump,sha,paired,normalized,no_validation
from round19_select import selection_key
from round19_model import load_model,encode_bank,predict_data
from round19_kernel import score as kernel_scores
from round3_selector import scores as baseline_scores
from features import metadata
from data_access import forbid_test
import argparse,datetime,json,time
import numpy as np
import torch
from scipy.stats import binomtest
import round4_gates


def encoder(cfg):
    if cfg.get('anchored'):
        from round19_anchored import encode_anchored
        return encode_anchored
    return encode_bank


def score_record(record,bank,arrays):
    path=OUT/record['path']
    if record['family'] in ('kernel','linear_control'):return kernel_scores(bank,arrays,path)
    model,vocab,cfg=load_model(path)
    x,_=encoder(cfg)(bank,arrays,vocab,cfg)
    return predict_data(model,x,batch=cfg['batch'])[0]


def integrity():
    root=ROOT/'experiments/round4/selected_bundle'
    manifest=json.loads((ROOT/'experiments/round4/asset_manifest.json').read_text())
    assert all(sha(root/k)==v['sha256'] for k,v in manifest.items())
    old=json.loads((ROOT/'experiments/round18/frozen.json').read_text())
    assert all(sha(ROOT/k)==v for k,v in old['hashes'].items())
    return dict(incumbent_assets_unchanged=len(manifest),round18_frozen_unchanged=True)


def checks(records):
    torch.set_num_threads(4);plan=read_plan();parts=plan['partitions'];groups=plan['group_hashes']
    assert len(set(sum(parts.values(),[])))==sum(map(len,parts.values()))==4755
    for a in parts:
        for b in parts:
            if a!=b:assert not {groups[str(i)] for i in parts[a]}&{groups[str(i)] for i in parts[b]}
    old_audit=set(json.loads((ROOT/'experiments/round18/plan.json').read_text())['partitions']['audit'])
    assert not old_audit&set(parts['development']+parts['audit'])
    for guard,name in [(no_validation,'gsm8k_validation.jsonl'),(forbid_test,'gsm8k_test.npz')]:
        try:guard('open',(name,))
        except PermissionError:pass
        else:raise AssertionError('Guard failure')
    b,a,_,_=load_part('development');b=b[:12];a={'embeddings':a['embeddings'][:12]}
    perm=[2,0,3,1];pb=[dict(question=r['question'],candidates=[dict(text=r['candidates'][i]['text'],index=j) for j,i in enumerate(perm)]) for r in b]
    pa={'embeddings':a['embeddings'][:,perm]};results={}
    for record in records:
        s=score_record(record,b,a);ps=score_record(record,pb,pa)
        with np.load(OUT/record['path']/'development.npz') as z:stored=z['scores'][:12]
        err=float(np.max(np.abs(s-stored)));pe=float(np.max(np.abs(ps-s[:,perm])))
        assert np.allclose(s,stored,atol=3e-4,rtol=2e-4),(record['name'],err)
        assert np.allclose(ps,s[:,perm],atol=3e-4,rtol=2e-4),(record['name'],pe)
        assert np.isfinite(s).all()
        extra={}
        if record['family']=='neural':
            model,vocab,cfg=load_model(OUT/record['path']);x,_=encoder(cfg)(b,a,vocab,cfg)
            x={k:v.copy() for k,v in x.items()}
            for k in ('c','cn','e','ck'):
                if k in x:x[k][:,1]=x[k][:,0]
            _,m=predict_data(model,x,batch=cfg['batch']);dup=float(np.max(np.abs(m[:,0])))
            assert dup<2e-5;extra['identical_pair_error']=dup
            if cfg.get('anchored'):
                from round19_model import Comparator
                anchor_model,_,_=load_model(OUT/record['path']);anchor_model.cfg=dict(cfg,anchor_bias=0.)
                plain=Comparator(cfg,len(vocab));plain.load_state_dict(anchor_model.state_dict());plain.eval()
                x,_=encoder(cfg)(b,a,vocab,cfg);s0,_=predict_data(anchor_model,x,batch=cfg['batch'])
                sx,_=predict_data(plain,{k:v for k,v in x.items() if k not in ('qk','ck')},batch=cfg['batch'])
                assert np.allclose(s0,sx,atol=2e-5,rtol=2e-5)
                extra['zero_anchor_reproduces_unanchored']=True
        results[record['name']]=dict(export_error=err,permutation_error=pe,**extra)
    base=baseline_scores(b,a,OUT/'baseline')
    with np.load(OUT/'baseline/development.npz') as z:assert np.allclose(base,z['scores'][:12],atol=2e-4,rtol=1e-4)
    result=dict(passed=True,models=results,question_template_isolation=True,old_audit_not_reused_for_evaluation=True,
                access_guard_checks_without_file_opens=True,**integrity())
    dump(OUT/'checks.json',result);return result


def freeze():
    assert not (OUT/'frozen.json').exists();plan=read_plan()
    neural=json.loads((OUT/'neural/selection.json').read_text())
    kernel=json.loads((OUT/'kernel/selection.json').read_text());records=neural+[kernel['selected'],kernel['linear_control']]
    admissible=set(plan['primary_candidates'])
    if (OUT/'amendment_01.json').exists():admissible.update(json.loads((OUT/'amendment_01.json').read_text())['primary_candidates'])
    base_bytes=json.loads((OUT/'baseline/done.json').read_text())['inference_asset_bytes']
    allowed=[r for r in records if (r['name'] in admissible or r['family']=='kernel') and r['asset_bytes']+base_bytes<16*1024**2]
    primary=max(allowed,key=lambda r:selection_key(r['selected']))
    checked=checks(records)
    protocol=dict(primary=primary,secondary=records,selection_completed_before_audit=True,
         gate=plan['gate'],baseline_bytes=base_bytes,total_primary_inference_bytes=base_bytes+primary['asset_bytes'],
         full_fit_if_passes='Refit same five base recipes on all TRAIN, keep development-learned stack weights. Refit selected new model on all TRAIN using fixed selected epochs/configuration and alpha. Freeze before one official validation comparison against old incumbent and matched full-refit base.',
         statistics='One primary exact one-sided paired binomial test and 20000 question-template bootstrap replicates; all other methods descriptive only.',
         after_audit='Never retune or switch primary using audit results. Failed gate keeps official validation closed.',
         time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks_passed=checked['passed'])
    paths=list((ROOT/'research').glob('*.py'))+[CODE/'common.py',ROOT/'starter/method.py']
    paths += [p for p in OUT.rglob('*') if p.is_file()]
    paths += [CODE/'data/candidates/gsm8k_train.jsonl',CODE/'data/candidates/gsm8k_train.npz',CODE/'data/gsm8k_train.jsonl']
    protocol['hashes']={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    dump(OUT/'frozen.json',protocol)
    print('FROZEN PRIMARY',primary['name'],primary['selected'],'bytes',protocol['total_primary_inference_bytes'],flush=True)


def verify_frozen():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/k)==v for k,v in f['hashes'].items()),'Frozen content changed.'
    return f


def stats(choice,base,y,hard,ids):
    result=paired(choice,base,y,hard);n=result['repaired']+result['damaged']
    result['paired_one_sided_p']=float(binomtest(result['repaired'],n,.5,alternative='greater').pvalue) if n else 1.
    diff=y[np.arange(len(y)),choice]-y[np.arange(len(y)),base];plan=read_plan();group={}
    for j,i in enumerate(ids):group.setdefault(plan['group_hashes'][str(i)],[]).append(j)
    sums=np.asarray([diff[g].sum() for g in group.values()]);sizes=np.asarray([len(g) for g in group.values()])
    rng=np.random.default_rng(plan['seed']);boot=[]
    for _ in range(20):
        ii=rng.integers(len(group),size=(1000,len(group)));boot.extend((100*sums[ii].sum(1)/sizes[ii].sum(1)).tolist())
    result['gain_percentage_points']=float(100*diff.mean())
    result['bootstrap_95_ci_percentage_points']=np.quantile(boot,[.025,.975]).tolist()
    return result


def audit():
    assert not (OUT/'audit_result.json').exists(),'Audit is already consumed.'
    f=verify_frozen();torch.set_num_threads(4);b,a,y,ids=load_part('audit');meta=metadata(b)
    t=time.perf_counter();bs=baseline_scores(b,a,OUT/'baseline');bc=round4_gates.apply_gate(b,bs,'parse_and_repair',4)
    timing=dict(baseline_warm_seconds=time.perf_counter()-t);results=dict(baseline=stats(bc,bc,y,meta['hard'],ids));saved=dict(ids=ids,labels=y,baseline_scores=bs,baseline_choice=bc)
    for rec in f['secondary']:
        tick=time.perf_counter();s=score_record(rec,b,a)
        c=round4_gates.apply_gate(b,bs+rec['selected']['alpha']*normalized(s),'parse_and_repair',4)
        single=round4_gates.apply_gate(b,s,'parse_and_repair',4)
        timing[rec['name']]=time.perf_counter()-tick
        results[rec['name']]=dict(**stats(c,bc,y,meta['hard'],ids),standalone=paired(single,bc,y,meta['hard']))
        saved[rec['name']+'_choice']=c;saved[rec['name']+'_scores']=s
        print('AUDIT',rec['name'],{k:v for k,v in results[rec['name']].items() if k!='standalone'},flush=True)
    name=f['primary']['name'];p=results[name];gate=f['gate']
    gc=dict(min_net=p['net']>=gate['min_net'],paired_significance=p['paired_one_sided_p']<=gate['paired_one_sided_p_max'],
            positive_bootstrap_lower=p['bootstrap_95_ci_percentage_points'][0]>0,development_gain=f['primary']['selected']['net']>0)
    changes=[];choice=saved[name+'_choice']
    for i in np.flatnonzero(choice!=bc):
        old,new=int(bc[i]),int(choice[i]);before,after=bool(y[i,old]),bool(y[i,new])
        changes.append(dict(train_index=int(ids[i]),question=b[i]['question'],baseline=old,chosen=new,
            before_correct=before,after_correct=after,answers=meta['answers'][i],
            kind='repaired' if after and not before else 'damaged' if before and not after else 'right_to_right' if after else 'wrong_to_wrong',
            before_text=b[i]['candidates'][old]['text'],after_text=b[i]['candidates'][new]['text']))
    (OUT/'audit_changes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in changes),encoding='utf-8')
    np.savez(OUT/'audit_predictions.npz',**saved)
    result=dict(n=len(b),partition='TRAIN audit, not official validation',primary=name,results=results,gate_checks=gc,
        train_side_gate_passed=all(gc.values()),audit_consumed=True,no_post_audit_selection=True,
        local_warm_cpu_timing=timing,timing_is_not_official_cold_qualification=True,
        reference_steps_used=False,synthetic_used=False,validation_opened=False,test_opened=False,integrity=integrity())
    dump(OUT/'audit_result.json',result);print('PRIMARY GATE',name,gc,'PASS',all(gc.values()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','audit']);a=p.parse_args()
    freeze() if a.stage=='freeze' else audit()
