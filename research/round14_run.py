"""Train local support control, freeze, then open TRAIN audit (never validation).

Two truth layers must remain separate: deterministic source-grammar entailment
is weak supervision; official final-answer correctness is NOT semantic gold.
"""
from data_access import ROOT, CODE, read_jsonl, parse_answer
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from collections import Counter
from fractions import Fraction as F
import numpy as np
from round14_local import extract, equation, status, inspect

OUT=ROOT/'experiments/round14'
CODE_FILES=['round14_local.py','round14_run.py','round14_prepare.py','round14_checks.py']


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_baseline():
    base=ROOT/'experiments/round4/selected_bundle'
    manifest=json.loads((ROOT/'experiments/round4/asset_manifest.json').read_text())
    for name,v in manifest.items():
        assert digest(base/name)==v['sha256'],name
        assert (base/name).stat().st_size==v['bytes']
    files=[p for p in base.rglob('*') if p.is_file()]
    assert {str(p.relative_to(base)) for p in files}==set(manifest)
    return dict(bytes=sum(p.stat().st_size for p in files),files=len(files),hashes_match=True)


def features(source,claim,with_evidence=True):
    k=claim['kind'];v=float(F(claim['value']));lt=float(F(claim['left_time']));rt=float(F(claim['right_time']))
    x=[float(k==a) for a in ('age_fact','age_ratio','age_difference')]+[math.copysign(math.log1p(abs(v)),v),np.tanh(lt/10),np.tanh(rt/10)]
    if not with_evidence:return x
    matches=[float(source.get(a)==claim.get(a)) for a in ('kind','left','right','value','left_time','right_time','unit')]
    # A small learned structural matcher, not a learned text parser. This is
    # deliberately reported as a control for the source grammar's supervision.
    return x+matches+[float(all(matches[:4])),float(all(matches)),float(source['coefficients']==claim['coefficients']),float(source['rhs']==claim['rhs'])]


def mutations(source,names):
    args=[source['kind'],source['left'],source['right'],F(source['value']),F(source['left_time']),F(source['right_time'])]
    variants=[('identity',list(args))]
    for idx,label in [(3,'quantity'),(4,'left_time'),(5,'right_time')]:
        if idx==5 and args[2] is None:continue
        for delta in (-1,1):
            other=list(args);other[idx]+=delta;variants.append((label,other))
    for idx,label in [(1,'left_entity'),(2,'right_entity')]:
        if args[idx] is None:continue
        for name in names:
            if name!=args[idx]:
                other=list(args);other[idx]=name;variants.append((label,other))
    seen=set()
    for kind,a in variants:
        c=equation(*a)
        key=json.dumps({k:c[k] for k in ('kind','left','right','value','left_time','right_time')},sort_keys=True)
        if key in seen:continue
        seen.add(key)
        truth=status([source],c)
        yield kind,c,truth


def examples(bank,groups,split):
    items=[]
    for i,row in enumerate(bank):
        if groups[str(i)]['split']!=split:continue
        rels,_=extract(row['question']);names=sorted({k for e in rels for k in e['coefficients']})
        for source in rels:
            for kind,claim,truth in mutations(source,names):
                items.append(dict(train_index=i,group=groups[str(i)]['group'],mutation=kind,
                                  source=source,claim=claim,weak_label=truth,semantic_gold=False))
    return items


def fit_model(items, evidence, name):
    from sklearn.linear_model import LogisticRegression
    x=np.asarray([features(a['source'],a['claim'],evidence) for a in items],dtype=np.float64)
    y=np.array([a['weak_label']=='entailed' for a in items],dtype=int)
    # Equal source-group mass, all descendants remain in their parent's split.
    counts=Counter(a['train_index'] for a in items)
    weights=np.array([1/counts[a['train_index']] for a in items]);weights*=len(weights)/weights.sum()
    model=LogisticRegression(C=1,solver='lbfgs',max_iter=500,class_weight='balanced',random_state=20261014)
    model.fit(x,y,sample_weight=weights)
    np.savez(OUT/f'{name}.npz',weight=model.coef_[0],bias=model.intercept_[0])
    return dict(training_examples=len(items),training_parent_questions=len(counts),features=x.shape[1],asset_bytes=(OUT/f'{name}.npz').stat().st_size)


def probability(source,claim,evidence,name):
    with np.load(OUT/f'{name}.npz') as m:
        z=float(np.asarray(features(source,claim,evidence))@m['weight']+m['bias'])
    return 1/(1+math.exp(-max(-50,min(50,z))))


def evaluate(items,evidence,name):
    with np.load(OUT/f'{name}.npz') as m:
        x=np.asarray([features(a['source'],a['claim'],evidence) for a in items],dtype=np.float64)
        if not len(items):return dict(n=0,accuracy=None)
        p=1/(1+np.exp(-np.clip(x@m['weight']+m['bias'],-50,50)))
    y=np.asarray([a['weak_label']=='entailed' for a in items]);choice=p>=.5
    positives=p>=.99
    return dict(n=len(items),parent_questions=len({a['train_index'] for a in items}),
                accuracy=float((choice==y).mean()),balanced_accuracy=float(((choice[y]==y[y]).mean()+(choice[~y]==y[~y]).mean())/2),
                high_confidence_emitted=int(positives.sum()),high_confidence_precision=float(y[positives].mean()) if positives.any() else None,
                note='Agreement with deterministic source-grammar labels on controlled mutations; NOT natural semantic accuracy.')


def build_records(bank,groups,split):
    records=[]
    for i,row in enumerate(bank):
        if groups[str(i)]['split']!=split:continue
        texts=[c['text'] for c in row['candidates']];answers=[parse_answer(t) for t in texts]
        result=inspect(row['question'],texts,answers)
        if not result['source_relations'] and not result['unresolved']:continue
        records.append(dict(train_index=i,question_id=row['id'],question=row['question'],partition=split,
                            group=groups[str(i)]['group'],candidate_answers=answers,**result))
    return records


def record_metrics(records,refs):
    metrics=Counter();problems=[]
    for r in records:
        metrics['questions_with_relations_or_unknown']+=1
        metrics['source_relations']+=len(r['source_relations'])
        metrics['contrast_pairs']+=len(r['contrasts'])
        metrics['questions_with_contrasts']+=bool(r['contrasts'])
        metrics['ambiguous_questions']+=r['ambiguous']
        truth=parse_answer(refs[r['train_index']]['answer'])
        for e in r['source_relations']:
            for name in ('evidence','claim_span'):
                s=e[name];assert r['question'][s['start']:s['end']]==s['text']
                metrics['exact_original_spans']+=1
        for c in r['claims']:
            metrics['claims']+=1;metrics['claim_'+c['status']]+=1
            if not c['claim'].get('final_target_claim'):continue
            a=r['candidate_answers'][c['candidate']]
            metrics['final_'+c['status']]+=1
            if c['status'] in ('entailed','contradicted'):
                good=(c['status']=='entailed')==(a==truth)
                metrics['final_decisions']+=1;metrics['final_agrees_official']+=good
                if not good:problems.append(dict(train_index=r['train_index'],candidate=c['candidate'],status=c['status'],answer=a,reference=truth))
    metrics=dict(metrics)
    metrics['official_final_label_agreement']=metrics.get('final_agrees_official',0)/metrics['final_decisions'] if metrics.get('final_decisions') else None
    metrics['semantic_precision']=None
    metrics['semantic_precision_note']='Original substring integrity is checked; semantic entailment still needs independent review. Final numeric agreement is a different metric.'
    return metrics,problems


def main(phase):
    plan=json.loads((OUT/'plan.json').read_text());assert plan['allowed_data']==['train']
    # Explicit filename whitelist, no data-directory scans, no validation load.
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl')
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    assert all(a['id']==b['id'] and a['question']==b['question'] for a,b in zip(bank,refs))
    started=time.perf_counter()
    if phase=='fit':
        if (OUT/'frozen.json').exists():raise RuntimeError('Frozen round exists; do not silently refit after audit.')
        info=dict(baseline=verify_baseline(),plan_sha256=digest(OUT/'plan.json'))
        items=examples(bank,plan['groups'],'fit')
        (OUT/'fit_mutations.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items),encoding='utf-8')
        info['models']={}
        for ev,name in [(True,'local_support'),(False,'no_question_evidence')]:
            info['models'][name]=fit_model(items,ev,name)
            info['models'][name]['development']=evaluate(examples(bank,plan['groups'],'development'),ev,name)
        for split in ('fit','development'):
            records=build_records(bank,plan['groups'],split);m,errors=record_metrics(records,refs)
            save(f'{split}_contrasts.json',records);save(f'{split}_diagnostic.json',dict(metrics=m,final_label_disagreements=errors))
        info['training_seconds']=time.perf_counter()-started
        save('training.json',info)
        frozen=dict(files={name:digest(ROOT/'research'/name) for name in CODE_FILES},
                    assets={name:digest(OUT/name) for name in ['local_support.npz','no_question_evidence.npz','plan.json']},
                    phase='frozen before new-module audit evaluation',annotation_truth='weak; not human gold')
        save('frozen.json',frozen);print(json.dumps(info),flush=True)
    else:
        frozen=json.loads((OUT/'frozen.json').read_text())
        for name,h in frozen['files'].items():assert digest(ROOT/'research'/name)==h,name
        for name,h in frozen['assets'].items():assert digest(OUT/name)==h,name
        records=build_records(bank,plan['groups'],'audit');m,errors=record_metrics(records,refs)
        items=examples(bank,plan['groups'],'audit')
        models={name:evaluate(items,ev,name) for ev,name in [(True,'local_support'),(False,'no_question_evidence')]}
        save('audit_contrasts.json',records)
        result=dict(natural_diagnostic=m,final_label_disagreements=errors,synthetic_controls=models,
                    audit_seconds=time.perf_counter()-started,baseline=verify_baseline(),
                    independent_human_semantic_labels=0,validation_access=False,test_access=False)
        save('audit_result.json',result)
        reason=['No independently reviewed semantic gold for this new claim/span schema.',
                'Training-source mutations measure matching of the parser output, not learning robust natural-language binding.',
                'No fair outer baseline/challenger retraining; no end-to-end gain established.']
        if m.get('questions_with_contrasts',0)<plan['gates']['min_natural_audit_questions']:reason.append('Natural contrast coverage below predeclared minimum.')
        if errors:reason.append('Natural final-claim labels disagree with official answers; extraction requires review.')
        decision=dict(mode='diagnostic_only',override_enabled=False,validation_allowed=False,test_access=False,
                      semantic_gate_passed=False,reasons=reason,incumbent='research/round4_selector.py',
                      checkpoint='experiments/round4/selected_bundle',ar_generator='not started: semantic gate unmet',
                      learned_support_asset_bytes=(OUT/'local_support.npz').stat().st_size,
                      source_freeze_sha256=digest(OUT/'frozen.json'))
        save('deployment.json',decision)
        print(json.dumps(result),flush=True);print(json.dumps(decision),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fit','audit']);main(p.parse_args().phase)
