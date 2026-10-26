"""Cross-fitted quality triage: uncertainty is a hypothesis, not a wrong-label verdict."""
from round20_common import ROOT,CODE,OUT,plan,load_part,subset,dump,read,digest,normalized
from round20_models import fit_member
from data_access import read_jsonl,parse_answer
from features import metadata
from round4_quantities import arithmetic_steps
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
import argparse,json,re,time
import numpy as np


def oof():
    root=OUT/'quality';root.mkdir(exist_ok=True)
    assert not (root/'oof.npz').exists()
    b,a,y,ids=load_part('fit');p=plan();members=p['quality_oof']['members']
    groups=np.asarray([p['group_hashes'][str(i)] for i in ids])
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=p['seed'])
    scores=np.full((len(b),len(members),4),np.nan,np.float32);folds=np.full(len(b),-1,int);ledger=[]
    for f,(fit,hold) in enumerate(splitter.split(ids,y.sum(1),groups)):
        assert not set(groups[fit])&set(groups[hold]);folds[hold]=f
        fb,fa,fy=subset(b,a,y,fit);hb,ha,_=subset(b,a,y,hold)
        dump(root/f'fold{f}_partition.json',dict(fit_ids=ids[fit].tolist(),hold_ids=ids[hold].tolist()))
        start=time.perf_counter()
        for j,m in enumerate(members):
            s=fit_member(m,fb,fa,fy,hb,ha,root/f'fold{f}'/f'member{m}')
            scores[hold,j]=s
        ledger.append(dict(fold=f,fit=len(fit),hold=len(hold),seconds=time.perf_counter()-start))
        print('OOF FOLD DONE',ledger[-1],flush=True)
    assert np.isfinite(scores).all() and (folds>=0).all()
    np.savez(root/'oof.npz',scores=scores,folds=folds,ids=ids,labels=y)
    dump(root/'oof_ledger.json',ledger)


def arithmetic_flags(text):
    out=[]
    for lhs,rhs,a,b in arithmetic_steps(text,reference=True,strict=True):
        # Ignore the rounding implied by a displayed decimal RHS. Integer RHS
        # mismatches still need semantic review: floor/ceil can be intentional.
        m=re.fullmatch(r'[-+]?\d+\.(\d+)',rhs)
        tolerance=max(1e-6,0.51*10**(-len(m.group(1)))) if m else 1e-6
        if abs(a-b)>tolerance:
            out.append(dict(lhs=lhs,rhs=rhs,computed=a,stated=b))
    return out


def scan():
    root=OUT/'quality';root.mkdir(exist_ok=True)
    b,_,y,ids=load_part('fit',embeddings=False)
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');records=[];duplicates={}
    for row,i in zip(b,ids):
        key=re.sub(r'\s+',' ',row['question'].lower()).strip()
        duplicates.setdefault(key,[]).append(int(i))
    conflicts=set()
    for group in duplicates.values():
        if len({parse_answer(refs[i]['answer']) for i in group})>1:conflicts.update(group)
    for j,(row,i) in enumerate(zip(b,ids)):
        target=parse_answer(refs[i]['answer']);flags=arithmetic_flags(refs[i]['answer'])
        positives=[k for k in range(4) if y[j,k]]
        positive_errors=[]
        for k in positives:
            equations=arithmetic_steps(row['candidates'][k]['text'],strict=True)
            bad=[dict(lhs=l,rhs=r,computed=v,stated=w) for l,r,v,w in equations if abs(v-w)>1e-4*max(1,abs(v),abs(w))]
            if bad:positive_errors.append(dict(candidate=k,flags=bad))
        if flags or int(i) in conflicts:
            records.append(dict(train_index=int(i),question=row['question'],reference=refs[i]['answer'],target=target,
                flags=flags,duplicate_label_conflict=int(i) in conflicts,candidates=row['candidates'],
                candidate_labels=y[j].astype(int).tolist(),positive_trace_flags=positive_errors))
    dump(root/'reference_review_packet.json',records)
    print('REFERENCE FLAGGED FOR REVIEW',len(records),flush=True)
    for r in records:print(json.dumps({k:v for k,v in r.items() if k not in ('candidates','positive_trace_flags')},ensure_ascii=False),flush=True)


def rank(preview=False):
    root=OUT/'quality';assert not (root/'ranking.json').exists()
    b,_,y,ids=load_part('fit',embeddings=False);meta=metadata(b)
    with np.load(root/'oof.npz') as z:
        raw=z['scores'];assert np.array_equal(ids,z['ids']);assert np.array_equal(y,z['labels'])
    s=np.stack([normalized(raw[:,k,:]) for k in range(raw.shape[1])],1)
    votes=np.where(meta['valid'][:,None,:],s,-1e30).argmax(2)
    wrong=(y[np.arange(len(y))[:,None],votes]==0).mean(1)
    risk=[];pairerr=[];disagreement=[]
    for j in range(len(b)):
        pos=np.flatnonzero(y[j]);neg=np.flatnonzero(1-y[j])
        if len(neg):
            margin=s[j][:,pos].max(1)-s[j][:,neg].max(1)
            risk.append(float((1/(1+np.exp(np.clip(margin,-20,20)))).mean()))
            differences=s[j][:,pos,None]-s[j][:,None,neg]
            pairerr.append(float(((differences<0)+.5*(differences==0)).mean()))
        else:risk.append(0.);pairerr.append(0.)
        answer_votes=[meta['answers'][j][int(k)] for k in votes[j]]
        disagreement.append(1-max(Counter(answer_votes).values())/len(answer_votes))
    review=read(root/'adjudication.json') if (root/'adjudication.json').exists() else []
    assert len({r['train_index'] for r in review})==len(review)
    assert set(r['train_index'] for r in review)<=set(ids.tolist())
    adjudicated={r['train_index']:r for r in review}
    records=[]
    for j,i in enumerate(ids):
        decision=adjudicated.get(int(i));issue=bool(decision and decision['decision']=='exclude')
        eligible=issue or (y[j].sum()<4 and (wrong[j]>=2/3-1e-6 or pairerr[j]>=.45 or disagreement[j]>=2/3-1e-6))
        if decision and decision['decision']=='keep':eligible=False
        score=.55*risk[j]+.25*float(wrong[j])+.20*pairerr[j]
        records.append(dict(train_index=int(i),issue_evidence=issue,adjudication=decision,
            eligible=bool(eligible),score=score,wrong_vote_fraction=float(wrong[j]),mean_pair_error=pairerr[j],
            standardized_margin_risk=risk[j],final_answer_vote_disagreement=disagreement[j],
            oof_chosen=votes[j].tolist(),candidate_labels=y[j].astype(int).tolist(),n_positive=int(y[j].sum()),hard=bool(meta['hard'][j]),
            question=row_question(b[j]),status='adjudicated_issue' if issue else 'model_uncertainty_not_proven_label_error'))
    records.sort(key=lambda r:(-int(r['issue_evidence']),-r['score'],digest('r20-tie:'+str(r['train_index']))))
    if preview:
        refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');original={int(i):b[j] for j,i in enumerate(ids)}
        packet=[dict(r,reference=refs[r['train_index']]['answer'],candidates=original[r['train_index']]['candidates']) for r in records[:60]]
        dump(root/'uncertainty_review_packet.json',packet)
        print(json.dumps(dict(preview_questions=len(packet),n_fit=len(b))),flush=True)
        return
    dump(root/'ranking.json',records)
    eligible=[r['train_index'] for r in records if r['eligible']]
    evidence=[r['train_index'] for r in records if r['issue_evidence']]
    assert len(evidence)<=int(len(b)*.1)
    sets=dict(full=[],evidence_only=evidence,quality_05=eligible[:int(len(b)*.05)],quality_10=eligible[:int(len(b)*.10)])
    for name,remove in sets.items():
        assert len(remove)<=int(len(b)*.10)
        retained=[int(i) for i in ids if i not in set(remove)]
        dump(root/f'{name}.json',dict(name=name,fit_ids=retained,removed_ids=remove,removed_fraction=len(remove)/len(b),
            issue_evidence=sum(int(i in set(evidence)) for i in remove),uncertainty_only=sum(int(i not in set(evidence)) for i in remove)))
    summary=dict(n_fit=len(b),oof_members=plan()['quality_oof']['members'],eligible=len(eligible),adjudicated_issues=len(evidence),
        reviewed_protected=sum(r['decision']=='keep' for r in review),
        removal_counts={k:len(v) for k,v in sets.items()},
        oof_correct_by_member=[int(y[np.arange(len(y)),votes[:,k]].sum()) for k in range(votes.shape[1])],
        filter_data='FIT only; no development or audit predictions used')
    dump(root/'summary.json',summary);print(json.dumps(summary),flush=True)
    # A transparent review queue, never fed to inference or evaluation.
    original={int(i):b[j] for j,i in enumerate(ids)}
    dump(root/'removed_training_review.json',[dict(r,candidates=original[r['train_index']]['candidates']) for r in records if r['train_index'] in set(sets['quality_10'])])


def row_question(row):return row['question']


def random_matched(selected):
    root=OUT/'quality';assert not (root/'random_matched.json').exists()
    arm=read(root/f'{selected}.json');records=read(root/'ranking.json');rng=np.random.default_rng(plan()['seed'])
    by_id={r['train_index']:r for r in records};stratum=lambda r:(r['n_positive'],r['hard'])
    counts=Counter(stratum(by_id[i]) for i in arm['removed_ids']);pool={}
    for r in records:pool.setdefault(stratum(r),[]).append(r['train_index'])
    removed=[]
    for k,n in sorted(counts.items()):removed.extend(map(int,rng.choice(sorted(pool[k]),n,replace=False)))
    fit=plan()['partitions']['fit'];assert len(removed)==len(arm['removed_ids'])
    dump(root/'random_matched.json',dict(name='random_matched',matched_to=selected,fit_ids=[i for i in fit if i not in set(removed)],
        removed_ids=sorted(removed),removed_fraction=len(removed)/len(fit),strata_counts={str(k):v for k,v in counts.items()},
        overlap_with_quality=len(set(removed)&set(arm['removed_ids'])),selection='Uniform random within positive-count/hardness strata, not quality scored.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['oof','scan','preview','rank']);a=p.parse_args()
    {'oof':oof,'scan':scan,'preview':lambda:rank(preview=True),'rank':rank}[a.stage]()
