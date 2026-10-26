"""TRAIN-only data-quality experiment. Original data and prior rounds are immutable."""
from data_access import ROOT, CODE, read_jsonl, parse_answer
from pathlib import Path
import datetime, hashlib, json, os, sys
import numpy as np

OUT = ROOT / 'experiments/round20'


def no_validation(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace('\\', '/').lower()
        if name.endswith(('gsm8k_validation.jsonl', 'gsm8k_validation.npz')):
            raise PermissionError('Round20 is restricted to TRAIN; official validation remains closed.')


sys.addaudithook(no_validation)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(text): return hashlib.sha256(text.encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def plan(): return read(OUT / 'plan.json')
def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')


def normalized(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-5)


def subset(bank, arrays, y, ix):
    return [bank[i] for i in ix], {'embeddings': arrays['embeddings'][ix].copy()}, y[ix]


def load_part(name, embeddings=True):
    assert name in ('fit', 'development', 'audit')
    if name == 'audit':
        assert (OUT / 'frozen.json').exists(), 'Freeze models and selections before audit.'
    ids = np.asarray(plan()['partitions'][name], int)
    rows = read_jsonl(CODE / 'data/candidates/gsm8k_train.jsonl')
    refs = read_jsonl(CODE / 'data/gsm8k_train.jsonl')
    bank, labels = [], []
    for i in ids:
        row, ref = rows[i], refs[i]
        assert row['id'] == ref['id'] and row['question'] == ref['question']
        target = parse_answer(ref['answer'])
        assert target is not None
        # Strip IDs and any non-input fields at the model boundary.
        bank.append(dict(question=row['question'], candidates=[dict(text=c['text'], index=j) for j, c in enumerate(row['candidates'])]))
        labels.append([parse_answer(c['text']) == target for c in row['candidates']])
    y = np.asarray(labels, np.float32)
    assert (y.sum(1) > 0).all()
    arrays = {}
    if embeddings:
        with np.load(CODE / 'data/candidates/gsm8k_train.npz', allow_pickle=False) as z:
            e = z['embeddings']
            assert e.shape == (len(rows) * 4, 1024)
            arrays['embeddings'] = e.reshape(len(rows), 4, 1024)[ids].copy()
    return bank, arrays, y, ids


def paired(choice, base, y, hard):
    a = y[np.arange(len(y)), choice].astype(bool)
    b = y[np.arange(len(y)), base].astype(bool)
    changed = choice != base
    return dict(n=len(y), correct=int(a.sum()), accuracy=float(a.mean()), hard_n=int(hard.sum()),
                hard_correct=int(a[hard].sum()), repaired=int((a & ~b).sum()), damaged=int((~a & b).sum()),
                net=int(a.sum() - b.sum()), changed=int(changed.sum()),
                right_to_right=int((changed & a & b).sum()), wrong_to_wrong=int((changed & ~a & ~b).sum()))


def prepare():
    from round14_prepare import family
    from round15_prepare import exclusions
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'plan.json').exists()
    bank = read_jsonl(CODE / 'data/candidates/gsm8k_train.jsonl')
    previous_audit = set()
    for r in (18, 19):
        previous_audit.update(read(ROOT / f'experiments/round{r}/plan.json')['partitions']['audit'])
    seen = previous_audit | exclusions()
    for r in (15, 17):
        seen.update(map(int, read(ROOT / f'experiments/round{r}/plan.json')['groups']))
    seen.update(read(ROOT / 'experiments/round16/plan.json')['fresh_audit_parents'])
    groups = {}
    for i, row in enumerate(bank): groups.setdefault(digest(family(row['question'])), []).append(i)
    audit, audit_groups = [], set()
    for k in sorted(groups, key=lambda k: digest('r20-quality-audit:' + k)):
        if set(groups[k]) & seen: continue
        audit.extend(groups[k]); audit_groups.add(k)
        if len(audit) >= 950: break
    dev, dev_groups = [], set()
    for k in sorted(groups, key=lambda k: digest('r20-quality-dev:' + k)):
        if k in audit_groups or set(groups[k]) & previous_audit: continue
        dev.extend(groups[k]); dev_groups.add(k)
        if len(dev) >= 650: break
    fit = sorted(i for k, v in groups.items() if k not in audit_groups | dev_groups for i in v)
    parts = dict(fit=fit, development=sorted(dev), audit=sorted(audit))
    assert len(audit) >= 950 and len(dev) >= 650
    assert len(set(sum(parts.values(), []))) == sum(map(len, parts.values())) == 4755
    assert not previous_audit & set(dev + audit)
    cfg = next(c for c in read(ROOT / 'experiments/round19/plan.json')['neural'] if c['name'] == 'regularized_long')
    p = dict(round=20, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), seed=20020,
        partitions=parts, group_hashes={str(i): k for k,v in groups.items() for i in v},
        split_caveat='Fresh fitting isolation, not untouched external evidence: earlier experiments used other TRAIN subsets. Round18/19 audits enter FIT only.',
        method='Same five-member baseline + regularized_long joint comparator, fixed alpha=0.1. Baseline-only and fixed rbf_g3.0_C0.1 alpha=0.1 are diagnostics.',
        primary_head='long', neural=dict(cfg, seed=19019), kernel=dict(pca=96, anchors=256, gamma_factor=3., C=.1, seed=19019),
        alpha=.1, baseline_stack_C=.1,
        variants=['full', 'evidence_only', 'quality_05', 'quality_10', 'random_matched'],
        maximum_removal_fraction=.10, quality_fractions=[.05, .10],
        filter_unit='Whole question with all four real candidates. Original files unchanged. No development/audit deletion or relabeling.',
        quality_oof=dict(folds=3, members=[0,3,4], grouped=True, fit_only=True,
                         recipe='Existing embedding MLP, joint lexical model, text ranker; fixed training lengths; all learned preprocessing refit inside each fold.'),
        quality_evidence='Review exact reference arithmetic mismatches and obvious annotation defects on FIT only; uncertain signals are not confirmed wrong labels. Wrong candidates and hard questions are expected.',
        quality_rule=dict(eligible='mixed-label question AND (>=2/3 OOF heads choose a wrong final answer OR mean pair error >=0.45 OR final-answer vote disagreement >=2/3), plus explicitly adjudicated issues',
             rank='adjudicated issue first; then 0.55*mean(sigmoid(negative_max-positive_max in standardized scores)) + 0.25*wrong_vote_fraction + 0.20*mean positive-negative pair error; ties stable hash',
             removal='Top eligible whole questions, at most floor(n_fit*0.05) or floor(n_fit*0.10). No filling with unflagged questions.'),
        random_control='Uniform random removal matched exactly to selected quality arm by number of positives and input hard/easy flag; same seed; all preprocessing refitted.',
        training_budget='Same epochs/early-stopping rules for every arm; fewer retained examples imply fewer updates. Random control diagnoses quantity/regularization effects.',
        selection='Select one quality arm on development long-head correct; ties fewer harms, fewer removed. Full and random are controls. Frozen before audit; no alpha/config search.',
        retention='All candidates retained regardless of significance. Incremental benefit means positive development and audit net; report uncertainty without an arbitrary minimum-5 rule.',
        quality_attribution='Require comparison with size/hardness-matched random removal; do not call uncertain data mislabeled.',
        audit_statistics='One preselected primary vs full; exact paired test and question-template bootstrap CI, secondary descriptive only.',
        official_validation_policy='Not opened in this filtering experiment. Train-side benefit must be established before a subsequent fixed official validation comparison.',
        validation_opened=False, test_opened=False, synthetic_used=False, reference_step_supervision=False)
    dump(OUT / 'plan.json', p)
    print(json.dumps(dict(partitions={k:len(v) for k,v in parts.items()}, prior_audit_fit_only=len(previous_audit), max_remove=int(len(fit)*.10))))


if __name__ == '__main__': prepare()
