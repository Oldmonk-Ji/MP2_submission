"""Freeze TRAIN-only groups and gates before inspecting this round's audit.

The split is independent for the NEW module only: the incumbent has already
seen all train questions. No end-to-end generalization claim is licensed.
"""
from data_access import ROOT, CODE, read_jsonl
import hashlib
import json
import re

OUT = ROOT / 'experiments/round14'


def family(question):
    s = re.sub(r"\b[A-Z][a-z]+(?:'s)?\b", '<person>', question)
    s = re.sub(r'\d[\d,]*(?:\.\d+)?', '<number>', s)
    s = re.sub(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty)\b', '<number>', s, flags=re.I)
    return re.sub(r'\s+', ' ', s.lower()).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'plan.json').exists():
        print('Existing plan retained.'); return
    rows = read_jsonl(CODE / 'data/candidates/gsm8k_train.jsonl')
    groups = {}
    for i, row in enumerate(rows):
        key = hashlib.sha256(family(row['question']).encode()).hexdigest()
        bucket = int(hashlib.sha256(('r14-local-20261014:' + key).encode()).hexdigest()[:8], 16) % 10
        split = 'fit' if bucket < 6 else 'development' if bucket < 8 else 'audit'
        groups[str(i)] = dict(group=key, split=split)
    plan = dict(schema='mp2-local-contrast-plan/v1', seed=20261014,
                groups=groups, allowed_data=['train'], validation_allowed=False,
                audit_kind='new-module-only, not independent of previous baseline or earlier exploratory rounds',
                grouping_limit='Exact normalized template grouping; not a guarantee against all paraphrase duplicates.',
                hypothesis='Local candidate claims aligned to exact source spans expose relation conflicts without full graph extraction.',
                domains=['explicit named-person age facts and ratios, ordered endpoints and time'],
                primary_model='from-scratch local binary logistic support classifier; deterministic training-source mutations',
                controls=['exact local rule', 'same classifier without question evidence'],
                gates=dict(min_natural_audit_claims=50, min_natural_audit_questions=30,
                           min_human_semantic_precision=.98, min_human_reviewed=50,
                           min_audit_label_accuracy=.98, min_evidence_ablation_gain=.05,
                           min_fair_outer_net_gain=1, min_fair_outer_changes=20,
                           invariants_required=True),
                stop='If semantic supervision, natural audit, or fair comparison is insufficient, deploy diagnostic mode only; no validation and no answer overrides.',
                annotation_policy='Source grammar labels are weak supervision, never human gold; official answer labels are a separate diagnostic.',
                ar_generation='Deferred until natural semantic audit passes; no pretrained or external model calls.')
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    counts={s:sum(g['split']==s for g in groups.values()) for s in ('fit','development','audit')}
    (OUT / 'split_summary.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts))
    # Only development-fit texts are displayed; no audit texts/references opened.
    selected=[(i,r['question']) for i,r in enumerate(rows) if groups[str(i)]['split']=='fit' and re.search(r'\b(?:years old|older than|as old as|age of)\b',r['question'],re.I)]
    print(json.dumps(selected[:30], ensure_ascii=True, indent=2))


if __name__ == '__main__': main()
