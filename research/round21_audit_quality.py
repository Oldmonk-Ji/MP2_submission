"""Post-hoc audit sensitivity; never fit a model or alter the original audit.

All model outcomes were previously available, so this is not a new blind audit.
Review packets omit model predictions/correctness. Uncertainty ranking never uses
references or correctness. Every model is evaluated on identical retained IDs.
"""
from round20_common import ROOT, CODE, OUT as PREVIOUS, read, dump, sha, plan, read_jsonl, parse_answer
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import re
import numpy as np

OUT = ROOT / 'experiments/round21_audit_quality'
NAMES = ('full', 'evidence_only', 'quality_05', 'quality_10', 'random_matched')
HEADS = ('baseline', 'long', 'kernel')


def stable(i):
    return hashlib.sha256(f'round21-audit-quality:{i}'.encode()).hexdigest()


def source_fingerprints():
    files = [PREVIOUS / f for f in ('frozen.json', 'plan.json', 'audit_predictions.npz', 'audit_result.json')]
    files += [CODE / 'data/gsm8k_train.jsonl', CODE / 'data/candidates/gsm8k_train.jsonl']
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'plan.json').exists()
    # No labels/choices are loaded at this stage. Full-data model fixed in advance.
    with np.load(PREVIOUS / 'audit_predictions.npz', allow_pickle=False) as z:
        ids = z['ids'].astype(int)
        scores = z['full_long_scores'].astype(float)
    assert ids.tolist() == plan()['partitions']['audit']
    assert len(ids) == 950
    bank = read_jsonl(CODE / 'data/candidates/gsm8k_train.jsonl')
    ranking = []
    for i, s in zip(ids, scores):
        answers = [parse_answer(c['text']) for c in bank[i]['candidates']]
        # Candidate max respects the selector's argmax; equivalent answers merge.
        groups = {}
        for a, v in zip(answers, s):
            key = a if a is not None else '<unparsable>'
            groups[key] = max(groups.get(key, -math.inf), float(v))
        values = sorted(groups.values(), reverse=True)
        margin = (values[0] - values[1]) / max(float(np.std(s)), 1e-8) if len(values) > 1 else None
        ranking.append(dict(train_index=int(i), answer_groups=len(groups), margin=margin,
                            all_answers_identical=len(groups) == 1))
    ranking.sort(key=lambda r: (r['margin'] is None, r['margin'] or 0., stable(r['train_index'])))
    cap = math.floor(.1 * len(ids))
    uncertain = [r['train_index'] for r in ranking[:cap]]
    # Semantic flags only nominate review cases, never automatically exclude them.
    patterns = {
        'pricing_scope': r'\b(?:for \$|costs? \$|at \$)',
        'relative_scope': r'\b(?:more than|less than|half as|twice as|times as)',
        'time_or_boundary': r'\b(?:first|last|remaining|before|after|until|between|average|at least|at most|round)\b',
        'discrete_or_percent': r'\b(?:percent|percentage|people|tickets|books|students|children|animals|days|weeks)\b|%',
    }
    flags = {int(i): [k for k, p in patterns.items() if re.search(p, bank[i]['question'], re.I)] for i in ids}
    semantic_order = sorted((int(i) for i in ids if i not in uncertain), key=lambda i: (-len(flags[i]), stable(i)))
    semantic = semantic_order[:40]
    first25 = [int(i) for i in ids[:25]]
    selected = set(uncertain + semantic + first25)
    random = sorted((int(i) for i in ids if i not in selected), key=stable)[:20]
    selected.update(random)
    p = dict(round=21, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Correct user scope: filter AUDIT, not FIT. Post-hoc quality/selective-accuracy sensitivity only.',
        n=950, maximum_removed=95, model_training=False, models_unchanged=True,
        prior_outcomes_seen=True, not_blind_or_independent_new_evidence=True,
        audit_ids=ids.tolist(), uncertainty_model='full_long_scores (frozen pre-gate scores)',
        uncertainty_rule='Ascending (highest answer-group max score - runner-up) / candidate score standard deviation. All-equivalent answers last. SHA256 stable ties. No correctness/reference input.',
        uncertainty_is_not_bad_data=True,
        review_selection=dict(top_uncertainty=uncertain, semantic=semantic, first25=first25, random=random),
        review_scope='Question/reference reviews plus real candidates for flagged cases; limited to nominated IDs. Other IDs are unreviewed, not certified clean.',
        defect_criteria='Exclude only an inconsistent reference target/constraints, or materially underspecified wording with at least two reasonable answers. Ordinary difficulty, wrong candidates, irrelevant numbers, or implausible-but-coherent premises do not establish a defect.',
        no_relabeling=True, no_audit_items_added_to_training=True,
        views=['full', 'defects_only', 'defects_and_ambiguities', 'uncertainty_05', 'uncertainty_10', 'quality_then_uncertainty_10'],
        quality_priority='Confirmed defects first, then material ambiguities; stable ID hash if cap binds. Need not fill cap for quality-only views.',
        primary_view='quality_then_uncertainty_10: quality IDs first then label-free uncertainty ranking up to 95 total',
        fair_comparison='The identical exclusion mask applies to all 15 frozen model/head choices. Never remove only errors or retune per model.',
        validation_opened=False, test_opened=False, input_hashes=source_fingerprints())
    dump(OUT / 'plan.json', p)
    dump(OUT / 'uncertainty_ranking.json', ranking)
    refs = read_jsonl(CODE / 'data/gsm8k_train.jsonl')
    packet = []
    for i in sorted(selected):
        assert refs[i]['question'] == bank[i]['question']
        packet.append(dict(train_index=i, question=bank[i]['question'], reference=refs[i]['answer'],
            reference_answer=parse_answer(refs[i]['answer']), semantic_review_flags=flags[i]))
    dump(OUT / 'review_packet_no_predictions.json', packet)
    print(json.dumps(dict(n=950, cap=cap, nominated=len(packet), prediction_or_correctness_in_packet=False)))


def show(start, end):
    packet = read(OUT / 'review_packet_no_predictions.json')
    for r in packet[start:end]:
        print(f"{r['train_index']} | {r['question']} | TARGET {r['reference_answer']}")


def inspect(ids):
    bank = read_jsonl(CODE / 'data/candidates/gsm8k_train.jsonl')
    refs = read_jsonl(CODE / 'data/gsm8k_train.jsonl')
    allowed = set(read(OUT / 'plan.json')['audit_ids'])
    for i in ids:
        assert i in allowed
        print(json.dumps(dict(train_index=i, question=bank[i]['question'], reference=refs[i]['answer'],
                              candidates=[c['text'] for c in bank[i]['candidates']]), ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'show', 'inspect'])
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=40)
    ap.add_argument('--ids', type=int, nargs='*', default=[])
    a = ap.parse_args()
    if a.stage == 'prepare': prepare()
    elif a.stage == 'show': show(a.start, a.end)
    elif a.stage == 'inspect': inspect(a.ids)
