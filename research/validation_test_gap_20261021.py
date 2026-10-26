"""Split-gap diagnosis using TRAIN/VAL and already published aggregate TEST scores.

No TEST inputs, references, choices or individual errors are read. Importing
data_access installs the existing guard before any data access. No model is fit.
"""
from data_access import ROOT, CODE, load, parse_answer
from collections import Counter
from pathlib import Path
import datetime
import hashlib
import json
import math
import numpy as np

OUT = ROOT / 'experiments/validation_test_gap_20261021'
READ_FILES = set()


def read(relative):
    path = ROOT / relative
    READ_FILES.add(path)
    return json.loads(path.read_text(encoding='utf-8'))


def accuracy(correct, mask):
    n = int(mask.sum())
    k = int(np.asarray(correct)[mask].sum())
    return dict(correct=k, n=n, accuracy_percent=100*k/n if n else None)


def describe(bank, labels, methods):
    answers = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    distinct = np.asarray([len(set(a)-{None}) for a in answers])
    invalid = np.asarray([a.count(None) for a in answers])
    patterns = np.asarray(['+'.join(map(str, sorted(Counter(a).values(), reverse=True)))
                           if None not in a else 'has_unparsed' for a in answers])
    n_correct = labels.sum(1).astype(int)
    groups = {'all': np.ones(len(bank), bool), 'non_conflict': distinct < 2,
              'conflict': distinct >= 2}
    groups.update({f'distinct_answers_{i}': distinct == i for i in sorted(set(distinct))})
    groups.update({f'correct_candidates_{i}': n_correct == i for i in sorted(set(n_correct))})
    groups.update({f'answer_multiplicity_{p}': patterns == p for p in sorted(set(patterns))})
    scores = {}
    for name, choices in methods.items():
        assert choices.shape == (len(bank),)
        assert np.all((choices >= 0) & (choices < 4))
        correct = labels[np.arange(len(bank)), choices].astype(bool)
        scores[name] = {name: accuracy(correct, mask) for name, mask in groups.items()}
    return dict(n=len(bank), distinct_valid_answers=dict(Counter(map(int, distinct))),
                invalid_candidates=dict(Counter(map(int, invalid))),
                correct_candidate_counts=dict(Counter(map(int, n_correct))),
                answer_multiplicity=dict(Counter(patterns.tolist())), scores=scores)


def wilson(k, n):
    # Descriptive intervals only: repeated VAL selection violates a fresh-test interpretation.
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [100*(center-half), 100*(center+half)]


def main():
    vb, _, vy = load('validation')
    predictions = {}
    for name, source in (
        ('baseline_round25', 'experiments/final_evaluation_20261021/packaging_validation_predictions.json'),
        ('candidate_grpo_round45', 'experiments/round45_grpo_final_test_20261021/validation_predictions.json')):
        p = read(source)
        assert p['split'] == 'validation'
        assert p['ids'] == [r['id'] for r in vb]
        predictions[name] = np.asarray(p['selected'], dtype=int)
    validation = describe(vb, vy, predictions)
    assert validation['scores']['baseline_round25']['all']['correct'] == 752
    assert validation['scores']['candidate_grpo_round45']['all']['correct'] == 750

    tb, _, ty = load('train')
    policy = read('experiments/round22/plan.json')
    ids = np.asarray(policy['training_ids'], dtype=int)
    assert len(ids) == 4743 and len(set(ids)) == 4743
    oof_path = ROOT/'experiments/round38/oof_baseline.npz'
    READ_FILES.add(oof_path)
    with np.load(oof_path, allow_pickle=False) as z:
        assert np.array_equal(z['ids'], ids)
        choices, folds = z['choices'].copy(), z['folds'].copy()
        assert np.array_equal(z['correct'], ty[ids, choices].astype(bool))
    train = describe([tb[i] for i in ids], ty[ids], {'historical_grouped_oof': choices})
    train['oof_caveat'] = ('Historical five-expert model from round22, not cross-fitted round25; '
                           'TRAIN schemas have already been explored. No pristine benchmark claim.')
    train['folds'] = {str(f): accuracy(ty[ids, choices].astype(bool), folds == f)
                      for f in sorted(set(folds))}

    summaries = {
        'baseline_round25': read('experiments/final_evaluation_20261021/summary.json'),
        'candidate_grpo_round45': read('experiments/round45_grpo_final_test_20261021/summary.json'),
    }
    decomposition = {}
    for name, summary in summaries.items():
        s = validation['scores'][name]
        all_acc = s['all']['accuracy_percent']
        hard_acc = s['conflict']['accuracy_percent']
        test_acc = 100*summary['test_correct']/summary['test_n']
        assert s['all']['correct'] == summary['validation_correct']
        decomposition[name] = dict(
            validation_all_percent=all_acc,
            validation_conflict_percent=hard_acc,
            already_recorded_test_correct=summary['test_correct'],
            already_recorded_test_n=summary['test_n'],
            already_recorded_test_percent=test_acc,
            total_gap_pp=all_acc-test_acc,
            validation_mix_component_pp=all_acc-hard_acc,
            remaining_conflict_gap_pp=hard_acc-test_acc,
            mix_component_fraction_of_gap=(all_acc-hard_acc)/(all_acc-test_acc),
            descriptive_wilson95_validation_conflict=wilson(s['conflict']['correct'], s['conflict']['n']),
            descriptive_wilson95_test=wilson(summary['test_correct'], summary['test_n']),
        )

    pairs = read('experiments/round44/constraint_pairs.json')
    partition = read('experiments/round45/diagnostic_partition.json')
    held_ids = set(partition['held_ids'])
    relation_supervision = dict(
        pairs=len(pairs), parent_questions=len({p['train_index'] for p in pairs}),
        actions=dict(Counter(p['action'] for p in pairs)),
        held_pairs=sum(p['train_index'] in held_ids for p in pairs),
        held_parent_questions=len({p['train_index'] for p in pairs if p['train_index'] in held_ids}),
        caveat='Assistant-reviewed source-grounded annotations, not independent human gold.')
    reward = read('experiments/round45/reward_diagnostics.json')
    rollout = next(r for r in reward['rollout_training']
                   if r['stage'] == 'full' and r['level'] == 'candidate' and r['arm'] == 'grpo')
    relation_errors = next(r for r in reward['held_reward']
                           if r['level'] == 'relation' and r['arm'] == 'grpo_relations'
                           and r['group'] == 'baseline_errors')

    report = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope=dict(test_inputs_read=False, test_references_read=False, test_predictions_read=False,
                   new_test_evaluations=0, new_training_runs=0,
                   test_evidence='Two already recorded aggregate scores and public split specification only.'),
        public_rule_source='https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md',
        public_rule='Every TEST group has >=2 different valid parsed answers; TRAIN/VAL include easier groups.',
        train_eligible=train, validation=validation, arithmetic_gap_decomposition=decomposition,
        decomposition_caveat='Descriptive arithmetic, not causal attribution. Finer TEST distribution '
                            'and individual errors were not inspected; residual gap cannot be assigned '
                            'quantitatively to selection bias, distribution differences or sampling noise. '
                            'Wilson intervals are descriptive and do not correct for repeated selection.',
        train_root_cause_review=read('experiments/round38/root_cause_summary.json'),
        relation_supervision=relation_supervision,
        relation_held_original_errors=relation_errors,
        candidate_grpo_full_training=rollout,
        training_reward_coverage=reward['training_reward_coverage'],
        target=dict(test_n=467, minimum_correct_for_97_percent=math.ceil(.97*467),
                    maximum_errors=467-math.ceil(.97*467),
                    net_correct_gain_needed_from_baseline=math.ceil(.97*467)-386),
    )
    READ_FILES.update([Path(__file__), ROOT/'research/data_access.py', CODE.parent/'guide/GUIDE.md'])
    for split in ('train','validation'):
        READ_FILES.update([CODE/'data'/f'gsm8k_{split}.jsonl',
                           CODE/'data/candidates'/f'gsm8k_{split}.jsonl'])
    report['source_sha256'] = {str(p.relative_to(ROOT)).replace('\\','/'):
                              hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(READ_FILES)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('validation','arithmetic_gap_decomposition',
                                           'relation_supervision','target')}, ensure_ascii=False, indent=2))
    print('TRAIN historical OOF:', train['scores']['historical_grouped_oof'])


if __name__ == '__main__':
    main()
