"""Fit three fixed baselines on train; evaluate only validation, never test.

Runs the official prediction and scoring tools in fresh processes. All recorded
times are local measurements, not proof of the reference-machine time limit.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'

import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'DASE7506-MP2-main' / 'code'
sys.path.insert(0, str(CODE))
from common import load_inputs, parse_answer, read_jsonl, score_choices
from method import make_features


def labels(bank, split):
    assert split in ('train', 'validation')
    gold = read_jsonl(CODE / 'data' / f'gsm8k_{split}.jsonl')
    result = []
    for row, ref in zip(bank, gold):
        assert row['id'] == ref['id'] and row['question'] == ref['question']
        answer = parse_answer(ref['answer'])
        result.append([int(parse_answer(c['text']) == answer) for c in row['candidates']])
    return np.asarray(result)


def main():
    out = ROOT / 'experiments' / 'baseline_validation'
    out.mkdir(parents=True, exist_ok=True)
    train, train_arrays = load_inputs('train')
    validation, _ = load_inputs('validation')
    y = labels(train, 'train')
    yv = labels(validation, 'validation')
    hard = np.asarray([len({a for c in row['candidates']
                           if (a := parse_answer(c['text'])) is not None}) >= 2
                       for row in validation])
    result = {
        'status': 'exploratory_validation_only', 'seed': 20261014,
        'python': sys.version, 'numpy': np.__version__, 'sklearn': sklearn.__version__,
        'platform': platform.platform(), 'processor': platform.processor(),
        'threads': 4, 'test_evaluated': False,
        'train_questions': len(train), 'validation_questions': len(validation),
        'validation_conflict_questions': int(hard.sum()),
        'validation_expected_uniform_random_accuracy': float(100 * yv.mean()),
        'validation_first_candidate_accuracy': float(100 * yv[:, 0].mean()),
        'train_correct_candidates_per_question': np.bincount(y.sum(axis=1), minlength=5).tolist(),
        'methods': [],
    }
    for kind in ('majority', 'embedding', 'text', 'hybrid'):
        checkpoint = None
        training_seconds = 0.0
        fit_warnings = []
        if kind != 'majority':
            tick = time.perf_counter()
            x = make_features(train, train_arrays, kind)
            scaler = StandardScaler()
            normalized = scaler.fit_transform(x)
            classifier = LogisticRegression(C=0.1, max_iter=1000, random_state=result['seed'])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                classifier.fit(normalized, y.reshape(-1))
            fit_warnings = [str(w.message) for w in caught]
            # Fold the training-only scaler into the coefficients for NumPy inference.
            weight = classifier.coef_[0] / scaler.scale_
            bias = classifier.intercept_[0] - scaler.mean_ @ weight
            checkpoint = out / f'{kind}.npz'
            np.savez(checkpoint, weight=weight, bias=bias, kind=np.asarray(kind))
            config = {'kind': kind, 'C': 0.1, 'max_iter': 1000, 'seed': result['seed'],
                      'training_split': 'train', 'classifier': 'binary LogisticRegression',
                      'scaler': 'StandardScaler folded into weight and bias'}
            (out / f'{kind}-config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
            training_seconds = time.perf_counter() - tick
        pred_path = out / f'{kind}-predictions.json'
        score_path = out / f'{kind}-score.json'
        command = [sys.executable, str(CODE/'predict.py'), '--method', str(ROOT/'starter'/'method.py'),
                   '--split', 'validation', '--output', str(pred_path)]
        if checkpoint is not None:
            command += ['--checkpoint', str(checkpoint)]
        tick = time.perf_counter()
        subprocess.run(command, cwd=CODE, check=True, capture_output=True, text=True)
        subprocess.run([sys.executable, str(CODE/'score_predictions.py'), '--split', 'validation',
                        '--predictions', str(pred_path), '--output', str(score_path)],
                       cwd=CODE, check=True, capture_output=True, text=True)
        elapsed = time.perf_counter()-tick
        predictions = json.loads(pred_path.read_text())['selected']
        score = json.loads(score_path.read_text())
        chosen_correct = yv[np.arange(len(validation)), predictions]
        entry = {'method': kind, 'official_score': score,
                 'conflict_subset_accuracy': float(100*chosen_correct[hard].mean()),
                 'local_fresh_prediction_plus_scoring_seconds': elapsed,
                 'training_seconds': training_seconds, 'training_warnings': fit_warnings,
                 'checkpoint_bytes': checkpoint.stat().st_size if checkpoint else 0,
                 'configuration_bytes': (out/f'{kind}-config.json').stat().st_size if checkpoint else 0,
                 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None}
        result['methods'].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        (out/'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
