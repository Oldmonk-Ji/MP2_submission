"""CPU-only reference selector; written with Codex assistance, 2026-10-14.

Uses the supplied common.parse_answer unchanged. Inference reads no references.
This is an exploratory baseline, not a finished coursework submission.
"""
from collections import Counter
from pathlib import Path
import re

import numpy as np
from common import parse_answer


def text_features(bank):
    rows = []
    for row in bank:
        candidates = row['candidates']
        answers = [parse_answer(c['text']) for c in candidates]
        counts = Counter(a for a in answers if a is not None)
        question_words = set(re.findall(r'[a-z]+', row['question'].lower()))
        lengths = [len(c['text']) for c in candidates]
        mean_length = max(float(np.mean(lengths)), 1.0)
        for c, answer, length in zip(candidates, answers, lengths):
            text = c['text']
            words = set(re.findall(r'[a-z]+', text.lower()))
            rows.append([
                float(answer is not None),
                counts.get(answer, 0) / 4.0,
                len(counts) / 4.0,
                max(counts.values(), default=0) / 4.0,
                np.log1p(length),
                length / mean_length,
                np.log1p(len(text.split())),
                np.log1p(text.count('\n')),
                np.log1p(text.count('=')),
                np.log1p(sum(ch.isdigit() for ch in text)),
                float('####' in text),
                float('\\boxed{' in text),
                len(words & question_words) / max(len(question_words), 1),
                float(bool(re.search(r'\b(wait|however|mistake|reconsider)\b', text.lower()))),
            ])
    return np.asarray(rows, dtype=np.float64)


def make_features(bank, arrays, kind):
    if kind == 'text':
        return text_features(bank)
    embeddings = arrays['embeddings'].reshape(-1, arrays['embeddings'].shape[-1])
    if kind == 'embedding':
        return embeddings.astype(np.float64)
    if kind == 'hybrid':
        return np.column_stack((embeddings, text_features(bank)))
    raise ValueError(f'Unknown feature kind: {kind}')


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    # No checkpoint means deterministic answer majority voting.
    if checkpoint is None:
        choices = []
        for row in bank:
            answers = [parse_answer(c['text']) for c in row['candidates']]
            counts = Counter(a for a in answers if a is not None)
            scores = [counts.get(a, 0) if a is not None else -1 for a in answers]
            choices.append(int(np.argmax(scores)))
        return choices
    with np.load(Path(checkpoint), allow_pickle=False) as model:
        x = make_features(bank, arrays, str(model['kind'].item()))
        scores = x @ model['weight'] + float(model['bias'].item())
    return [int(i) for i in scores.reshape(len(bank), 4).argmax(axis=1)]
