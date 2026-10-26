"""Train-only projected quadratic and question/candidate interaction ranker."""
import json, re
from pathlib import Path
import numpy as np
from features import prepare
from round3_features import grounding
from round4_gates import apply_gate
from round22_selector import normalized


def question_docs(bank):
    return [re.sub(r'\d[\d,]*(?:\.\d+)?', ' number ', r['question'].lower()) for r in bank]


def transform(bank, arrays, space, vocabulary):
    from sklearn.feature_extraction.text import TfidfVectorizer
    p = prepare(bank, arrays)
    e = np.clip(((p['embedding']-space['pca_mean'])@space['pca'].T)/space['pca_scale'], -8, 8)
    vec = TfidfVectorizer(vocabulary=vocabulary, ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
    vec.idf_ = space['idf']
    q = np.clip((vec.transform(question_docs(bank))@space['q_svd'].T)/space['q_scale'], -8, 8)
    small = np.concatenate([p['base'], p['numeric'][:, :, 5:], p['checks2'], grounding(bank)], 2)
    small = np.clip((small-space['small_mean'])/space['small_scale'], -8, 8)
    return e.astype(np.float32), q.astype(np.float32), small.astype(np.float32)


def design(e, q, small, kind):
    n, k, d = e.shape
    mean = e.mean(1, keepdims=True)
    parts = [e/np.sqrt(d), small/np.sqrt(small.shape[-1])]
    if kind != 'linear':
        left, right = np.triu_indices(d)
        parts.append(e[:, :, left]*e[:, :, right]/np.sqrt(len(left)))
    if kind not in ('linear', 'no_interaction'):
        parts.append((e[:, :, :, None]*mean[:, :, None, :]).reshape(n, k, -1)/d)
    if kind in ('question', 'no_interaction'):
        parts.append((e[:, :, :, None]*q[:, None, None, :]).reshape(n, k, -1)/np.sqrt(d*q.shape[-1]))
    return np.concatenate(parts, 2).astype(np.float32)


def branch_scores(bank, arrays, checkpoint):
    path = Path(checkpoint)
    cfg = json.loads((path/'config.json').read_text())
    vocabulary = json.loads((path/'vocabulary.json').read_text())
    with np.load(path/'space.npz', allow_pickle=False) as z:
        space = {k: z[k] for k in z.files}
    e, q, small = transform(bank, arrays, space, vocabulary)
    x = design(e, q, small, cfg['kind'])
    with np.load(path/'model.npz', allow_pickle=False) as z:
        return x@z['weight']


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    from round25_selector import scores
    path = Path(checkpoint)
    cfg = json.loads((path/'fusion.json').read_text())
    base = scores(bank, arrays, path/'baseline')
    delta = branch_scores(bank, arrays, path/'branch')
    s = base + cfg['alpha']*normalized(delta)
    return apply_gate(bank, s, 'parse_and_repair', 4).astype(int).tolist()
