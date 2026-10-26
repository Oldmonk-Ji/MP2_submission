"""Selector: exact104 fusion + gate-first, then strict-gap flip rule (tau=3.0, top-2 restricted).

Freeze candidate from round213_scan/round214: VAL 760 (+1 over 759), HELD flat.
The strict score comes from the round210 contrastive weight over round209
strict-trace features; flip only when the strict argmax is inside the raw
top-2, the gap exceeds tau, and the gate did not override the argmax.
"""
from pathlib import Path
import json, numpy as np
from round136_features import build, summarize
from round210_features import build as mean210_features
from round82_relations import probabilities
from round80_selector import augment
from round79_selector import project, design as previous
from round68_selector import components
from round4_gates import apply_gate


def extend(x, relation):
    return np.concatenate([x, relation, relation - relation.mean(1, keepdims=True)], 2)


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    import lightgbm as lgb
    p = Path(checkpoint)
    cfg = json.loads((p / 'config.json').read_text())
    ss, g = components(bank, arrays, p / 'baseline')
    root, extra, flow, pairs, ptr, packets, structure = build(bank, cfg['structure_arm'])
    with np.load(p / 'space.npz') as w:
        context = project(arrays, w)
    with np.load(p / 'flow.npz') as w:
        score = np.asarray(flow @ w['weight']).reshape(-1, 4)
    x, bs = previous(bank, ss, cfg['weights'], g, root, extra, context, 'context_influence')
    x = augment(x, score)
    with np.load(p / 'relation.npz') as w:
        prob = probabilities(pairs, w)
    x = extend(x, summarize(prob, ptr, packets))
    with np.load(p / 'structure.npz') as w:
        score = np.asarray(structure @ w['weight']).reshape(-1, 4)
    x = augment(x, score)
    m = lgb.Booster(model_file=str(p / 'meta.txt'))
    s1 = bs + m.predict(x.reshape(-1, x.shape[-1]), num_threads=4).reshape(-1, 4)
    base = apply_gate(bank, s1, 'parse_and_repair', 4)
    with np.load(p / 'strict.npz') as w:
        strict = np.asarray(mean210_features(bank) @ w['weight']).reshape(-1, 4)
    r = np.arange(len(bank))
    s_arg = strict.argmax(1)
    top2 = np.argsort(-s1, axis=1)[:, :2]
    in_top2 = np.asarray([s_arg[i] in top2[i] for i in r])
    gap = strict[r, s_arg] - strict[r, base]
    flip = (s_arg != base) & (gap > cfg['strict_tau']) & in_top2 & (base == s1.argmax(1))
    sel = base.copy()
    sel[flip] = s_arg[flip]
    return sel.astype(int).tolist()
