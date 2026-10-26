"""MLP fusion over the exact104 feature surface (different model family).

sklearn MLPClassifier with cross-entropy over the 4 candidates, mirroring the
LightGBM structure: model sees the 708-dim candidate features plus the row
base score, predicts a per-candidate delta; selection = argmax(base + delta).

Fit-fold training with nested features for held sanity; full-train oof fit for
VAL. Any config must beat 759 before TEST.
"""
from round22_common import ROOT, load, read, dump, sha
from round83_selector import extend
from round80_selector import augment
from round79_selector import design, project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from threadpoolctl import threadpool_limits
from sklearn.neural_network import MLPClassifier
import numpy as np, time

OUT = ROOT / 'experiments/round212'
CONFIGS = [
    ('h256', (256,)),
    ('h256x64', (256, 64)),
    ('h512x128', (512, 128)),
    ('h256s', (256,)),
]


def build_features():
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    weights = [m['weight'] for m in read(ROOT / 'experiments/round25/selected_bundle/config.json')['members']]
    cache = {}
    for split in ('train', 'validation'):
        with np.load(ROOT / f'experiments/round78/{split}_features.npz') as z:
            cache[split] = (z['root'], z['extra'])
    with np.load(ROOT / 'experiments/round68/oof.npz') as z:
        ss = z['components']
    with np.load(ROOT / 'experiments/round77/nested/components.npz') as z:
        nested = z['components']; folds = z['folds']
    with np.load(ROOT / 'experiments/round79/full_space.npz') as w:
        context = project(a, w); vc = project(va, w)
    with np.load(ROOT / 'experiments/round79/diagnostic_space.npz') as w:
        nc = project(a, w)
    g = grounding(b)
    vs, vg = components(v, va, ROOT / 'experiments/round25/selected_bundle')
    x, bs = design(b, ss, weights, g, *cache['train'], context, 'context_influence')
    nx, nbs = design(b, nested, weights, g, *cache['train'], nc, 'context_influence')
    vx, vbs = design(v, vs, weights, vg, *cache['validation'], vc, 'context_influence')
    with np.load(ROOT / 'experiments/round80/unsigned_scores.npz') as z:
        x = augment(x, z['oof']); nx = augment(nx, z['nested']); vx = augment(vx, z['validation'])
    with np.load(ROOT / 'experiments/round83/relation_scores.npz') as z:
        x = extend(x, z['oof']); nx = extend(nx, z['nested']); vx = extend(vx, z['validation'])
    with np.load(ROOT / 'experiments/round103/local_scores.npz') as z:
        x = augment(x, z['oof']); nx = augment(nx, z['nested']); vx = augment(vx, z['validation'])
    x = np.concatenate([x, bs[:, :, None]], 2)
    nx = np.concatenate([nx, nbs[:, :, None]], 2)
    vx = np.concatenate([vx, vbs[:, :, None]], 2)
    with np.load(ROOT / 'experiments/round210/scores.npz') as z:
        strict = z['oof'], z['nested'], z['validation']
    return b, v, y, vy, x, nx, vx, bs, nbs, vbs, folds, strict


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    dump(OUT / 'plan.json', dict(
        method='MLP cross-entropy fusion on exact104 features plus row base score. '
               'Nested-fit for held sanity; oof full-train fit for VAL. TEST sealed until >759.',
        configs=[c for n, c in CONFIGS], source_sha256=sha(__file__), test_opened=False))
    b, v, y, vy, x, nx, vx, bs, nbs, vbs, folds, strict = build_features()
    fit = folds != 2
    held = ~fit
    labels = y.argmax(1)
    vlabels = vy.argmax(1)
    records = []
    for name, hidden in CONFIGS:
        t0 = time.perf_counter()
        ax, an, av = x, nx, vx
        if name.endswith('s'):
            ax = np.concatenate([ax, strict[0][:, :, None]], 2)
            an = np.concatenate([an, strict[1][:, :, None]], 2)
            av = np.concatenate([av, strict[2][:, :, None]], 2)
        m = MLPClassifier(hidden_layer_sizes=hidden, activation='relu', solver='adam', alpha=1e-4,
                          batch_size=1024, learning_rate_init=1e-3, max_iter=400,
                          early_stopping=True, n_iter_no_change=20, validation_fraction=0.05,
                          random_state=68068, verbose=False)
        m.fit(an[fit].reshape(-1, an.shape[-1]), np.repeat(labels[fit], 4))
        hprob = m.predict_proba(an[held].reshape(-1, an.shape[-1])).reshape(-1, 4, 4)
        hdiag = hprob[np.arange(held.sum())[:, None], np.arange(4)[None, :], np.arange(4)[None, :]]
        hscores = nbs[held] + hdiag
        hc = apply_gate([r for r, k in zip(b, held) if k], hscores, 'parse_and_repair', 4)
        held_correct = int(y[held][np.arange(held.sum()), hc].sum())
        m2 = MLPClassifier(hidden_layer_sizes=hidden, activation='relu', solver='adam', alpha=1e-4,
                           batch_size=1024, learning_rate_init=1e-3, max_iter=400,
                           early_stopping=True, n_iter_no_change=20, validation_fraction=0.05,
                           random_state=68068, verbose=False)
        m2.fit(ax.reshape(-1, ax.shape[-1]), np.repeat(labels, 4))
        vprob = m2.predict_proba(av.reshape(-1, av.shape[-1])).reshape(-1, 4, 4)
        vdiag = vprob[np.arange(len(v))[:, None], np.arange(4)[None, :], np.arange(4)[None, :]]
        vscores = vbs + vdiag
        c = apply_gate(v, vscores, 'parse_and_repair', 4)
        ok = vy[np.arange(len(v)), c]
        bc = apply_gate(v, vbs, 'parse_and_repair', 4)
        bo = vy[np.arange(len(v)), bc]
        r = dict(name=name, hidden=list(hidden), held=held_correct, val=int((ok == 1).sum()),
                 repairs=int(((ok == 1) & (bo == 0)).sum()),
                 harms=int(((ok == 0) & (bo == 1)).sum()),
                 iterations=(int(m.n_iter_), int(m2.n_iter_)),
                 seconds=time.perf_counter() - t0)
        records.append(r)
        print('MLP', r, flush=True)
    dump(OUT / 'records.json', records)
    best = max(records, key=lambda r: (r['val'], r['held'], -r['harms']))
    dump(OUT / 'selection.json', dict(best=best, records=records, test_opened=False,
                                      selected='round212' if best['val'] > 759 else 'round104'))
    print('MLP BEST', best, flush=True)
