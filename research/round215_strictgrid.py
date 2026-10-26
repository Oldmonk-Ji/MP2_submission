"""Strict-score retraining grid over regularization C, then strict-gap rule sweep.

The strict-gap rule's only working configuration (tau=3.0, top-2 restricted,
VAL 760) hinges on the round210 contrastive weight trained at C=1. A C-grid may
yield sharper gaps and more decisive flips. C=1 must reproduce round210
bit-exactly as a pipeline sanity check. TEST stays sealed.
"""
from round22_common import ROOT, load, read, dump, sha
from round80_train import pair_data
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np, time

OUT = ROOT / 'experiments/round215'
CS = [0.1, 0.3, 1.0, 3.0, 10.0]


def fit(x, y, ix, ids, path, C=1.0):
    rows = (ix[:, None] * 4 + np.arange(4)).ravel()
    xx, yy, ww = pair_data(x[rows], y[ix])
    m = LogisticRegression(C=C, fit_intercept=False, solver='liblinear', max_iter=1500,
                           random_state=80080).fit(xx, yy, sample_weight=ww)
    w = m.coef_[0].astype(np.float32)
    np.savez(path, weight=w)
    dump(path.with_suffix('.json'), dict(training_ids=ids[ix].tolist(), C=C,
                                         iterations=m.n_iter_.tolist(),
                                         labels='candidate final-answer correctness; not human semantic gold'))
    return w


def run_fit(C, tag):
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    x = sparse.load_npz(ROOT / 'experiments/round210/train_features.npz')
    vx = sparse.load_npz(ROOT / 'experiments/round210/validation_features.npz')
    with np.load(ROOT / 'experiments/round77/nested/components.npz') as z:
        folds = z['folds']
        assert np.array_equal(ids, z['ids'])
    oof = np.zeros((len(b), 4)); nested = np.zeros_like(oof)
    for stage in ('oof0', 'oof1', 'oof2', 'nested0', 'nested1', 'full'):
        held = int(stage[-1]) if stage != 'full' else None
        ix = np.arange(len(b)) if held is None else np.flatnonzero(folds == 1 - held) \
            if stage.startswith('nested') else np.flatnonzero(folds != held)
        w = fit(x, y, ix, ids, OUT / f'{tag}_{stage}.npz', C=C)
        if held is not None:
            te = np.flatnonzero(folds == held)
            rr = (te[:, None] * 4 + np.arange(4)).ravel()
            (nested if stage.startswith('nested') else oof)[te] = np.asarray(x[rr] @ w).reshape(-1, 4)
        else:
            val = np.asarray(vx @ w).reshape(-1, 4)
        print('FIT', tag, stage, flush=True)
    nested[folds == 2] = oof[folds == 2]
    np.savez(OUT / f'scores_{tag}.npz', oof=oof, nested=nested, validation=val)
    return oof, nested, val


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    dump(OUT / 'plan.json', dict(
        method='C-grid retraining of the round210 strict contrastive weight on the same 65536-dim '
               'features; then strict-gap rule sweep per C. Design on fit folds via clean oof scores, '
               'held sanity, VAL decision. C=1 checked bit-exact against round210.', cs=CS,
        source_sha256=sha(__file__), test_opened=False))
    for C in CS:
        tag = f'c{str(C).replace(".", "")}'
        if (OUT / f'scores_{tag}.npz').exists():
            print('SKIP', tag, flush=True)
            continue
        run_fit(C, tag)
    # sanity: C=1 matches round210 bit-exactly
    with np.load(OUT / 'scores_c10.npz') as z:
        with np.load(ROOT / 'experiments/round210/scores.npz') as ref:
            for key in ('oof', 'nested', 'validation'):
                assert np.array_equal(z[key], ref[key]), key
    print('C=1 bit-exact OK', flush=True)
    dump(OUT / 'fit_complete.json', dict(seconds=time.perf_counter() - tick, test_opened=False))
