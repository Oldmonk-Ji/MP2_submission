"""Fusion hyperparameter grid on the exact-104 control feature readout.

The fixed 200-tree/7-leaf/100-min-data config has never been re-tuned. Grid
trains on fit folds (nested features, clean) and evaluates held plus complete
official VAL. TEST stays sealed; a config must beat 759 on VAL to matter.
"""
from round22_common import ROOT, load, read, dump, sha
from round83_selector import extend
from round80_selector import augment
from round79_selector import design, project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from threadpoolctl import threadpool_limits
import numpy as np, lightgbm as lgb, time

OUT = ROOT / 'experiments/round211'
CONFIGS = [
    ('baseline', dict(num_leaves=7, min_data_in_leaf=100, learning_rate=.03, num_boost_round=200, lambda_l2=50)),
    ('l15', dict(num_leaves=15, min_data_in_leaf=100, learning_rate=.03, num_boost_round=400, lambda_l2=50)),
    ('l15m50', dict(num_leaves=15, min_data_in_leaf=50, learning_rate=.03, num_boost_round=400, lambda_l2=50)),
    ('l31', dict(num_leaves=31, min_data_in_leaf=100, learning_rate=.03, num_boost_round=400, lambda_l2=50)),
    ('l31m50', dict(num_leaves=31, min_data_in_leaf=50, learning_rate=.03, num_boost_round=800, lambda_l2=50)),
    ('l7m50', dict(num_leaves=7, min_data_in_leaf=50, learning_rate=.015, num_boost_round=800, lambda_l2=50)),
    ('l63', dict(num_leaves=63, min_data_in_leaf=100, learning_rate=.03, num_boost_round=800, lambda_l2=50)),
    ('l15lr015', dict(num_leaves=15, min_data_in_leaf=100, learning_rate=.015, num_boost_round=800, lambda_l2=50)),
    ('l31m50lr015', dict(num_leaves=31, min_data_in_leaf=50, learning_rate=.015, num_boost_round=800, lambda_l2=50)),
    ('l7lr05', dict(num_leaves=7, min_data_in_leaf=100, learning_rate=.05, num_boost_round=400, lambda_l2=50)),
    ('l15l2-10', dict(num_leaves=15, min_data_in_leaf=100, learning_rate=.03, num_boost_round=400, lambda_l2=10)),
    ('l31m50l2-10', dict(num_leaves=31, min_data_in_leaf=50, learning_rate=.03, num_boost_round=800, lambda_l2=10)),
]
if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    dump(OUT / 'plan.json', dict(
        method='Grid over LightGBM fusion hyperparameters on the exact104 control readout. '
               'Trained on fit folds with nested features; evaluated on held and complete VAL. Design evidence only.',
        configs=[c for n, c in CONFIGS],
        source_sha256=sha(__file__), test_opened=False))
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    weights = [m['weight'] for m in read(ROOT / 'experiments/round25/selected_bundle/config.json')['members']]
    cache = {}
    for split in ('train', 'validation'):
        with np.load(ROOT / f'experiments/round78/{split}_features.npz') as z:
            cache[split] = (z['root'], z['extra'])
    with np.load(ROOT / 'experiments/round68/oof.npz') as z:
        assert np.array_equal(z['ids'], ids)
        ss = z['components']
    with np.load(ROOT / 'experiments/round77/nested/components.npz') as z:
        assert np.array_equal(z['ids'], ids)
        nested = z['components']
        folds = z['folds']
    with np.load(ROOT / 'experiments/round79/full_space.npz') as w:
        context = project(a, w)
        vc = project(va, w)
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
    fixed = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', seed=68068, verbosity=-1,
                 num_threads=4, deterministic=True, force_col_wise=True)
    fit = folds != 2
    held = ~fit
    records = []
    for name, cfg in CONFIGS:
        t0 = time.perf_counter()
        params = dict(fixed, **cfg)
        rounds = params.pop('num_boost_round')
        ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel(),
                         group=np.full(fit.sum(), 4), init_score=nbs[fit].ravel())
        hm = lgb.train(params, ds, num_boost_round=rounds)
        hc = apply_gate([r for r, k in zip(b, held) if k],
                        nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4),
                        'parse_and_repair', 4)
        held_correct = int(y[held][np.arange(held.sum()), hc].sum())
        ds = lgb.Dataset(x.reshape(-1, x.shape[-1]), label=y.ravel(), group=np.full(len(b), 4),
                         init_score=bs.ravel())
        m = lgb.train(params, ds, num_boost_round=rounds)
        c = apply_gate(v, vbs + m.predict(vx.reshape(-1, vx.shape[-1])).reshape(-1, 4), 'parse_and_repair', 4)
        bc = apply_gate(v, vbs, 'parse_and_repair', 4)
        ok = vy[np.arange(len(v)), c]
        bo = vy[np.arange(len(v)), bc]
        r = dict(name=name, params=cfg, held=held_correct, val=int(ok.sum()),
                 repairs=int(((ok == 1) & (bo == 0)).sum()), harms=int(((ok == 0) & (bo == 1)).sum()),
                 seconds=time.perf_counter() - t0)
        records.append(r)
        print('TUNE', r, flush=True)
    dump(OUT / 'tuning_records.json', records)
    best = max(records, key=lambda r: (r['val'], r['held'], -r['harms']))
    dump(OUT / 'tuning_selection.json', dict(best=best, all_records=records, test_opened=False,
                                              selected='round211' if best['val'] > 759 else 'round104'))
    print('BEST', best, flush=True)
