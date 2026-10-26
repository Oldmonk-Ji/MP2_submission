"""Gate margin scan on the exact104 fusion scores (no retraining).

apply_gate only accepts a proposal when the score gap is within margin_limit;
limit=4 was tuned in round4 on old base scores. On fusion scores the gate nets
+7 (13 repairs / 6 harms) at limit 4 - a tighter limit may keep most repairs
while dropping harms.
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

OUT = ROOT / 'experiments/round213'
LIMITS = [0, 0.25, 0.5, 1, 2, 3, 4, 8]
if __name__ == '__main__':
    threadpool_limits(4)
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    dump(OUT / 'margin_plan.json', dict(
        method='Gate margin_limit scan over exact104 fusion scores; limits in {0,0.25,0.5,1,2,3,4,8}. '
               'Held reported for sanity, VAL decision vs 759.', limits=LIMITS,
        source_sha256=sha(__file__), test_opened=False))
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
    params = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', num_leaves=7,
                  min_data_in_leaf=100, learning_rate=.03, lambda_l2=50, seed=68068, verbosity=-1,
                  num_threads=4, deterministic=True, force_col_wise=True)
    fit = folds != 2
    held = ~fit
    ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel(),
                     group=np.full(fit.sum(), 4), init_score=nbs[fit].ravel())
    hm = lgb.train(params, ds, num_boost_round=200)
    held_s1 = nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4)
    m104 = lgb.Booster(model_file=str(ROOT / 'experiments/round210/bundle_control/meta.txt'))
    val_s1 = vbs + m104.predict(vx.reshape(-1, vx.shape[-1]), num_threads=4).reshape(-1, 4)
    held_bank = [r for r, k in zip(b, held) if k]
    records = []
    for limit in LIMITS:
        hc = apply_gate(held_bank, held_s1, 'parse_and_repair', limit)
        c = apply_gate(v, val_s1, 'parse_and_repair', limit)
        hok = y[held][np.arange(held.sum()), hc]
        ok = vy[np.arange(len(v)), c]
        r = dict(limit=limit, held=int((hok == 1).sum()), val=int((ok == 1).sum()))
        records.append(r)
        print('MARGIN', r, flush=True)
    dump(OUT / 'margin_records.json', dict(records=records, test_opened=False))
    best = max(records, key=lambda r: (r['val'], r['held']))
    dump(OUT / 'margin_selection.json', dict(best=best, records=records, test_opened=False,
                                             selected='round213' if best['val'] > 759 else 'round104'))
    print('MARGIN BEST', best, flush=True)
