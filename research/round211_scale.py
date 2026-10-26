"""Delta-scale and temperature scan over the exact104 fusion (no retraining).

The fusion output is base + delta where delta = learned tree correction. If the
fusion is too timid or too aggressive, scaling delta by alpha or sharpening the
row softmax by temperature can recover points without touching the model.
Evaluated on held (nested features, fit-trained booster) and complete VAL.
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
ALPHAS = [0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
TEMPS = [1.0, 2.0, 4.0]
if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    dump(OUT / 'scale_plan.json', dict(
        method='Alpha scaling and temperature sharpening of the exact104 fusion on held and VAL. '
               'No retraining; candidate argmax over softmax((base + alpha*delta)/T).',
        alphas=ALPHAS, temps=TEMPS, source_sha256=sha(__file__), test_opened=False))
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
    hdelta = hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4)
    m = lgb.Booster(model_file=str(ROOT / 'experiments/round210/bundle_control/meta.txt'))
    vdelta = m.predict(vx.reshape(-1, vx.shape[-1]), num_threads=4).reshape(-1, 4)
    vbase_pred = apply_gate(v, vbs + vdelta, 'parse_and_repair', 4)
    assert (vy[np.arange(len(v)), vbase_pred].sum()) == 759
    hbase_pred = apply_gate([r for r, k in zip(b, held) if k], nbs[held] + hdelta, 'parse_and_repair', 4)
    held_base = int(y[held][np.arange(held.sum()), hbase_pred].sum())
    records = []
    for alpha in ALPHAS:
        for temp in TEMPS:
            soft = vbs + alpha * vdelta
            hscores = nbs[held] + alpha * hdelta
            if temp != 1.0:
                soft = np.exp((soft - soft.max(1, keepdims=True)) / temp)
                hscores = np.exp((hscores - hscores.max(1, keepdims=True)) / temp)
            c = apply_gate(v, soft, 'parse_and_repair', 4)
            hc = apply_gate([r for r, k in zip(b, held) if k], hscores, 'parse_and_repair', 4)
            ok = vy[np.arange(len(v)), c]
            r = dict(alpha=alpha, temp=temp, held=int(y[held][np.arange(held.sum()), hc].sum()),
                     val=int(ok.sum()))
            records.append(r)
            print('SCALE', r, flush=True)
    dump(OUT / 'scale_records.json', dict(base_held=held_base, records=records, test_opened=False))
    best = max(records, key=lambda r: (r['val'], r['held']))
    print('SCALE BEST', best, 'base_held', held_base, flush=True)
    dump(OUT / 'scale_selection.json', dict(best=best, base_held=held_base, test_opened=False,
                                            selected='round211' if best['val'] > 759 else 'round104'))
