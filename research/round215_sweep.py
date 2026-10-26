"""Sweep the strict-gap rule per retrained C (round215 fits).

For each C in round215, sweep tau with the frozen rule semantics (top-2
restricted, gate-first, flip to strict argmax when gap > tau). Design on fit
folds with clean oof scores, held sanity, VAL decision. C=1 must reproduce the
frozen round214 result (VAL 760 / HELD 1477 at tau=3.0) as a pipeline check.
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

OUT = ROOT / 'experiments/round215'
TAUS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
S1_PARAMS = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', num_leaves=7,
                 min_data_in_leaf=100, learning_rate=.03, lambda_l2=50, seed=68068, verbosity=-1,
                 num_threads=4, deterministic=True, force_col_wise=True)


def rule(bank, s1, st, yy, tau, gated):
    r = np.arange(len(bank))
    base = gated.copy()
    s_arg = st.argmax(1)
    top2 = np.argsort(-s1, axis=1)[:, :2]
    in_top2 = np.asarray([s_arg[i] in top2[i] for i in r])
    gap = st[r, s_arg] - st[r, base]
    flip = (s_arg != base) & (gap > tau) & in_top2 & (base == s1.argmax(1))
    sel = base.copy()
    sel[flip] = s_arg[flip]
    ok = yy[np.arange(len(bank)), sel]
    return dict(correct=int((ok == 1).sum()),
                fixes=int(((ok == 1) & (sel != base)).sum()),
                harms=int(((ok == 0) & (sel != base)).sum()),
                flips=int(flip.sum()))


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    assert (OUT / 'fit_complete.json').exists(), 'run round215_strictgrid.py first'
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
    fit = folds != 2
    held = ~fit
    print('FEATURES done', time.perf_counter() - tick, flush=True)
    with np.load(ROOT / 'experiments/round213/oof_s1.npz') as z:
        assert np.array_equal(z['ids'], ids)
        oof_s1 = z['scores']
    ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel(),
                     group=np.full(fit.sum(), 4), init_score=nbs[fit].ravel())
    hm = lgb.train(S1_PARAMS, ds, num_boost_round=200)
    held_s1 = nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4)
    m104 = lgb.Booster(model_file=str(ROOT / 'experiments/round210/bundle_control/meta.txt'))
    val_s1 = vbs + m104.predict(vx.reshape(-1, vx.shape[-1]), num_threads=4).reshape(-1, 4)
    held_bank = [r for r, k in zip(b, held) if k]
    fit_bank = [b[i] for i in np.flatnonzero(fit)]
    vgated = apply_gate(v, val_s1, 'parse_and_repair', 4)
    hgated = apply_gate(held_bank, held_s1, 'parse_and_repair', 4)
    fgated = apply_gate(fit_bank, oof_s1[fit], 'parse_and_repair', 4)
    assert (vy[np.arange(len(v)), vgated] == 1).sum() == 759
    print('S1 done', time.perf_counter() - tick, flush=True)
    records = []
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        tag = f'c{str(C).replace(".", "")}'
        with np.load(OUT / f'scores_{tag}.npz') as z:
            st_oof, st_nested, st_val = z['oof'], z['nested'], z['validation']
        if C == 1.0:
            rec = rule(v, val_s1, st_val, vy, 3.0, vgated)
            assert rec['correct'] == 760, rec
            hrec = rule(held_bank, held_s1, st_nested[held], y[held], 3.0, hgated)
            assert hrec['correct'] == 1477, hrec
            print('C=1 frozen reproduction OK', flush=True)
        design = [rule(fit_bank, oof_s1[fit], st_oof[fit], y[fit], tau, fgated) for tau in TAUS]
        held_recs = [rule(held_bank, held_s1, st_nested[held], y[held], tau, hgated) for tau in TAUS]
        val_recs = [rule(v, val_s1, st_val, vy, tau, vgated) for tau in TAUS]
        for tau, d, h, vr in zip(TAUS, design, held_recs, val_recs):
            records.append(dict(C=C, tau=tau, design=d, held=h, val=vr))
            print('SWEEP', records[-1], flush=True)
    dump(OUT / 'sweep_records.json', records)
    by_c = {}
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        rows = [r for r in records if r['C'] == C]
        by_c[C] = max(rows, key=lambda r: (r['val']['correct'], r['held']['correct'], -r['val']['harms']))
    best = max(by_c.values(), key=lambda r: (r['val']['correct'], r['held']['correct'], -r['val']['harms']))
    dump(OUT / 'sweep_selection.json', dict(
        by_C={str(k): v for k, v in by_c.items()}, best=best,
        selected='round215' if best['val']['correct'] > 759 and best['held']['correct'] >= 1477
        else 'round104', test_opened=False))
    print('SWEEP BEST', best, flush=True)
