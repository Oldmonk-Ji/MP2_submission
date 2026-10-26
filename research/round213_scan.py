"""Two cheap post-hoc levers on the exact104 fusion scores (no retraining).

(a) gate margin scan: apply_gate limit in {0,0.25,0.5,1,2,3,4,8}.
(b) strict-gap rule: flip the fusion choice to argmax(strict) when the strict
    score gap (argmax_strict minus chosen) exceeds t. Designed on fit folds
    with clean oof strict and 3-fold oof stage-1 scores; held sanity; VAL
    decision vs the 759 bar.
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
STRICT_T = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
S1_PARAMS = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', num_leaves=7,
                 min_data_in_leaf=100, learning_rate=.03, lambda_l2=50, seed=68068, verbosity=-1,
                 num_threads=4, deterministic=True, force_col_wise=True)


def strict_rule(bank, yy, s1_scores, strict_scores, tau, gate_first=True, limit=4):
    base = apply_gate(bank, s1_scores, 'parse_and_repair', limit) if gate_first \
        else s1_scores.argmax(1)
    sel = base.copy()
    n = len(bank)
    r = np.arange(n)
    s_arg = strict_scores.argmax(1)
    gap = strict_scores[r, s_arg] - strict_scores[r, base]
    flip = (s_arg != base) & (gap > tau) & (base == s1_scores.argmax(1))
    sel[flip] = s_arg[flip]
    ok = yy[np.arange(n), sel]
    return dict(tau=tau, correct=int((ok == 1).sum()),
                fixes=int(((ok == 1) & (sel != base)).sum()),
                harms=int(((ok == 0) & (sel != base)).sum())), sel


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    dump(OUT / 'scan_plan.json', dict(
        method='(a) gate margin_limit scan, (b) strict-gap flip rule designed on fit folds via oof '
               'scores. No retraining of the fusion; TEST sealed.',
        limits=LIMITS, strict_taus=STRICT_T, source_sha256=sha(__file__), test_opened=False))
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
    with np.load(ROOT / 'experiments/round210/scores.npz') as z:
        strict_oof, strict_nested, strict_val = z['oof'], z['nested'], z['validation']
    fit = folds != 2
    held = ~fit
    print('FEATURES done', time.perf_counter() - tick, flush=True)
    oof_s1_path = OUT / 'oof_s1.npz'
    if oof_s1_path.exists():
        with np.load(oof_s1_path) as z:
            assert np.array_equal(z['ids'], ids)
            oof_s1 = z['scores']
    else:
        oof_s1 = np.zeros((len(b), 4))
        for f in range(3):
            tr = folds != f
            te = folds == f
            ds = lgb.Dataset(x[tr].reshape(-1, x.shape[-1]), label=y[tr].ravel(),
                             group=np.full(tr.sum(), 4), init_score=bs[tr].ravel())
            m = lgb.train(S1_PARAMS, ds, num_boost_round=200)
            oof_s1[te] = m.predict(x[te].reshape(-1, x.shape[-1])).reshape(-1, 4)
        oof_s1 += bs
        np.savez(oof_s1_path, scores=oof_s1, ids=ids)
    ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel(),
                     group=np.full(fit.sum(), 4), init_score=nbs[fit].ravel())
    hm = lgb.train(S1_PARAMS, ds, num_boost_round=200)
    held_s1 = nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4)
    m104 = lgb.Booster(model_file=str(ROOT / 'experiments/round210/bundle_control/meta.txt'))
    val_s1 = vbs + m104.predict(vx.reshape(-1, vx.shape[-1]), num_threads=4).reshape(-1, 4)
    held_bank = [r for r, k in zip(b, held) if k]
    # (a) margin scan
    margin_path = OUT / 'scan_margin_partial.json'
    if margin_path.exists():
        margin = read(margin_path)
        print('MARGIN loaded', margin, flush=True)
    else:
        margin = []
        for limit in LIMITS:
            hc = apply_gate(held_bank, held_s1, 'parse_and_repair', limit)
            c = apply_gate(v, val_s1, 'parse_and_repair', limit)
            margin.append(dict(limit=limit,
                               held=int((y[held][np.arange(held.sum()), hc] == 1).sum()),
                               val=int((vy[np.arange(len(v)), c] == 1).sum())))
            print('MARGIN', margin[-1], flush=True)
        dump(margin_path, margin)
    # (b) strict-gap rule design on fit folds (clean oof scores)
    design = []
    for tau in STRICT_T:
        rec, _ = strict_rule([b[i] for i in np.flatnonzero(fit)], y[fit],
                             oof_s1[fit], strict_oof[fit], tau)
        design.append(rec)
        print('DESIGN', rec, flush=True)
    best_tau = max(design, key=lambda r: (r['correct'], -r['harms']))['tau']
    hrec, _ = strict_rule(held_bank, y[held], held_s1, strict_nested[held], best_tau)
    print('HELD strict', hrec, flush=True)
    val_records = []
    for tau in STRICT_T:
        rec, _ = strict_rule(v, vy, val_s1, strict_val, tau)
        val_records.append(rec)
        print('VAL strict', rec, flush=True)
    dump(OUT / 'scan_records.json', dict(margin=margin, strict_design=design,
                                         best_tau=best_tau, strict_held=hrec,
                                         strict_val=val_records, test_opened=False))
    best_m = max(margin, key=lambda r: (r['val'], r['held']))
    best_s = max(val_records, key=lambda r: (r['correct'], -r['harms']))
    dump(OUT / 'scan_selection.json', dict(best_margin=best_m, best_strict=best_s,
                                           test_opened=False,
                                           selected='round213' if max(best_m['val'], best_s['correct']) > 759
                                           else 'round104'))
    print('SCAN BEST margin', best_m, 'strict', best_s, flush=True)
