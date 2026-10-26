"""Ensemble the per-C strict scores / rules from round215.

(a) score-ensemble: z-score each C's strict matrix then average; sweep tau with
    the frozen rule semantics.
(b) rule-ensemble: per-C tau chosen on fit folds (max design correct), then
    vote over flip decisions with threshold k in {any, all} designed on fit.
Held sanity, VAL decision. TEST sealed.
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
TAUS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
CS = [0.1, 0.3, 1.0, 3.0, 10.0]
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
                flips=int(flip.sum())), flip, s_arg


def zscore(m):
    return (m - m.mean(0)) / (m.std(0) + 1e-9)


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
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
    with np.load(ROOT / 'experiments/round213/oof_s1.npz') as z:
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
    print('FEATURES+S1 done', time.perf_counter() - tick, flush=True)
    splits = {}
    for C in CS:
        tag = f'c{str(C).replace(".", "")}'
        with np.load(OUT / f'scores_{tag}.npz') as z:
            splits[C] = (z['oof'], z['nested'], z['validation'])
    # (a) score-ensemble
    for agg in ('mean', 'median'):
        so = zscore(np.stack([splits[C][0] for C in CS])).mean(0) if agg == 'mean' \
            else np.median(np.stack([splits[C][0] for C in CS]), 0)
        sn = zscore(np.stack([splits[C][1] for C in CS])).mean(0) if agg == 'mean' \
            else np.median(np.stack([splits[C][1] for C in CS]), 0)
        sv = zscore(np.stack([splits[C][2] for C in CS])).mean(0) if agg == 'mean' \
            else np.median(np.stack([splits[C][2] for C in CS]), 0)
        recs = []
        for tau in TAUS:
            d, _, _ = rule(fit_bank, oof_s1[fit], so[fit], y[fit], tau, fgated)
            h, _, _ = rule(held_bank, held_s1, sn[held], y[held], tau, hgated)
            vr, _, _ = rule(v, val_s1, sv, vy, tau, vgated)
            recs.append(dict(tau=tau, design=d, held=h, val=vr))
            print('SCORE-ENS', agg, recs[-1], flush=True)
        dump(OUT / f'score_ens_{agg}.json', recs)
    # (b) rule-ensemble: per-C tau on design folds, then vote
    per_c = {}
    for C in CS:
        so, sn, sv = splits[C]
        best = None
        for tau in TAUS:
            rec, _, _ = rule(fit_bank, oof_s1[fit], so[fit], y[fit], tau, fgated)
            if best is None or rec['correct'] > best[0]['correct'] or \
                    (rec['correct'] == best[0]['correct'] and rec['harms'] < best[0]['harms']):
                best = (rec, tau)
        per_c[C] = best[1]
    print('PER-C TAUS', per_c, flush=True)
    flips = {}
    for C in CS:
        so, sn, sv = splits[C]
        tau = per_c[C]
        for (tag, s1, st, gated, bank) in (('design', oof_s1[fit], so[fit], fgated, fit_bank),
                                           ('held', held_s1, sn[held], hgated, held_bank),
                                           ('val', val_s1, sv, vgated, v)):
            yy = y[fit] if tag == 'design' else y[held] if tag == 'held' else vy
            _, flip, s_arg = rule(bank, s1, st, yy, tau, gated)
            flips.setdefault(tag, {})[C] = (flip, s_arg)
    for tag in ('design', 'held', 'val'):
        bank = fit_bank if tag == 'design' else held_bank if tag == 'held' else v
        gated = fgated if tag == 'design' else hgated if tag == 'held' else vgated
        yy = y[fit] if tag == 'design' else y[held] if tag == 'held' else vy
        base = gated.copy()
        n = len(bank)
        votes = sum(flips[tag][C][0].astype(int) for C in CS)
        for k in (1, 2, 3, 4, 5):
            agree = np.flatnonzero(votes >= k)
            sel = base.copy()
            for i in agree:
                cands = [flips[tag][C][1][i] for C in CS if flips[tag][C][0][i]]
                sel[i] = max(set(cands), key=(lambda c: (cands.count(c), -abs(c - flips[tag][1.0][1][i]))))
            ok = yy[np.arange(n), sel]
            print('RULE-ENS', tag, 'k>=', k,
                  dict(correct=int((ok == 1).sum()),
                       fixes=int(((ok == 1) & (sel != base)).sum()),
                       harms=int(((ok == 0) & (sel != base)).sum()),
                       flips=int(len(agree))), flush=True)
    print('DONE', time.perf_counter() - tick, flush=True)
