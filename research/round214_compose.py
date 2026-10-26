"""Refine the strict-gap rule: fine tau sweep, top-2 restriction, logistic flipper.

round213_scan found strict-gap tau=3.0 gives VAL 760 (+1) but HELD 1476 (-1).
Before any freeze, check whether a configuration is positive on BOTH held and
val: fine tau sweep, a top-2-restricted variant, and a small logistic flip
model trained on fit folds (2-fold swap) over minimal features.
"""
from round22_common import ROOT, load, read, dump, sha
from round83_selector import extend
from round80_selector import augment
from round79_selector import design, project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegression
import numpy as np, lightgbm as lgb, time

OUT = ROOT / 'experiments/round214'
TAUS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
S1_PARAMS = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', num_leaves=7,
                 min_data_in_leaf=100, learning_rate=.03, lambda_l2=50, seed=68068, verbosity=-1,
                 num_threads=4, deterministic=True, force_col_wise=True)


def flip_frame(s1_scores, strict_scores, verified, rows):
    r = np.arange(len(rows))
    s = s1_scores[rows]
    st = strict_scores[rows]
    vf = verified[rows] if verified is not None else np.zeros_like(st, bool)
    first = s.argmax(1)
    second = s.argsort(1)[:, -2]
    base = first
    feats = np.stack([
        s[r, first] - s[r, second],
        st[r, second] - st[r, first],
        st[r, first],
        st[r, second],
        st.max(1),
        st.max(1) - st[r, first],
        vf[r, first].astype(float),
        vf[r, second].astype(float),
    ], 1)
    return feats, first, second


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    dump(OUT / 'plan.json', dict(
        method='Refine strict-gap rule: fine tau sweep, top-2 restricted variant, logistic flip model '
               'on minimal features (2-fold swap design on fit folds). Held and VAL both reported.',
        taus=TAUS, source_sha256=sha(__file__), test_opened=False))
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
    with np.load(ROOT / 'experiments/round211/verified_masks.npz') as z:
        mv_fit, mv_held, mv_val = z['fit'], z['held'], z['val']
    fit = folds != 2
    held = ~fit
    print('FEATURES done', time.perf_counter() - tick, flush=True)
    with np.load(ROOT / 'experiments/round213/oof_s1.npz') as z:
        oof_s1 = z['scores']
    ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel(),
                     group=np.full(fit.sum(), 4), init_score=nbs[fit].ravel())
    hm = lgb.train(S1_PARAMS, ds, num_boost_round=200)
    held_s1 = nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4)
    m104 = lgb.Booster(model_file=str(ROOT / 'experiments/round210/bundle_control/meta.txt'))
    val_s1 = vbs + m104.predict(vx.reshape(-1, vx.shape[-1]), num_threads=4).reshape(-1, 4)
    held_bank = [r for r, k in zip(b, held) if k]
    vgated = apply_gate(v, val_s1, 'parse_and_repair', 4)
    hgated = apply_gate(held_bank, held_s1, 'parse_and_repair', 4)
    assert (vy[np.arange(len(v)), vgated] == 1).sum() == 759
    # fine tau sweep with optional top-2 restriction (gate-first semantics)
    records = []
    for restrict in (False, True):
        for tau in TAUS:
            for (bank, s1, st, yy, tag) in (
                    (held_bank, held_s1, strict_nested[held], y[held], 'held'),
                    (v, val_s1, strict_val, vy, 'val')):
                r = np.arange(len(bank))
                base = hgated if tag == 'held' else vgated
                s_arg = st.argmax(1)
                gap = st[r, s_arg] - st[r, base]
                flip = (s_arg != base) & (gap > tau) & (base == s1.argmax(1))
                if restrict:
                    top2 = np.argsort(-s1, axis=1)[:, :2]
                    flip &= np.asarray([s_arg[i] in top2[i] for i in range(len(bank))])
                sel = base.copy()
                sel[flip] = s_arg[flip]
                ok = yy[np.arange(len(bank)), sel]
                records.append(dict(tag=tag, restrict=restrict, tau=tau,
                                    base=int((yy[np.arange(len(bank)), base] == 1).sum()),
                                    correct=int((ok == 1).sum()),
                                    flips=int(flip.sum())))
    for r in records:
        print('SWEEP', r, flush=True)
    dump(OUT / 'tau_sweep.json', records)
    # logistic flip model: design 2-fold swap on fit folds
    fit_idx = np.flatnonzero(fit)
    ffeats, ffirst, fsecond = flip_frame(oof_s1, strict_oof, None, fit_idx)
    yfirst = y[fit_idx][np.arange(len(fit_idx)), ffirst] == 1
    ysecond = y[fit_idx][np.arange(len(fit_idx)), fsecond] == 1
    fvalid = yfirst != ysecond
    ftarget = ysecond.astype(int)
    design_records = []
    for tr_f, te_f in ((0, 1), (1, 0)):
        trm = np.flatnonzero(folds[fit_idx] == tr_f)
        tem = np.flatnonzero(folds[fit_idx] == te_f)
        use = trm[fvalid[trm]]
        clf = LogisticRegression(C=1.0, solver='liblinear', max_iter=200)
        clf.fit(ffeats[use], ftarget[use])
        p = clf.predict_proba(ffeats[tem])[:, 1]
        base_ok = (y[fit_idx[tem]][np.arange(len(tem)), ffirst[tem]] == 1)
        sel = ffirst[tem].copy()
        flip = p > 0.5
        sel[flip] = fsecond[tem][flip]
        ok = y[fit_idx[tem]][np.arange(len(tem)), sel]
        design_records.append(dict(train_fold=tr_f, eval_fold=te_f, n_pairs=int(use.sum()),
                                   base=int(base_ok.sum()), correct=int((ok == 1).sum()),
                                   fixes=int(((ok == 1) & (sel != ffirst[tem])).sum()),
                                   harms=int(((ok == 0) & (sel != ffirst[tem])).sum())))
        print('LOGISTIC DESIGN', design_records[-1], flush=True)
    # held + val with logistic trained on all fit folds
    use = np.flatnonzero(fvalid)
    clf = LogisticRegression(C=1.0, solver='liblinear', max_iter=200)
    clf.fit(ffeats[use], ftarget[use])
    for (bank, s1, st, yy, tag) in ((held_bank, held_s1, strict_nested[held], y[held], 'held'),
                                    (v, val_s1, strict_val, vy, 'val')):
        rows = np.arange(len(bank))
        feats, first, second = flip_frame(s1, st, None, rows)
        p = clf.predict_proba(feats)[:, 1]
        base = (hgated if tag == 'held' else vgated)
        sel = base.copy()
        flip = (base == first) & (p > 0.5)
        sel[flip] = second[flip]
        ok = yy[np.arange(len(rows)), sel]
        print('LOGISTIC', tag, dict(base=int((yy[np.arange(len(rows)), base] == 1).sum()),
                                    correct=int((ok == 1).sum()),
                                    fixes=int(((ok == 1) & (sel != base)).sum()),
                                    harms=int(((ok == 0) & (sel != base)).sum())), flush=True)
    dump(OUT / 'logistic_records.json', design_records)
