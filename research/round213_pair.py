"""Two-stage fusion: stage-2 pairwise disambiguator over the stage-1 top-2.

The top-2 diagnostic showed 24 of the 39 VAL errors already have the correct
candidate ranked second by the exact104 fusion. Stage 2 learns, from the two
candidates' feature vectors, whether flipping to second is right, and flips
only above a confidence threshold tau.

Composition (production order): gate first (exact104 behavior, computed once
per split), then stage-2 may flip a row only when the gate did NOT override the
raw argmax (gate rows stay untouched, preserving its +7 net on VAL).

Protocol: stage-2 design on fit folds (2-fold swap, stage-1 top-2 from clean
3-fold oof fusions), held sanity with nested stage-1, decision on complete VAL.
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
TAUS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
S1_PARAMS = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', num_leaves=7,
                 min_data_in_leaf=100, learning_rate=.03, lambda_l2=50, seed=68068, verbosity=-1,
                 num_threads=4, deterministic=True, force_col_wise=True)
S2_PARAMS = dict(objective='binary', metric='None', num_leaves=31, min_data_in_leaf=20,
                 learning_rate=.03, seed=68068, verbosity=-1, num_threads=4, deterministic=True,
                 force_col_wise=True, feature_fraction=0.8)


def pair_features(feat, scores, rows):
    order = np.argsort(-scores[rows], axis=1)
    first = order[:, 0]
    second = order[:, 1]
    r = np.arange(len(rows))
    f = feat[rows][r, first]
    s = feat[rows][r, second]
    gap = scores[rows][r, first] - scores[rows][r, second]
    return np.concatenate([f, s, f - s, gap[:, None]], 1), first, second


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    dump(OUT / 'plan.json', dict(
        method='Stage-2 pairwise GBM over exact104 stage-1 top-2; gate first, flip above tau only on '
               'non-gated rows. Design on fit-fold swap, held sanity, VAL decision.',
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
    fit = folds != 2
    held = ~fit
    print('FEATURES done', time.perf_counter() - tick, flush=True)
    # clean stage-1 top-2 for every train row via 3-fold oof fusion
    oof_s1 = np.zeros((len(b), 4))
    for f in range(3):
        tr = folds != f
        te = folds == f
        ds = lgb.Dataset(x[tr].reshape(-1, x.shape[-1]), label=y[tr].ravel(),
                         group=np.full(tr.sum(), 4), init_score=bs[tr].ravel())
        m = lgb.train(S1_PARAMS, ds, num_boost_round=200)
        oof_s1[te] = m.predict(x[te].reshape(-1, x.shape[-1])).reshape(-1, 4)
        print('S1 fold', f, flush=True)
    oof_s1 += bs
    ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel(),
                     group=np.full(fit.sum(), 4), init_score=nbs[fit].ravel())
    hm = lgb.train(S1_PARAMS, ds, num_boost_round=200)
    held_s1 = nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4)
    m104 = lgb.Booster(model_file=str(ROOT / 'experiments/round210/bundle_control/meta.txt'))
    val_s1 = vbs + m104.predict(vx.reshape(-1, vx.shape[-1]), num_threads=4).reshape(-1, 4)

    def s2_train(rows):
        feats, first, second = pair_features(x, oof_s1, rows)
        yfirst = y[rows][np.arange(len(rows)), first] == 1
        ysecond = y[rows][np.arange(len(rows)), second] == 1
        keep = yfirst != ysecond
        ds = lgb.Dataset(feats[keep], label=ysecond[keep].astype(float))
        n0 = int((~ysecond[keep]).sum())
        n1 = int(ysecond[keep].sum())
        return lgb.train(dict(S2_PARAMS, scale_pos_weight=n0 / max(n1, 1)), ds, num_boost_round=200), \
            dict(pairs=int(keep.sum()), second_correct=n1)

    # design: 2-fold swap on fit folds
    design = {}
    for f in (0, 1):
        m, info = s2_train(np.flatnonzero(folds == f))
        other = np.flatnonzero(folds == 1 - f)
        feats_eval, first2, second2 = pair_features(x[other], oof_s1[other], np.arange(len(other)))
        p_eval = m.predict(feats_eval, num_threads=4)
        base_gated = apply_gate([b[i] for i in other], oof_s1[other], 'parse_and_repair', 4)
        base_ok = y[other][np.arange(len(other)), base_gated]
        records = []
        for tau in TAUS:
            sel = base_gated.copy()
            flip = (base_gated == first2) & (p_eval > tau)
            sel[flip] = second2[flip]
            ok = y[other][np.arange(len(other)), sel]
            records.append(dict(tau=tau, correct=int((ok == 1).sum()),
                                fixes=int(((ok == 1) & (sel != base_gated)).sum()),
                                harms=int(((ok == 0) & (sel != base_gated)).sum())))
        design[f] = dict(info=info, base=int((base_ok == 1).sum()), records=records)
        print('DESIGN fold', f, design[f], flush=True)
    nets = {}
    for tau in TAUS:
        nets[tau] = sum(r['fixes'] - r['harms'] for f in (0, 1)
                        for r in design[f]['records'] if r['tau'] == tau)
    best_tau = max(nets, key=lambda t: (nets[t], t))
    print('DESIGN nets', nets, 'best_tau', best_tau, flush=True)
    # held sanity at best_tau
    hm2, hinfo = s2_train(np.flatnonzero(fit))
    held_rows = np.flatnonzero(held)
    hfeats, hfirst, hsecond = pair_features(x[held_rows], held_s1, np.arange(len(held_rows)))
    hp = hm2.predict(hfeats, num_threads=4)
    hbase_gated = apply_gate([b[i] for i in held_rows], held_s1, 'parse_and_repair', 4)
    hbase_ok = y[held][np.arange(len(held_rows)), hbase_gated]
    hsel = hbase_gated.copy()
    hflip = (hbase_gated == hfirst) & (hp > best_tau)
    hsel[hflip] = hsecond[hflip]
    hok = y[held][np.arange(len(held_rows)), hsel]
    held_record = dict(base=int((hbase_ok == 1).sum()), tau=best_tau,
                       correct=int((hok == 1).sum()),
                       fixes=int(((hok == 1) & (hsel != hbase_gated)).sum()),
                       harms=int(((hok == 0) & (hsel != hbase_gated)).sum()))
    print('HELD', hinfo, held_record, flush=True)
    # val: stage-2 trained on all train rows with oof stage-1
    fm, finfo = s2_train(np.arange(len(b)))
    vfeats, vfirst, vsecond = pair_features(vx, val_s1, np.arange(len(v)))
    vp = fm.predict(vfeats, num_threads=4)
    vbase_gated = apply_gate(v, val_s1, 'parse_and_repair', 4)
    assert (vy[np.arange(len(v)), vbase_gated] == 1).sum() == 759
    records = []
    for tau in TAUS:
        vsel = vbase_gated.copy()
        vflip = (vbase_gated == vfirst) & (vp > tau)
        vsel[vflip] = vsecond[vflip]
        vok = vy[np.arange(len(v)), vsel]
        records.append(dict(tau=tau, correct=int((vok == 1).sum()),
                            fixes=int(((vok == 1) & (vsel != vbase_gated)).sum()),
                            harms=int(((vok == 0) & (vsel != vbase_gated)).sum())))
    print('VAL', records, flush=True)
    best_val = max(records, key=lambda r: (r['correct'], -r['harms']))
    dump(OUT / 'records.json', dict(design={str(f): design[f] for f in (0, 1)},
                                    best_tau=best_tau, held=hinfo | held_record,
                                    val=records, val_best=best_val, test_opened=False))
    dump(OUT / 'selection.json', dict(best=best_val, best_tau=best_tau, test_opened=False,
                                      selected='round213' if best_val['correct'] > 759 else 'round104'))
    print('PAIR BEST', best_val, flush=True)
