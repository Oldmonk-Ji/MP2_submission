"""Hard-rule analysis: force the unique verified_trace candidate.

A candidate is 'verified' when its own final answer value is derivable from an
accepted equation chain inside its text (round209 strict trace). Smooth feature
injection failed (round210 events arm); a hard rule may jump the local optimum.

Protocol: design statistics on fit folds and held (held used for reporting and
sanity only; no parameter is tuned on held), decision on complete VAL.
"""
from round22_common import ROOT, load, read, dump, sha
from round83_selector import extend
from round80_selector import augment
from round79_selector import design, project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from round209_trace import extract, FLAGS
from threadpoolctl import threadpool_limits
import numpy as np, lightgbm as lgb, time

OUT = ROOT / 'experiments/round211'


def verified_mask(bank, flags=None):
    return np.asarray([[extract(row['question'], c['text'], flags)[1].get('reason') == 'verified_trace'
                        for c in row['candidates']] for row in bank], bool)


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    dump(OUT / 'rule_plan.json', dict(
        method='Force the unique verified_trace candidate (round209 strict trace). Design on fit folds, '
               'held reported for sanity only, decision on complete VAL against the 759 bar.',
        flags=FLAGS, source_sha256=sha(__file__), test_opened=False))
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    with np.load(ROOT / 'experiments/round77/nested/components.npz') as z:
        folds = z['folds']
    fit = folds != 2
    held = ~fit
    t0 = time.perf_counter()
    masks_path = OUT / 'verified_masks.npz'
    if masks_path.exists():
        with np.load(masks_path) as z:
            mv, mfit, mheld = z['val'], z['fit'], z['held']
    else:
        mv = verified_mask(v)
        mfit = verified_mask([r for r, k in zip(b, fit) if k])
        mheld = verified_mask([r for r, k in zip(b, held) if k])
        np.savez(masks_path, val=mv, fit=mfit, held=mheld)
    print('TRACE seconds', time.perf_counter() - t0, flush=True)

    def stats(mask, yy, chosen=None):
        n = len(mask)
        vi = mask.argmax(1)
        uniq = mask.sum(1) == 1
        anyv = mask.sum(1) >= 1
        uniq_correct = uniq & yy[np.arange(n), vi].astype(bool)
        uni = int(uniq.sum())
        if chosen is not None:
            force_ok = uniq & yy[np.arange(n), vi].astype(bool)
            force_bad = uniq & ~yy[np.arange(n), vi].astype(bool)
            fixes = int((force_ok & (chosen != vi)).sum())
            harms = int((force_bad & (chosen != vi)).sum())
        else:
            fixes = harms = None
        return dict(n=n, unique_verified=int(uni),
                    unique_verified_correct=int(uniq_correct.sum()),
                    any_verified=int(anyv.sum()), fixes=fixes, harms=harms)

    fit_stats = stats(mfit, y[fit], None)
    print('FIT', fit_stats, flush=True)
    # fusion held model (nested features) + val model (exact104)
    weights = [m['weight'] for m in read(ROOT / 'experiments/round25/selected_bundle/config.json')['members']]
    cache = {}
    for split in ('train', 'validation'):
        with np.load(ROOT / f'experiments/round78/{split}_features.npz') as z:
            cache[split] = (z['root'], z['extra'])
    with np.load(ROOT / 'experiments/round68/oof.npz') as z:
        ss = z['components']
    with np.load(ROOT / 'experiments/round77/nested/components.npz') as z:
        nested = z['components']
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
    ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel(),
                     group=np.full(fit.sum(), 4), init_score=nbs[fit].ravel())
    hm = lgb.train(params, ds, num_boost_round=200)
    hbase_scores = nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4)
    hchosen = apply_gate([r for r, k in zip(b, held) if k], hbase_scores, 'parse_and_repair', 4)
    held_base = int(y[held][np.arange(held.sum()), hchosen].sum())
    held_stats = stats(mheld, y[held], hchosen)
    print('HELD', held_stats, 'base', held_base, flush=True)
    m = lgb.Booster(model_file=str(ROOT / 'experiments/round210/bundle_control/meta.txt'))
    vbase_scores = vbs + m.predict(vx.reshape(-1, vx.shape[-1]), num_threads=4).reshape(-1, 4)
    vchosen = apply_gate(v, vbase_scores, 'parse_and_repair', 4)
    assert (vy[np.arange(len(v)), vchosen].sum()) == 759
    records = []
    ranks = 4 - 1 - vbase_scores.argsort(1).argsort(1)
    correct_mask = vy == 1
    diag = dict(errors_top2=int(((ranks <= 1) & correct_mask).any(1).sum() - 759),
                all_top2=int(((ranks <= 1) & correct_mask).any(1).sum()),
                errors_top1=39)
    print('DIAG', diag, flush=True)
    for name, cond in [('A_force', lambda rank, n: np.ones(len(rank), bool)),
                       ('B_top2', lambda rank, n: rank <= 1),
                       ('C_top1', lambda rank, n: rank == 0)]:
        mv_rank = ranks
        uniq = mv.sum(1) == 1
        vi = mv.argmax(1)
        force = uniq & cond(mv_rank[np.arange(len(mv)), vi], 4)
        sel = vchosen.copy()
        sel[force] = vi[force]
        ok = vy[np.arange(len(v)), sel]
        r = dict(variant=name, forced=int(force.sum()), correct=int((ok == 1).sum()),
                 fixes=int(((ok == 1) & (sel != vchosen)).sum()),
                 harms=int(((ok == 0) & (sel != vchosen)).sum()))
        records.append(r)
        print('VAL', r, flush=True)
    dump(OUT / 'rule_records.json', dict(fit=fit_stats, held=held_stats, held_base=held_base,
                                         val=records, diag=diag, test_opened=False))
    best = max(records, key=lambda r: (r['correct'], -r['harms']))
    dump(OUT / 'rule_selection.json', dict(best=best, records=records, test_opened=False,
                                           selected='round211' if best['correct'] > 759 else 'round104'))
    print('RULE BEST', best, flush=True)
