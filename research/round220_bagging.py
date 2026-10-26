"""Round220: real bagging (row + feature subsampling) instead of inert seed averaging.

Round216's seed ensemble produced identical predictions because feature_fraction
was 1 and bagging was off, so seeds changed nothing.  Here each member sees a
different row subset and feature subset, which is genuine variance reduction.
Reported on both VAL and HELD, and on the hard rows that decide the test score.
TEST is never touched.
"""
from round22_common import ROOT, load, dump
from round4_gates import apply_gate
from common import parse_answer
from threadpoolctl import threadpool_limits
import numpy as np, lightgbm as lgb, time

OUT = ROOT / 'experiments/round220'
BASE = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', num_leaves=7,
            min_data_in_leaf=100, learning_rate=.03, lambda_l2=50, verbosity=-1,
            num_threads=4, deterministic=True, force_col_wise=True,
            bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8)


def hard_mask(bank):
    return np.array([len({p for p in (parse_answer(c['text']) for c in r['candidates'])
                          if p is not None}) >= 2 for r in bank])


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    z = np.load(ROOT / 'experiments/round216/cache.npz')
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    y, vy = np.asarray(y), np.asarray(vy)
    held = z['folds'] == 2
    fit = ~held
    held_bank = [r for r, k in zip(b, held) if k]
    yh = y[held]
    hh, hv = hard_mask(held_bank), hard_mask(v)
    x, nx, vx = z['x'], z['nx'], z['vx']
    bs, nbs, vbs = z['bs'], z['nbs'], z['vbs']
    SEEDS = (68068, 1009, 2027, 3041, 4093, 5077, 6091, 7103)
    ph, pv = [], []
    recs = []
    for i, seed in enumerate(SEEDS):
        p = dict(BASE, seed=seed)
        ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel(),
                         group=np.full(int(fit.sum()), 4), init_score=nbs[fit].ravel())
        hm = lgb.train(p, ds, num_boost_round=200)
        ph.append(nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4))
        ds = lgb.Dataset(x.reshape(-1, x.shape[-1]), label=y.ravel(),
                         group=np.full(len(b), 4), init_score=bs.ravel())
        m = lgb.train(p, ds, num_boost_round=200)
        pv.append(vbs + m.predict(vx.reshape(-1, vx.shape[-1])).reshape(-1, 4))
        sh, sv = np.mean(ph, 0), np.mean(pv, 0)
        hc = apply_gate(held_bank, sh, 'parse_and_repair', 4)
        vc = apply_gate(v, sv, 'parse_and_repair', 4)
        rh, rv = np.arange(len(held_bank)), np.arange(len(v))
        r = dict(n=i + 1, held=int((yh[rh, hc] == 1).sum()),
                 held_hard=int((yh[hh, hc[hh]] == 1).sum()),
                 val=int((vy[rv, vc] == 1).sum()),
                 val_hard=int((vy[hv, vc[hv]] == 1).sum()))
        recs.append(r)
        print('BAG', r, flush=True)
    print('base was held 1477 / held_hard 592 / val 759 / val_hard 316', flush=True)
    dump(OUT / 'bagging.json', dict(records=recs, params=BASE, test_opened=False,
                                    seconds=time.perf_counter() - tick))
    np.savez(OUT / 'bag_scores.npz', ph=np.mean(ph, 0), pv=np.mean(pv, 0))
    print('DONE', time.perf_counter() - tick, flush=True)
