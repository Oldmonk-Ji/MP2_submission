"""Round221: train on the minority-correct regime that TEST actually samples.

TRAIN is 44% hard rows of which most have 3-4 correct candidates; TEST keeps
only rows with >=2 distinct answers, so the useful training signal is the
minority-correct rows.  This sweeps sample weights and row filters keyed on
k = number of correct candidates, scoring VAL and HELD (overall and hard).
TEST is never touched.
"""
from round22_common import ROOT, load, dump
from round4_gates import apply_gate
from common import parse_answer
from threadpoolctl import threadpool_limits
import numpy as np, lightgbm as lgb, time

OUT = ROOT / 'experiments/round221'
BASE = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', num_leaves=7,
            min_data_in_leaf=100, learning_rate=.03, lambda_l2=50, seed=68068, verbosity=-1,
            num_threads=4, deterministic=True, force_col_wise=True)


def hard_mask(bank):
    return np.array([len({p for p in (parse_answer(c['text']) for c in r['candidates'])
                          if p is not None}) >= 2 for r in bank])


def run(z, b, v, y, vy, hh, hv, label, weight_of=None, keep=None):
    held = z['folds'] == 2
    fit = ~held
    held_bank = [r for r, k in zip(b, held) if k]
    x, nx, vx = z['x'], z['nx'], z['vx']
    bs, nbs, vbs = z['bs'], z['nbs'], z['vbs']
    w = weight_of(y) if weight_of else None
    if keep is not None:
        fit = fit & keep
        full = keep
    else:
        full = np.ones(len(b), bool)
    ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel(),
                     group=np.full(int(fit.sum()), 4), init_score=nbs[fit].ravel(),
                     weight=np.repeat(w[fit], 4) if w is not None else None)
    hm = lgb.train(BASE, ds, num_boost_round=200)
    sh = nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4)
    ds = lgb.Dataset(x[full].reshape(-1, x.shape[-1]), label=y[full].ravel(),
                     group=np.full(int(full.sum()), 4), init_score=bs[full].ravel(),
                     weight=np.repeat(w[full], 4) if w is not None else None)
    m = lgb.train(BASE, ds, num_boost_round=200)
    sv = vbs + m.predict(vx.reshape(-1, vx.shape[-1])).reshape(-1, 4)
    hc = apply_gate(held_bank, sh, 'parse_and_repair', 4)
    vc = apply_gate(v, sv, 'parse_and_repair', 4)
    rh, rv = np.arange(len(held_bank)), np.arange(len(v))
    r = dict(label=label, held=int((y[held][rh, hc] == 1).sum()),
             held_hard=int((y[held][hh, hc[hh]] == 1).sum()),
             val=int((vy[rv, vc] == 1).sum()),
             val_hard=int((vy[hv, vc[hv]] == 1).sum()))
    print('EXP', r, flush=True)
    return r


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    z = dict(np.load(ROOT / 'experiments/round216/cache.npz'))
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    y, vy = np.asarray(y), np.asarray(vy)
    held = z['folds'] == 2
    held_bank = [r for r, k in zip(b, held) if k]
    hh = hard_mask(held_bank)
    hv = hard_mask(v)
    k = y.sum(1).astype(int)
    recs = [run(z, b, v, y, vy, hh, hv, 'base')]
    for mult in (1.5, 2.0, 3.0, 5.0):
        f = lambda yy, m=mult: np.where(yy.sum(1) == 1, m, 1.0).astype(np.float32)
        recs.append(run(z, b, v, y, vy, hh, hv, f'k1x{mult}', weight_of=f))
    for mult in (2.0, 3.0):
        f = lambda yy, m=mult: np.where(yy.sum(1) <= 2, m, 1.0).astype(np.float32)
        recs.append(run(z, b, v, y, vy, hh, hv, f'kle2x{mult}', weight_of=f))
    recs.append(run(z, b, v, y, vy, hh, hv, 'only_kle2', keep=k <= 2))
    recs.append(run(z, b, v, y, vy, hh, hv, 'only_k1', keep=k == 1))
    dump(OUT / 'kweight.json', dict(records=recs, test_opened=False,
                                    seconds=time.perf_counter() - tick))
    print('DONE', time.perf_counter() - tick, flush=True)
