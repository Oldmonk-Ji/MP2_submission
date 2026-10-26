"""Round216: fast model/feature experiments on the cached fusion surface.

Held rows (fold 2) use nested features and fit-fold training. VAL uses the
exact104 design and all-4743 training, matching the production recipe. Every
record reports val, held, val-hard (>=2 distinct parsed answers) and flip
counts. TEST is never touched.
"""
from round22_common import ROOT, load, read, dump
from round4_gates import apply_gate
from threadpoolctl import threadpool_limits
import numpy as np, lightgbm as lgb, time, json, sys

OUT = ROOT / 'experiments/round216'
BASE = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', num_leaves=7,
            min_data_in_leaf=100, learning_rate=.03, lambda_l2=50, seed=68068, verbosity=-1,
            num_threads=4, deterministic=True, force_col_wise=True)


def load_cache():
    z = np.load(OUT / 'cache.npz')
    return {k: z[k] for k in z.files}


def gate_accuracy(bank, scores, yy, limit=4):
    c = apply_gate(bank, scores, 'parse_and_repair', limit)
    return int((yy[np.arange(len(bank)), c] == 1).sum()), c


def train_eval(cfg, cache, b, v, y, vy, extra=None, rounds=200, seeds=(68068,), tag=''):
    fit = cache['folds'] != 2
    held = ~fit
    x, nx, vx = cache['x'], cache['nx'], cache['vx']
    bs, nbs, vbs = cache['bs'], cache['nbs'], cache['vbs']
    if extra is not None:
        x = np.concatenate([x, extra[0]], 2)
        nx = np.concatenate([nx, extra[1]], 2)
        vx = np.concatenate([vx, extra[2]], 2)
    held_bank = [r for r, k in zip(b, held) if k]
    preds_h, preds_v = [], []
    for seed in seeds:
        p = dict(BASE, **{k: v2 for k, v2 in cfg.items() if k != 'rounds'}, seed=seed)
        ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel(),
                         group=np.full(fit.sum(), 4), init_score=nbs[fit].ravel())
        hm = lgb.train(p, ds, num_boost_round=cfg.get('rounds', rounds))
        preds_h.append(nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4))
        ds = lgb.Dataset(x.reshape(-1, x.shape[-1]), label=y.ravel(), group=np.full(len(b), 4),
                         init_score=bs.ravel())
        m = lgb.train(p, ds, num_boost_round=cfg.get('rounds', rounds))
        preds_v.append(vbs + m.predict(vx.reshape(-1, vx.shape[-1])).reshape(-1, 4))
    held_s1 = np.mean(preds_h, 0)
    val_s1 = np.mean(preds_v, 0)
    hv, hc = gate_accuracy(held_bank, held_s1, y[held])
    vv, vc = gate_accuracy(v, val_s1, vy)
    hard = cache['vhard']
    rec = dict(tag=tag, seeds=list(seeds), val=vv, held=hv,
               val_hard=int((vy[hard, vc[hard]] == 1).sum()),
               val_base=759, held_base=1477)
    return rec, val_s1, held_s1, vc


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    cache = load_cache()
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    from common import parse_answer
    parsed = [[parse_answer(c['text']) for c in row['candidates']] for row in v]
    cache['vhard'] = np.array([len({p for p in ps if p is not None}) >= 2 for ps in parsed])
    records = []
    grid = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
        dict(),
        dict(num_leaves=15, min_data_in_leaf=50, rounds=400),
        dict(num_leaves=15, min_data_in_leaf=50, rounds=600),
        dict(num_leaves=31, min_data_in_leaf=50, rounds=400),
        dict(num_leaves=31, min_data_in_leaf=30, rounds=600),
    ]
    for cfg in grid:
        rec, _, _, _ = train_eval(cfg, cache, b, v, y, vy, tag=f'grid{len(records)}')
        records.append(rec)
        print('GRID', rec, flush=True)
    dump(OUT / 'exp_capacity.json', dict(records=records, test_opened=False,
                                         seconds=time.perf_counter() - tick))
    print('DONE', time.perf_counter() - tick, flush=True)
