"""Round216: training-composition and feature-block experiments.

Variants:
  * hard-only training (rows with >=2 distinct parsed answers)
  * hard-row oversampling via weights
  * extra feature blocks: answer-cluster structure, strict-score detail,
    chain-number overlap between candidates
  * seed ensembles
Held uses nested features (fit-fold training); VAL uses exact104 + all rows.
"""
from round22_common import ROOT, load, read, dump
from round4_gates import apply_gate
from threadpoolctl import threadpool_limits
import numpy as np, lightgbm as lgb, time

OUT = ROOT / 'experiments/round216'
BASE = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', num_leaves=7,
            min_data_in_leaf=100, learning_rate=.03, lambda_l2=50, seed=68068, verbosity=-1,
            num_threads=4, deterministic=True, force_col_wise=True)


def strict_block(strict, verified):
    n = len(strict)
    s = strict
    mx = s.max(1, keepdims=True)
    mn = s.min(1, keepdims=True)
    span = np.maximum(mx - mn, 1e-6)
    rank = (-s).argsort(1).argsort(1)
    vcount = verified.sum(1, keepdims=True)
    prob = np.exp(s - mx) / np.exp(s - mx).sum(1, keepdims=True)
    return np.stack([
        (s - mn) / span,
        (s - mx) / span,
        prob,
        rank / 3.0,
        (rank == 0).astype(float),
        np.broadcast_to(vcount / 4.0, s.shape),
        verified.astype(float),
        np.broadcast_to((vcount == 1).astype(float), s.shape),
    ], 2).astype(np.float32)


def blocks(cache, split):
    if split == 'train':
        f = np.concatenate([cache['tfeat'], cache['tchain'],
                            strict_block(cache['strict_oof'], cache['verified'])], 2)
    else:
        f = np.concatenate([cache['vfeat'], cache['vchain'],
                            strict_block(cache['strict_val'], cache['verified_val'])], 2)
    return f


def run(cache, b, v, y, vy, cfg, label, seeds=(68068,), use_extra=None,
        train_rows='all', weight=None):
    fit = cache['folds'] != 2
    held = ~fit
    x, nx, vx = cache['x'], cache['nx'], cache['vx']
    bs, nbs, vbs = cache['bs'], cache['nbs'], cache['vbs']
    if use_extra:
        te, ve = blocks(cache, 'train'), blocks(cache, 'val')
        x = np.concatenate([x, te], 2)
        nx = np.concatenate([nx, te], 2)
        vx = np.concatenate([vx, ve], 2)
    hard = cache['tfeat'][:, 0, 3] >= 2
    train_mask = (fit & hard) if train_rows == 'hard' else fit
    full_mask = hard if train_rows == 'hard' else np.ones(len(b), bool)
    w = None
    if weight == 'hard3':
        w = np.where(hard, 3.0, 1.0).astype(np.float32)
    rounds = cfg.get('rounds', 200)
    p = dict(BASE, **{k: v2 for k, v2 in cfg.items() if k != 'rounds'})
    ph, pv = [], []
    for seed in seeds:
        pp = dict(p, seed=seed)
        wh = np.repeat(w[train_mask], 4) if w is not None else None
        ds = lgb.Dataset(nx[train_mask].reshape(-1, nx.shape[-1]), label=y[train_mask].ravel(),
                         group=np.full(int(train_mask.sum()), 4), init_score=nbs[train_mask].ravel(),
                         weight=wh)
        hm = lgb.train(pp, ds, num_boost_round=rounds)
        ph.append(nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4))
        wf = np.repeat(w[full_mask], 4) if w is not None else None
        ds = lgb.Dataset(x[full_mask].reshape(-1, x.shape[-1]), label=y[full_mask].ravel(),
                         group=np.full(int(full_mask.sum()), 4), init_score=bs[full_mask].ravel(),
                         weight=wf)
        m = lgb.train(pp, ds, num_boost_round=rounds)
        pv.append(vbs + m.predict(vx.reshape(-1, vx.shape[-1])).reshape(-1, 4))
    held_s1 = np.mean(ph, 0)
    val_s1 = np.mean(pv, 0)
    held_bank = [r for r, k in zip(b, held) if k]
    hc = apply_gate(held_bank, held_s1, 'parse_and_repair', 4)
    vc = apply_gate(v, val_s1, 'parse_and_repair', 4)
    hardv = cache['vhard']
    rec = dict(label=label, cfg=cfg, seeds=list(seeds), train_rows=train_rows, weight=weight,
               extra=bool(use_extra), val=int((vy[np.arange(len(v)), vc] == 1).sum()),
               held=int((y[held][np.arange(int(held.sum())), hc] == 1).sum()),
               val_hard=int((vy[hardv, vc[hardv]] == 1).sum()),
               val_free=int((vy[~hardv, vc[~hardv]] == 1).sum()))
    print('EXP', rec, flush=True)
    return rec, val_s1


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    cache = dict(np.load(OUT / 'cache.npz'))
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    from common import parse_answer
    parsed = [[parse_answer(c['text']) for c in row['candidates']] for row in v]
    cache['vhard'] = np.array([len({p for p in ps if p is not None}) >= 2 for ps in parsed])
    REC = []
    REC.append(run(cache, b, v, y, vy, {}, 'base')[0])
    REC.append(run(cache, b, v, y, vy, {}, 'base_seed5',
                    seeds=(68068, 7, 19, 73, 137))[0])
    REC.append(run(cache, b, v, y, vy, {}, 'extra', use_extra=True)[0])
    REC.append(run(cache, b, v, y, vy, {}, 'hard_only', train_rows='hard')[0])
    REC.append(run(cache, b, v, y, vy, {}, 'hard3', weight='hard3')[0])
    REC.append(run(cache, b, v, y, vy, {}, 'extra_hard3', use_extra=True, weight='hard3')[0])
    REC.append(run(cache, b, v, y, vy, {}, 'extra_seed5', use_extra=True,
                   seeds=(68068, 7, 19, 73, 137))[0])
    dump(OUT / 'exp_phase1.json', dict(records=REC, test_opened=False,
                                       seconds=time.perf_counter() - tick))
    print('DONE', time.perf_counter() - tick, flush=True)
