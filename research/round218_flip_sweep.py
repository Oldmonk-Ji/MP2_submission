"""Round218: can an observable anti-consensus condition improve the strict flip rule?

All remaining VAL error sits on rows where exactly one candidate is correct
(k=1: 114 rows, 71.9% accuracy, 32 of 38 hard errors).  There the strict
trace prefers the right candidate far more often than the current tau=3 flip
allows.  k itself is unobservable at inference, so this sweep conditions on
observable answer-cluster structure instead and measures VAL and HELD
together.  TEST is never touched.
"""
from round22_common import ROOT, load, dump
from round4_gates import apply_gate
from common import parse_answer
import numpy as np, itertools, json

OUT = ROOT / 'experiments/round218'


def clusters(bank):
    """cluster size of each candidate's parsed answer, 0 for unparseable."""
    out = np.zeros((len(bank), 4), int)
    for i, row in enumerate(bank):
        ps = [parse_answer(c['text']) for c in row['candidates']]
        cnt = {}
        for p in ps:
            if p is not None:
                cnt[p] = cnt.get(p, 0) + 1
        for j, p in enumerate(ps):
            out[i, j] = cnt.get(p, 0) if p is not None else 0
    return out


def sweep(s1, st, cs, bank, yy, base):
    n = len(bank)
    r = np.arange(n)
    sarg = st.argmax(1)
    top2 = np.argsort(-s1, axis=1)[:, :2]
    in_top2 = (top2 == sarg[:, None]).any(1)
    gap = st[r, sarg] - st[r, base]
    cs_base, cs_arg = cs[r, base], cs[r, sarg]
    conds = {
        'all': np.ones(n, bool),
        'top2': in_top2,
        'top2_argmax': in_top2 & (base == s1.argmax(1)),
        'anti': (cs_base >= 2) & (cs_arg == 1),
        'anti_top2': (cs_base >= 2) & (cs_arg == 1) & in_top2,
        'anti_any': cs_arg == 1,
    }
    res = {}
    for name, c in conds.items():
        row = {}
        for tau in (-1.0, 0.0, .1, .2, .3, .5, .75, 1.0, 1.5, 2.0, 3.0, 4.0):
            flip = c & (sarg != base) & (gap > tau)
            pick = np.where(flip, sarg, base)
            row[tau] = (int((yy[r, pick] == 1).sum()), int(flip.sum()))
        res[name] = row
    return res, conds, sarg, gap


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    z = np.load(ROOT / 'experiments/round216/cache.npz')
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    y, vy = np.asarray(y), np.asarray(vy)
    folds = z['folds']
    held = folds == 2
    fit = ~held
    held_bank = [r for r, k in zip(b, held) if k]
    yh = y[held]
    base_h = apply_gate(held_bank, z['held_s1'], 'parse_and_repair', 4)
    base_v = apply_gate(v, z['val_s1'], 'parse_and_repair', 4)
    cs_h = clusters(held_bank)
    cs_v = clusters(v)
    bh = int((yh[np.arange(len(held_bank)), base_h] == 1).sum())
    bv = int((vy[np.arange(len(v)), base_v] == 1).sum())
    print('base held', bh, 'val', bv, flush=True)
    rh, ch, sarg_h, gap_h = sweep(z['held_s1'], z['strict_nested'][held], cs_h,
                                   held_bank, yh, base_h)
    rv, cv, sarg_v, gap_v = sweep(z['val_s1'], z['strict_val'], cs_v, v, vy, base_v)
    out = []
    for name in rh:
        for tau in rh[name]:
            h, fh = rh[name][tau]
            vv, fv = rv[name][tau]
            rec = dict(cond=name, tau=tau, held=h, dh=h - bh, flips_h=fh,
                       val=vv, dv=vv - bv, flips_v=fv)
            out.append(rec)
            if h >= bh:
                print('GAIN', rec, flush=True)
    dump(OUT / 'flip_sweep.json', dict(base_held=bh, base_val=bv, records=out,
                                       test_opened=False))
    # observable-stratum composition
    for name, c in cv.items():
        print('val cond', name, int(c.sum()), 'held', int(ch[name].sum()), flush=True)
    np.savez(OUT / 'clusters.npz', held=cs_h, val=cs_v)
    print('DONE', flush=True)
