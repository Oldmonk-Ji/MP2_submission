"""Round223: sweep gate kind x margin on VAL and HELD with the cached scores.

The production post-processor is parse_and_repair with margin 4; round4 chose
it on a different score surface.  proposals() is computed once per kind and
the margin is applied afterwards, so this is cheap.  TEST is never touched.
"""
from round22_common import ROOT, load, dump
from round4_gates import proposals
from common import parse_answer
import numpy as np, time

OUT = ROOT / 'experiments/round223'
KINDS = ('parse_alignment', 'units', 'arithmetic_repair', 'combined', 'parse_and_repair')
LIMITS = (0, .5, 1, 2, 4, 8, 1e6)


def hard_mask(bank):
    return np.array([len({p for p in (parse_answer(c['text']) for c in r['candidates'])
                          if p is not None}) >= 2 for r in bank])


if __name__ == '__main__':
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    z = np.load(ROOT / 'experiments/round216/cache.npz')
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    y, vy = np.asarray(y), np.asarray(vy)
    held = z['folds'] == 2
    held_bank = [r for r, k in zip(b, held) if k]
    yh = y[held]
    rh, rv = np.arange(len(held_bank)), np.arange(len(v))
    hh, hv = hard_mask(held_bank), hard_mask(v)
    recs = []
    for kind in KINDS:
        oh, nh, _ = proposals(held_bank, z['held_s1'], kind)
        ov, nv, _ = proposals(v, z['val_s1'], kind)
        print(kind, 'held changed', int((oh != nh).sum()), 'val changed', int((ov != nv).sum()),
              't=%.0fs' % (time.perf_counter() - tick), flush=True)
        for lim in LIMITS:
            gh = z['held_s1'][rh, oh] - z['held_s1'][rh, nh]
            gv = z['val_s1'][rv, ov] - z['val_s1'][rv, nv]
            ph = np.where((oh != nh) & (gh <= lim), nh, oh)
            pv = np.where((ov != nv) & (gv <= lim), nv, ov)
            r = dict(kind=kind, limit=lim,
                     held=int((yh[rh, ph] == 1).sum()),
                     held_hard=int((yh[hh, ph[hh]] == 1).sum()),
                     val=int((vy[rv, pv] == 1).sum()),
                     val_hard=int((vy[hv, pv[hv]] == 1).sum()))
            recs.append(r)
            if r['held'] >= 1477 or r['held_hard'] >= 592:
                print('GATE', r, flush=True)
    dump(OUT / 'gate_sweep.json', dict(records=recs, test_opened=False,
                                       seconds=time.perf_counter() - tick))
    print('DONE', time.perf_counter() - tick, flush=True)
