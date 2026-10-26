"""Round219: re-rank every round216 member/blend on the TEST-aligned metric.

TEST contains no free rows: every row has >=2 distinct parsed answers.  VAL is
798 rows of which 443 are free (accuracy is 1.0 by construction), so a model
that is better on VAL overall can be worse on the rows that actually decide
the test score.  This script scores the cached members and a blend grid on
val-hard and held-hard only.  TEST is never touched.
"""
from round22_common import ROOT, load, dump
from round4_gates import apply_gate
from common import parse_answer
import numpy as np, itertools

OUT = ROOT / 'experiments/round219'


def zrow(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)


def hard_mask(bank):
    return np.array([len({p for p in (parse_answer(c['text']) for c in r['candidates'])
                          if p is not None}) >= 2 for r in bank])


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    z = np.load(ROOT / 'experiments/round216/cache.npz')
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    y, vy = np.asarray(y), np.asarray(vy)
    held = z['folds'] == 2
    held_bank = [r for r, k in zip(b, held) if k]
    yh = y[held]
    hh = hard_mask(held_bank)
    hv = hard_mask(v)
    print('held rows', len(held_bank), 'hard', int(hh.sum()),
          'val', len(v), 'hard', int(hv.sum()), flush=True)
    ms = np.load(ROOT / 'experiments/round216/member_scores.npz')
    members = {k: (ms[k + '_h'], ms[k + '_v']) for k in ('P', 'W', 'E')}
    base_h, base_v = z['held_s1'], z['val_s1']

    def score(sh, sv):
        hc = apply_gate(held_bank, sh, 'parse_and_repair', 4)
        vc = apply_gate(v, sv, 'parse_and_repair', 4)
        rh = np.arange(len(held_bank))
        rv = np.arange(len(v))
        return (int((yh[rh, hc] == 1).sum()), int((yh[hh, hc[hh]] == 1).sum()),
                int((vy[rv, vc] == 1).sum()), int((vy[hv, vc[hv]] == 1).sum()))

    recs = []

    def add(name, sh, sv, w=None):
        h, hh_, v_, vh = score(sh, sv)
        r = dict(name=name, weights=w, held=h, held_hard=hh_, val=v_, val_hard=vh)
        recs.append(r)
        print('REC', r, flush=True)
        return r

    add('base', base_h, base_v)
    for k, (sh, sv) in members.items():
        add(k, sh, sv)
    best = None
    for w1 in (0, .05, .1, .2, .3, .4, .5, .7, 1.0):
        for w2 in (0, .05, .1, .2, .3, .4, .5):
            for w3 in (0, .05, .1, .2, .3, .4):
                sh, sv = zrow(base_h), zrow(base_v)
                for k, w in (('P', w1), ('W', w2), ('E', w3)):
                    if w:
                        sh = sh + w * zrow(members[k][0])
                        sv = sv + w * zrow(members[k][1])
                h, hh_, v_, vh = score(sh, sv)
                # selection key: worst-case hard accuracy across held and val
                key = (min(hh_ / hh.sum(), vh / hv.sum()), hh_ + vh)
                if best is None or key > best[0]:
                    best = (key, (w1, w2, w3), (h, hh_, v_, vh))
    print('BEST-BY-HARD', best, flush=True)
    recs.append(dict(name='best_by_hard', weights=best[1], held=best[2][0],
                     held_hard=best[2][1], val=best[2][2], val_hard=best[2][3]))
    dump(OUT / 'hard_metric.json', dict(base=dict(held=1477, val=759), records=recs,
                                        test_opened=False))
    print('DONE', flush=True)
