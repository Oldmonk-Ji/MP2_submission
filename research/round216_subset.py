"""Round216: test-like subset analysis (rows with >=2 distinct parsed answers).

All TEST questions are constructed to have answer disagreement, so the
validation rows with a single distinct parsed answer are 'free' and mask
real differences. This script measures the frozen selector, the
majority-cluster rule, and hybrid rules on the disagreement subset only.
"""
from round22_common import ROOT, load, read, dump
from common import parse_answer
import numpy as np

OUT = ROOT / 'experiments/round216'
OUT.mkdir(exist_ok=True)


def clusters(ps):
    cnt = {}
    for j, p in enumerate(ps):
        if p is not None:
            cnt.setdefault(p, []).append(j)
    return cnt


if __name__ == '__main__':
    v, va, vy, _ = load('validation')
    n = len(v)
    pred = read(ROOT / 'experiments/round214/validation_strict_gap_t3_predictions.json')
    sel = np.asarray(pred['selected'], int)
    parsed = [[parse_answer(c['text']) for c in row['candidates']] for row in v]
    ndist = np.array([len({p for p in ps if p is not None}) for ps in parsed])
    hard = ndist >= 2
    rows = {}

    def acc(mask_sel, m):
        ok = vy[m, mask_sel[m]] == 1
        return int(ok.sum()), int(m.sum())

    # current selector
    c, t = acc(sel, hard)
    rows['frozen_on_hard'] = dict(correct=c, n=t, acc=round(c / t, 4))
    c, t = acc(sel, ~hard)
    rows['frozen_on_free'] = dict(correct=c, n=t, acc=round(c / t, 4))

    # majority cluster with different tie policies
    def majority(mode, tie_use_sel=None):
        out = np.full(n, -1)
        for i in range(n):
            if not hard[i]:
                out[i] = 0
                continue
            cnt = clusters(parsed[i])
            if not cnt:
                out[i] = sel[i] if tie_use_sel is not None else 0
                continue
            best = max(len(x) for x in cnt.values())
            winners = [x for x in cnt.values() if len(x) == best]
            cands = [j for w in winners for j in w]
            if len(winners) == 1 and len(winners[0]) == 1:
                out[i] = winners[0][0]
            elif tie_use_sel is not None:
                out[i] = max(cands, key=lambda j: (sel[j] == 0, j)) if False else (
                    sel[i] if sel[i] in cands else cands[0])
            else:
                out[i] = -1
        return out

    for name, tie in (('majority_no_tiebreak', None),):
        m = majority('plain', tie)
        valid = m >= 0
        ok = (vy[np.arange(n), np.clip(m, 0, 3)] == 1) & valid
        rows[name] = dict(covered=int(valid.sum()), correct=int(ok.sum()),
                          acc=round(float(ok.sum() / max(valid.sum(), 1)), 4))

    # majority restricted to hard rows, tie -> frozen choice
    fix = np.zeros(n, bool)
    maj = np.full(n, -1)
    for i in range(n):
        if not hard[i]:
            maj[i] = sel[i]
            continue
        cnt = clusters(parsed[i])
        best = max(len(x) for x in cnt.values()) if cnt else 0
        winners = [x for x in cnt.values() if len(x) == best] if cnt else []
        if len(winners) == 1:
            maj[i] = winners[0][0]
        else:
            maj[i] = sel[i]
    ok = vy[np.arange(n), maj] == 1
    rows['majority_or_frozen'] = dict(correct=int(ok.sum()), n=n,
                                      hard_correct=int(ok[hard].sum()), hard_n=int(hard.sum()),
                                      flips_vs_frozen=int((maj != sel).sum()),
                                      fix=int(((maj != sel) & ok).sum()),
                                      harm=int(((maj != sel) & ~ok).sum()))

    # breakdown by split shape
    shape = []
    for i in np.flatnonzero(hard):
        cnt = clusters(parsed[i])
        sizes = sorted((len(x) for x in cnt.values()), reverse=True)
        key = 'x'.join(map(str, sizes))
        shape.append((key, int(vy[i, sel[i]] == 1)))
    from collections import Counter
    agg = {}
    for key, okk in shape:
        a = agg.setdefault(key, [0, 0])
        a[0] += okk
        a[1] += 1
    rows['by_cluster_shape'] = {k: dict(correct=v2[0], n=v2[1], acc=round(v2[0] / v2[1], 4))
                                for k, v2 in sorted(agg.items(), key=lambda kv: -kv[1][1])}
    # hard accuracy by ndist
    rows['by_ndist'] = {}
    for k in (2, 3, 4):
        m = ndist == k
        ok = vy[m, sel[m]] == 1
        rows['by_ndist'][str(k)] = dict(correct=int(ok.sum()), n=int(m.sum()),
                                        acc=round(float(ok.mean()), 4))
    print(rows, flush=True)
    dump(OUT / 'subset_analysis.json', rows)
