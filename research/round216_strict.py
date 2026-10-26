"""Round216: strict-verification signal quality on the test-like subset.

Loads the round209 trace masks and round210 contrastive scores, then measures
how much of the disagreement-row error mass the verification signal can see:
coverage of verified candidates, strict-argmax accuracy, and the 'verified
filter' rule. No retraining.
"""
from round22_common import ROOT, load, read, dump
from common import parse_answer
import numpy as np

OUT = ROOT / 'experiments/round216'


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
    with np.load(ROOT / 'experiments/round211/verified_masks.npz') as z:
        mv = z['val'].astype(bool)
    with np.load(ROOT / 'experiments/round210/scores.npz') as z:
        strict = z['validation']
    ok_sel = vy[np.arange(n), sel] == 1
    r = np.arange(n)
    res = {}
    # verified coverage
    nver = mv.sum(1)
    res['verified_per_row'] = {str(k): int((nver == k).sum()) for k in range(5)}
    res['rows_any_verified'] = int((nver > 0).sum())
    res['hard_rows_any_verified'] = int((nver > 0)[hard].sum())
    res['hard_n'] = int(hard.sum())
    # correctness of the chosen candidate vs verification state
    res['chosen_verified_when_correct'] = int((mv[r, sel] & ok_sel)[hard].sum())
    res['chosen_verified_when_wrong'] = int((mv[r, sel] & ~ok_sel)[hard].sum())
    # strict argmax (ties -> first) accuracy on hard rows
    s_arg = strict.argmax(1)
    res['strict_argmax_hard'] = dict(
        correct=int((vy[r, s_arg] == 1)[hard].sum()), n=int(hard.sum()),
        correct_free=int((vy[r, s_arg] == 1)[~hard].sum()))
    # verified-filter rule: among verified candidates pick max strict; else fusion
    pick = sel.copy()
    for i in np.flatnonzero(hard & (nver > 0)):
        cand = np.flatnonzero(mv[i])
        j = cand[np.argmax(strict[i, cand])]
        pick[i] = j
    ok2 = vy[r, pick] == 1
    changed = pick != sel
    res['verified_filter_hard'] = dict(
        correct=int(ok2[hard].sum()), n=int(hard.sum()),
        flips=int(changed[hard].sum()),
        answer_changes=int(((pick != sel) & np.array(
            [parsed[i][pick[i]] != parsed[i][sel[i]] for i in range(n)]))[hard].sum()),
        fix=int((changed & ok2 & ~ok_sel)[hard].sum()),
        harm=int((changed & ~ok2 & ok_sel)[hard].sum()))
    # among hard ERROR rows: is the correct candidate verified?
    err = hard & ~ok_sel
    any_ok = vy == 1
    corr_verified = np.array([bool((mv[i] & any_ok[i]).any()) for i in range(n)])
    res['hard_errors'] = dict(n=int(err.sum()),
                              correct_verified=int((err & corr_verified).sum()),
                              any_verified=int((err & (nver > 0)).sum()))
    print(res, flush=True)
    dump(OUT / 'strict_signal.json', res)
