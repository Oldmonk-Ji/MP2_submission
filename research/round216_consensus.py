"""Round216: consensus exploitation analysis on train (OOF-free, label view)
and validation. Compares 'pick largest answer cluster' with the frozen
selector and measures how often consensus misleads."""
from round22_common import ROOT, load, read, dump
from common import parse_answer
import numpy as np

OUT = ROOT / 'experiments/round216'


def cluster_info(ps):
    cnt = {}
    for j, p in enumerate(ps):
        if p is not None:
            cnt.setdefault(p, []).append(j)
    sizes = sorted((len(v) for v in cnt.values()), reverse=True)
    return cnt, sizes


if __name__ == '__main__':
    res = {}
    for split in ('train', 'validation'):
        b, a, y, ids = load(split)
        n = len(b)
        parsed = [[parse_answer(c['text']) for c in row['candidates']] for row in b]
        ndist = np.array([len({p for p in ps if p is not None}) for ps in parsed])
        hard = ndist >= 2
        # largest-cluster rule, ties -> index 0
        pick = np.zeros(n, int)
        for i in range(n):
            cnt, sizes = cluster_info(parsed[i])
            if not cnt:
                continue
            best = sizes[0]
            winners = [v for v in cnt.values() if len(v) == best]
            pick[i] = winners[0][0]
        res[split] = {}
        res[split]['largest_cluster_all'] = round(float(y[np.arange(n), pick].mean()), 4)
        m = hard
        idx = np.flatnonzero(m)
        res[split]['largest_cluster_hard'] = round(float(y[idx, pick[idx]].mean()), 4)
        # fusion (validation only: frozen predictions; train: majority baseline only)
        if split == 'validation':
            sel = np.asarray(read(ROOT / 'experiments/round214/validation_strict_gap_t3_predictions.json')['selected'], int)
            ok = y[np.arange(n), sel] == 1
            res[split]['frozen_all'] = round(float(ok.mean()), 4)
            res[split]['frozen_hard'] = round(float(ok[m].mean()), 4)
            # how often is the chosen answer in the largest cluster?
            chosen_in_largest = np.zeros(n, bool)
            correct_cluster_size = np.zeros(n)
            chosen_cluster_size = np.zeros(n)
            for i in np.flatnonzero(m):
                cnt, sizes = cluster_info(parsed[i])
                best = sizes[0]
                winners = [v for v in cnt.values() if len(v) == best]
                large = {j for v in winners for j in v}
                chosen_in_largest[i] = sel[i] in large
                chosen_cluster_size[i] = len(cnt[parsed[i][sel[i]]]) if parsed[i][sel[i]] is not None else 0
                if y[i, sel[i]] == 0:
                    ok_clusters = [len(v) for j, v in cnt.items() if parsed[i][sel[i]] is not None and j != parsed[i][sel[i]] and any(y[i, k] for k in v)]
                    correct_cluster_size[i] = ok_clusters[0] if ok_clusters else 0
            err = m & ~ok
            res[split]['errors_total'] = int(err.sum())
            res[split]['errors_chosen_in_largest'] = int((err & chosen_in_largest).sum())
            res[split]['correct_rows_chosen_in_largest'] = int((ok & m & chosen_in_largest).sum())
            res[split]['rows_hard'] = int(m.sum())
            # accuracy by chosen cluster size
            for k in (1, 2, 3, 4):
                mk = m & (chosen_cluster_size == k)
                res[split][f'acc_chosen_cluster_{k}'] = dict(
                    n=int(mk.sum()), acc=round(float(ok[mk].mean()), 4) if mk.any() else None)
            # error cluster sizes
            res[split]['error_chosen_cluster_hist'] = {
                str(k): int((err & (chosen_cluster_size == k)).sum()) for k in (1, 2, 3, 4)}
            res[split]['error_correct_cluster_hist'] = {
                str(k): int((err & (correct_cluster_size == k)).sum()) for k in (0, 1, 2, 3, 4)}
        print(split, res[split], flush=True)
    dump(OUT / 'consensus.json', res)
