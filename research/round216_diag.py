"""Round216 diagnostics: answer-structure composition of train/val and
accuracy conditional on the number of distinct parsed answers.

Uses only official candidate texts + official parse_answer on VALIDATION
references (allowed). No test access.
"""
from round22_common import ROOT, load, read, dump
from common import parse_answer
import numpy as np

OUT = ROOT / 'experiments/round216'
OUT.mkdir(exist_ok=True)

if __name__ == '__main__':
    stats = {}
    for split in ('train', 'validation'):
        b, a, y, ids = load(split)
        n = len(b)
        parsed = []
        ndist = np.zeros(n, int)
        for i, row in enumerate(b):
            ps = [parse_answer(c['text']) for c in row['candidates']]
            parsed.append(ps)
            ndist[i] = len({p for p in ps if p is not None})
        # per-candidate correctness
        acc_by_pos = y.mean(0)
        # majority-cluster rule: choose a candidate from the largest answer cluster
        # (ties or none -> no prediction); measure coverage/accuracy
        mc_ok = 0
        mc_n = 0
        mc_tie = 0
        for i, ps in enumerate(parsed):
            vals = [p for p in ps if p is not None]
            if not vals:
                continue
            cnt = {}
            for j, p in enumerate(ps):
                if p is not None:
                    cnt.setdefault(p, []).append(j)
            best = max(len(v) for v in cnt.values())
            winners = [v for v in cnt.values() if len(v) == best]
            if len(winners) != 1:
                mc_tie += 1
                continue
            mc_n += 1
            mc_ok += int(any(y[i, j] for j in winners[0]))
        # any-correct and best-of-position stats
        rows = []
        for k in sorted(set(ndist.tolist())):
            m = ndist == k
            rows.append(dict(distinct=int(k), n=int(m.sum()), frac=round(float(m.mean()), 4),
                             acc_best=round(float(y[m].max(1).mean()), 4)))
        stats[split] = dict(
            n=n,
            distinct_hist={int(k): int((ndist == k).sum()) for k in sorted(set(ndist.tolist()))},
            acc_by_position=[round(float(v), 4) for v in acc_by_pos],
            any_correct=round(float(y.max(1).mean()), 4),
            majority_cluster=dict(covered=mc_n, ties=int(mc_tie),
                                  accuracy=round(mc_ok / max(mc_n, 1), 4),
                                  coverage_of_attempt=int(mc_n) / n),
            by_distinct=rows,
            parsed_none=int(sum(p is None for ps in parsed for p in ps)),
        )
        print(split, stats[split], flush=True)
    dump(OUT / 'answer_structure.json', stats)
