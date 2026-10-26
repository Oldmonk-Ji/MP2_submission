"""Fixed lexical disagreement experiment; design on fit, held gate, then VAL."""
from round22_common import ROOT, load, dump, sha
from round224_target_conclusion import strict_choice
from round225_disagreement import build, pairwise_fit, score
from threadpoolctl import threadpool_limits
from functools import lru_cache
import lightgbm as lgb
import numpy as np
import time
import round4_gates

OUT = ROOT / "experiments/round225"
ALPHAS = [0, .1, .25, .5, 1.]
PARAMS = dict(objective="lambdarank", lambdarank_truncation_level=1, metric="None",
              num_leaves=7, min_data_in_leaf=100, learning_rate=.03, lambda_l2=50,
              seed=68068, verbosity=-1, num_threads=4, deterministic=True,
              force_col_wise=True)


def normalized(s):
    return (s - s.mean(1, keepdims=True)) / np.maximum(s.std(1, keepdims=True), 1e-6)


def record(y, choice, old):
    r = np.arange(len(y))
    a, b = y[r, choice], y[r, old]
    return dict(correct=int(a.sum()), baseline=int(b.sum()),
                fixes=int((a > b).sum()), harms=int((a < b).sum()),
                changed=int((choice != old).sum()))


def design_baseline(cache):
    path = OUT / "fit_only_s1.npz"
    if path.exists():
        return np.load(path)["scores"]
    scores = np.zeros_like(cache["nbs"])
    for f in (0, 1):
        tr, te = cache["folds"] == 1-f, cache["folds"] == f
        x = cache["nx"]
        ds = lgb.Dataset(x[tr].reshape(-1, x.shape[-1]), label=cache["y"][tr].ravel(),
                         group=np.full(tr.sum(), 4), init_score=cache["nbs"][tr].ravel())
        model = lgb.train(PARAMS, ds, num_boost_round=200)
        scores[te] = cache["nbs"][te] + model.predict(
            x[te].reshape(-1, x.shape[-1])).reshape(-1, 4)
    np.savez(path, scores=scores)
    return scores


def main():
    threadpool_limits(4)
    round4_gates.trace = lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=True)
    tick = time.perf_counter()
    dump(OUT / "plan.json", dict(
        method="Hashed numeric-normalized distinguishing phrases crossed with question conditions and target words",
        source_hash=sha(ROOT / "research/round225_disagreement.py"),
        C=1, features=65536, alphas=ALPHAS, seed=225225,
        selection="Choose alpha on fit folds 0/1 only; require no loss in either design fold. Then held>=1477; only then VAL.",
        limitation="Fit design is a reused development split; outer held is excluded from design training.",
        test_opened=False))
    with np.load(ROOT / "experiments/round216/cache.npz") as z:
        c = {k: z[k] for k in ("nx", "nbs", "folds", "y", "ids",
             "held_s1", "val_s1", "strict_nested", "strict_val")}
    bank, _, y, ids = load("train")
    assert np.array_equal(ids, c["ids"]) and np.array_equal(y, c["y"])
    fit, held = c["folds"] != 2, c["folds"] == 2
    design_bank = [b for b, keep in zip(bank, fit) if keep]
    held_bank = [b for b, keep in zip(bank, held) if keep]
    base = design_baseline(c)
    x = build(bank)
    print("FEATURES", x.shape, x.nnz, "seconds", time.perf_counter()-tick, flush=True)
    perm = np.array([2, 0, 3, 1])
    sample = bank[:3]
    perm_bank = [dict(question=r["question"],
                      candidates=[r["candidates"][j] for j in perm]) for r in sample]
    diff = build(perm_bank) - x[:12][np.array([4*i+j for i in range(3) for j in perm])]
    assert diff.nnz == 0 or np.abs(diff.data).max() < 1e-6
    oof = np.zeros_like(y)
    for f in (0, 1):
        weight = pairwise_fit(x, y, np.flatnonzero(c["folds"] == 1-f))
        oof[c["folds"] == f] = score(x, weight)[c["folds"] == f]
        print("DESIGN_MODEL", f, "seconds", time.perf_counter()-tick, flush=True)
    old = strict_choice(design_bank, base[fit], c["strict_nested"][fit])
    design = []
    for alpha in ALPHAS:
        choice = strict_choice(design_bank, base[fit] + alpha*normalized(oof[fit]),
                               c["strict_nested"][fit])
        gains = []
        for f in (0, 1):
            mask = c["folds"][fit] == f
            yy = y[fit][mask]
            r = np.arange(len(yy))
            gains.append(int((yy[r, choice[mask]] - yy[r, old[mask]]).sum()))
        rec = dict(alpha=alpha, fold_gains=gains, **record(y[fit], choice, old))
        design.append(rec)
        print("DESIGN", rec, flush=True)
    eligible = [d for d in design if min(d["fold_gains"]) >= 0]
    chosen = max(eligible, key=lambda d: (d["correct"], -d["alpha"]))
    alpha = chosen["alpha"]
    weight = pairwise_fit(x, y, np.flatnonzero(fit))
    hs = score(x, weight)[held]
    oldh = strict_choice(held_bank, c["held_s1"], c["strict_nested"][held])
    newh = strict_choice(held_bank, c["held_s1"] + alpha*normalized(hs),
                        c["strict_nested"][held])
    result = dict(alpha=alpha, design=design, held=record(y[held], newh, oldh),
                  standalone_held=int(y[held][np.arange(held.sum()), hs.argmax(1)].sum()),
                  test_opened=False, validation_evaluated=False)
    print("HELD", result["held"], "alpha", alpha, "standalone", result["standalone_held"], flush=True)
    np.savez(OUT / "held_model.npz", weight=weight)
    np.savez(OUT / "scores.npz", design=oof, held=hs, ids=ids)
    if alpha > 0 and result["held"]["correct"] >= 1477:
        # Alpha is frozen before this first official validation evaluation.
        val, _, vy, _ = load("validation")
        weight = pairwise_fit(x, y, np.arange(len(bank)))
        vs = score(build(val), weight)
        oldv = strict_choice(val, c["val_s1"], c["strict_val"])
        newv = strict_choice(val, c["val_s1"] + alpha*normalized(vs), c["strict_val"])
        result.update(validation=record(vy, newv, oldv), validation_evaluated=True)
        np.savez(OUT / "full_model.npz", weight=weight)
        print("VALIDATION", result["validation"], flush=True)
    result["seconds"] = time.perf_counter()-tick
    dump(OUT / "result.json", result)


if __name__ == "__main__":
    main()
