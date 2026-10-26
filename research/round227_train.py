"""Conditional quadratic model: fixed design, then held gate, then validation."""
from round22_common import ROOT, load, dump, sha
from round224_target_conclusion import strict_choice
from round225_train import design_baseline, normalized, record, ALPHAS
from round227_quadratic import inputs, fit as train, scores
from threadpoolctl import threadpool_limits
from functools import lru_cache
import numpy as np
import time
import round4_gates

OUT = ROOT / "experiments/round227"


def main():
    threadpool_limits(4)
    round4_gates.trace = lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=True)
    tick = time.perf_counter()
    dump(OUT / "plan.json", dict(
        method="Whitened residual128 x context32 bilinear and residual32 quadratic candidate ranker",
        source_hash=sha(ROOT/"research/round227_quadratic.py"),
        C=1, seed=227227, alphas=ALPHAS,
        selection="Fit-fold alpha with nonnegative gain in both folds; held>=1477 before VAL.",
        pca="Fitted separately on each training partition; no held fitting",
        test_opened=False))
    with np.load(ROOT / "experiments/round216/cache.npz") as z:
        c = {k: z[k] for k in ("nx", "nbs", "folds", "y", "ids",
             "held_s1", "val_s1", "strict_nested", "strict_val")}
    bank, arrays, y, ids = load("train")
    assert np.array_equal(ids, c["ids"]) and np.array_equal(y, c["y"])
    fit, held = c["folds"] != 2, c["folds"] == 2
    bfit = [b for b, keep in zip(bank, fit) if keep]
    bheld = [b for b, keep in zip(bank, held) if keep]
    e, g = inputs(bank, arrays)
    base = design_baseline(c)
    oof = np.zeros_like(y)
    for f in (0, 1):
        w, s = train(e, g, y, np.flatnonzero(c["folds"] == 1-f))
        oof[c["folds"] == f] = s[c["folds"] == f]
        print("DESIGN_MODEL", f, time.perf_counter()-tick, flush=True)
    old = strict_choice(bfit, base[fit], c["strict_nested"][fit])
    design = []
    for alpha in ALPHAS:
        choice = strict_choice(bfit, base[fit] + alpha*normalized(oof[fit]), c["strict_nested"][fit])
        gains = []
        for f in (0, 1):
            mask = c["folds"][fit] == f
            yy = y[fit][mask]
            r = np.arange(len(yy))
            gains.append(int((yy[r, choice[mask]]-yy[r, old[mask]]).sum()))
        rec = dict(alpha=alpha, fold_gains=gains, **record(y[fit], choice, old))
        design.append(rec)
        print("DESIGN", rec, flush=True)
    chosen = max([d for d in design if min(d["fold_gains"]) >= 0],
                 key=lambda d: (d["correct"], -d["alpha"]))
    alpha = chosen["alpha"]
    w, s = train(e, g, y, np.flatnonzero(fit))
    perm = np.array([2, 0, 3, 1])
    assert np.allclose(scores(e[:3, perm], g[:3, perm], w), s[:3, perm], atol=2e-4)
    assert np.allclose(scores(e[:1], g[:1], w), s[:1], atol=2e-4)
    hs = s[held]
    oldh = strict_choice(bheld, c["held_s1"], c["strict_nested"][held])
    newh = strict_choice(bheld, c["held_s1"] + alpha*normalized(hs), c["strict_nested"][held])
    np.savez(OUT/"held_model.npz", **w)
    np.savez(OUT/"scores.npz", design=oof, held=hs, ids=ids)
    result = dict(alpha=alpha, design=design, held=record(y[held], newh, oldh),
                  standalone_held=int(y[held][np.arange(held.sum()), hs.argmax(1)].sum()),
                  asset_bytes=(OUT/"held_model.npz").stat().st_size,
                  test_opened=False, validation_evaluated=False)
    print("HELD", result["held"], "alpha", alpha, "standalone", result["standalone_held"], flush=True)
    if alpha > 0 and result["held"]["correct"] >= 1477:
        w, _ = train(e, g, y, np.arange(len(bank)))
        val, va, vy, _ = load("validation")
        vs = scores(*inputs(val, va), w)
        oldv = strict_choice(val, c["val_s1"], c["strict_val"])
        newv = strict_choice(val, c["val_s1"] + alpha*normalized(vs), c["strict_val"])
        result.update(validation=record(vy, newv, oldv), validation_evaluated=True)
        np.savez(OUT/"full_model.npz", **w)
        np.savez(OUT/"validation_scores.npz", scores=vs)
        print("VALIDATION", result["validation"], flush=True)
    result["seconds"] = time.perf_counter()-tick
    dump(OUT/"result.json", result)


if __name__ == "__main__":
    main()
