"""Independent scratch comparator, with design-only blending and held gate."""
from round22_common import ROOT, load, dump, sha
from round224_target_conclusion import strict_choice
from round225_train import design_baseline, normalized, record, ALPHAS
from round226_model import inputs, train, scores, SEED, EPOCHS
from threadpoolctl import threadpool_limits
from functools import lru_cache
import numpy as np
import time
import round4_gates

OUT = ROOT / "experiments/round226"


def main():
    threadpool_limits(4)
    round4_gates.trace = lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=True)
    tick = time.perf_counter()
    dump(OUT / "plan.json", dict(
        method="Scratch conditional residual encoder with explicit same-answer/opposing-answer pooling",
        source_hash=sha(ROOT / "research/round226_model.py"), seed=SEED, epochs=EPOCHS,
        external_weights=False, gpu_used=False, cpu_threads=4, alphas=ALPHAS,
        selection="Design folds 0/1 choose alpha, no loss in either; held >=1477 gates first VAL evaluation.",
        resource_policy="Standalone first; assess compression only if complementary gain is demonstrated.",
        test_opened=False))
    with np.load(ROOT / "experiments/round216/cache.npz") as z:
        c = {k: z[k] for k in ("nx", "nbs", "folds", "y", "ids",
             "held_s1", "val_s1", "strict_nested", "strict_val")}
    bank, arrays, y, ids = load("train")
    assert np.array_equal(ids, c["ids"]) and np.array_equal(y, c["y"])
    fit, held = c["folds"] != 2, c["folds"] == 2
    bfit = [b for b, keep in zip(bank, fit) if keep]
    bheld = [b for b, keep in zip(bank, held) if keep]
    data = inputs(bank, arrays)
    print("INPUTS_READY", time.perf_counter()-tick, flush=True)
    base = design_baseline(c)
    oof = np.zeros_like(y)
    histories = {}
    for f in (0, 1):
        tr = np.flatnonzero(c["folds"] == 1-f)
        te = np.flatnonzero(c["folds"] == f)
        model, history = train(data, y, tr, OUT / f"design_fold{f}.npz")
        histories[str(f)] = history
        oof[te] = scores(model, data, te)
    old = strict_choice(bfit, base[fit], c["strict_nested"][fit])
    design = []
    for alpha in ALPHAS:
        choice = strict_choice(bfit, base[fit] + alpha*normalized(oof[fit]),
                               c["strict_nested"][fit])
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
    model, history = train(data, y, np.flatnonzero(fit), OUT / "held_model.npz")
    histories["held_model"] = history
    hs = scores(model, data, np.flatnonzero(held))
    # Contract checks for this architecture before accepting any scores.
    perm = np.array([2, 0, 3, 1])
    small = {k: v[:3].copy() for k, v in data.items()}
    shuffled = {k: v[:, perm].copy() for k, v in small.items()}
    shuffled["same"] = shuffled["same"][:, :, perm]
    original = scores(model, small)
    assert np.allclose(scores(model, shuffled), original[:, perm], atol=2e-4)
    assert np.allclose(scores(model, {k: v[:1] for k, v in small.items()}), original[:1], atol=2e-4)
    oldh = strict_choice(bheld, c["held_s1"], c["strict_nested"][held])
    newh = strict_choice(bheld, c["held_s1"] + alpha*normalized(hs),
                        c["strict_nested"][held])
    standalone = int(y[held][np.arange(held.sum()), hs.argmax(1)].sum())
    result = dict(alpha=alpha, design=design, held=record(y[held], newh, oldh),
                  standalone_held=standalone, histories=histories,
                  standalone_asset_bytes=(OUT/"held_model.npz").stat().st_size,
                  contract_checks="candidate permutation and batch invariance passed",
                  test_opened=False, validation_evaluated=False)
    print("HELD", result["held"], "alpha", alpha, "standalone", standalone, flush=True)
    np.savez(OUT / "scores.npz", design=oof, held=hs, ids=ids)
    if alpha > 0 and result["held"]["correct"] >= 1477:
        model, history = train(data, y, np.arange(len(bank)), OUT / "full_model.npz")
        val, va, vy, _ = load("validation")
        vs = scores(model, inputs(val, va))
        oldv = strict_choice(val, c["val_s1"], c["strict_val"])
        newv = strict_choice(val, c["val_s1"] + alpha*normalized(vs), c["strict_val"])
        result.update(validation=record(vy, newv, oldv), validation_evaluated=True,
                      full_history=history)
        np.savez(OUT / "validation_scores.npz", scores=vs)
        print("VALIDATION", result["validation"], flush=True)
    result["seconds"] = time.perf_counter()-tick
    dump(OUT / "result.json", result)


if __name__ == "__main__":
    main()
