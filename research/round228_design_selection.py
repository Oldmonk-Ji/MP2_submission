"""Select new-model blend strength by total FIT-fold accuracy only.

The previous per-fold nonnegative filter was an extra research heuristic,
not a task constraint. This amendment is based on FIT design records. It
does not sweep HELD or VAL; each model gets one FIT-selected alpha.
"""
from round22_common import ROOT, load, dump, read, sha
from round224_target_conclusion import strict_choice
from round225_train import normalized, record
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
import time

OUT = ROOT / "experiments/round228"


def main():
    threadpool_limits(4)
    round4_gates.trace = lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=True)
    selected = {}
    for source in (227, 225):
        design = read(ROOT / f"experiments/round{source}/result.json")["design"]
        selected[str(source)] = max(design, key=lambda d: (d["correct"], -d["alpha"]))
    dump(OUT / "plan.json", dict(
        reason="Drop the additional per-design-fold nonnegative heuristic; use aggregate fit development accuracy.",
        selection_basis="Only design records from train folds 0 and 1",
        held_alpha_sweep=False, validation_alpha_sweep=False,
        selected=selected, test_opened=False, source_hash=sha(__file__)))
    bank, arrays, y, ids = load("train")
    with np.load(ROOT / "experiments/round216/cache.npz") as z:
        c = {k:z[k] for k in ("folds", "held_s1", "strict_nested", "val_s1", "strict_val")}
    held = c["folds"] == 2
    bheld = [b for b, keep in zip(bank, held) if keep]
    oldh = strict_choice(bheld, c["held_s1"], c["strict_nested"][held])
    results = {}
    for source in (227, 225):
        tick = time.perf_counter()
        with np.load(ROOT / f"experiments/round{source}/scores.npz") as z:
            assert np.array_equal(z["ids"], ids)
            hs = z["held"]
        alpha = selected[str(source)]["alpha"]
        newh = strict_choice(bheld, c["held_s1"] + alpha*normalized(hs), c["strict_nested"][held])
        result = dict(source=source, alpha=alpha, held=record(y[held], newh, oldh),
                      validation_evaluated=False, test_opened=False)
        print("HELD", result, flush=True)
        if alpha > 0 and result["held"]["correct"] >= 1477:
            if source == 227:
                from round227_quadratic import inputs, fit, scores
                e, g = inputs(bank, arrays)
                w, _ = fit(e, g, y, np.arange(len(bank)))
                val, va, vy, _ = load("validation")
                vs = scores(*inputs(val, va), w)
                np.savez(OUT/f"full_{source}.npz", **w)
            else:
                from round225_disagreement import build, pairwise_fit, score
                x = build(bank)
                w = pairwise_fit(x, y, np.arange(len(bank)))
                val, _, vy, _ = load("validation")
                vs = score(build(val), w)
                np.savez(OUT/f"full_{source}.npz", weight=w)
            oldv = strict_choice(val, c["val_s1"], c["strict_val"])
            newv = strict_choice(val, c["val_s1"] + alpha*normalized(vs), c["strict_val"])
            result.update(validation=record(vy, newv, oldv), validation_evaluated=True,
                          added_asset_bytes=(OUT/f"full_{source}.npz").stat().st_size)
            np.savez(OUT/f"validation_{source}.npz", scores=vs, selected=newv)
            print("VALIDATION", result, flush=True)
        result["seconds"] = time.perf_counter()-tick
        results[str(source)] = result
        dump(OUT/"result.json", results)


if __name__ == "__main__":
    main()
