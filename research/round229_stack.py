"""Learn when to trust the new quadratic expert using cross-fitted scores."""
from round22_common import ROOT, load, dump, sha
from round224_target_conclusion import strict_choice
from round225_train import PARAMS, record
from round227_quadratic import inputs, fit
from round80_selector import augment
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np
import lightgbm as lgb
import time

OUT = ROOT / "experiments/round229"


def main():
    threadpool_limits(4)
    round4_gates.trace = lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=True)
    tick = time.perf_counter()
    dump(OUT/"plan.json", dict(
        method="Append quadratic expert score/row-mean/contrast to the incumbent fusion; unchanged 200-tree recipe.",
        source_hash=sha(__file__), model_source="round227_quadratic.py",
        auxiliary_training="3-fold OOF for full fusion; 2-fold fit-only OOF plus fit-only model for held",
        selection="Single fixed architecture; held>=1477 before validation, val>760 required",
        prior_evidence="Quadratic standalone held1467; fixed blend held1479 but val759",
        test_opened=False))
    with np.load(ROOT/"experiments/round216/cache.npz") as z:
        c = {k:z[k] for k in ("x", "nx", "vx", "bs", "nbs", "vbs", "folds", "ids",
                              "held_s1", "val_s1", "strict_nested", "strict_val")}
    bank, arrays, y, ids = load("train")
    assert np.array_equal(ids, c["ids"])
    fit_rows, held = c["folds"] != 2, c["folds"] == 2
    with np.load(ROOT/"experiments/round227/scores.npz") as z:
        assert np.array_equal(z["ids"], ids)
        nested = z["design"].copy()
        nested[held] = z["held"]
    nx = augment(c["nx"], nested)
    ds = lgb.Dataset(nx[fit_rows].reshape(-1,nx.shape[-1]), label=y[fit_rows].ravel(),
                     group=np.full(fit_rows.sum(),4), init_score=c["nbs"][fit_rows].ravel())
    model = lgb.train(PARAMS, ds, num_boost_round=200)
    hs = c["nbs"][held] + model.predict(nx[held].reshape(-1,nx.shape[-1])).reshape(-1,4)
    bheld = [b for b,k in zip(bank,held) if k]
    oldh = strict_choice(bheld,c["held_s1"],c["strict_nested"][held])
    newh = strict_choice(bheld,hs,c["strict_nested"][held])
    result = dict(held=record(y[held],newh,oldh), validation_evaluated=False,test_opened=False)
    print("HELD",result,flush=True)
    model.save_model(str(OUT/"held_meta.txt"))
    if result["held"]["correct"] >= 1477:
        e,g = inputs(bank,arrays)
        oof = np.zeros_like(y)
        for f in range(3):
            w,s = fit(e,g,y,np.flatnonzero(c["folds"] != f))
            oof[c["folds"] == f] = s[c["folds"] == f]
            print("OOF",f,time.perf_counter()-tick,flush=True)
        with np.load(ROOT/"experiments/round228/validation_227.npz") as z:
            aux_val = z["scores"]
        x,vx = augment(c["x"],oof),augment(c["vx"],aux_val)
        ds = lgb.Dataset(x.reshape(-1,x.shape[-1]),label=y.ravel(),
                         group=np.full(len(y),4),init_score=c["bs"].ravel())
        model = lgb.train(PARAMS,ds,num_boost_round=200)
        vs = c["vbs"]+model.predict(vx.reshape(-1,vx.shape[-1])).reshape(-1,4)
        val,_,vy,_ = load("validation")
        oldv = strict_choice(val,c["val_s1"],c["strict_val"])
        newv = strict_choice(val,vs,c["strict_val"])
        result.update(validation=record(vy,newv,oldv),validation_evaluated=True)
        model.save_model(str(OUT/"full_meta.txt"))
        np.savez(OUT/"scores.npz",oof=oof,held=hs,validation=vs,selected=newv,ids=ids)
        print("VALIDATION",result["validation"],flush=True)
    result["seconds"]=time.perf_counter()-tick
    dump(OUT/"result.json",result)


if __name__=="__main__":
    main()
