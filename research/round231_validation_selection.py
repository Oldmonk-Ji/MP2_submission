"""Explicit validation development for the new quadratic expert, never TEST."""
from round22_common import ROOT, load, dump, sha
from round224_target_conclusion import strict_choice
from round225_train import normalized, record
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np

OUT=ROOT/"experiments/round231"
ALPHAS=[.05,.1,.2,.25,.35,.5]


def main():
    threadpool_limits(4)
    round4_gates.trace=lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=True)
    dump(OUT/"plan.json",dict(
        method="Finite calibration grid for the newly trained quadratic expert",
        modes=["raw","row_normalized"],alphas=ALPHAS,
        validation_role="Reused development/selection set, not an unbiased generalization estimate.",
        selection="Highest val count subject to held>=1477; adopt only val>760; then held count and smaller alpha.",
        test_opened=False,source_hash=sha(__file__)))
    bank,_,y,_=load("train")
    with np.load(ROOT/"experiments/round216/cache.npz") as z:
        c={k:z[k] for k in ("folds","held_s1","strict_nested","val_s1","strict_val")}
    held=c["folds"]==2
    bh=[b for b,k in zip(bank,held) if k]
    hs=np.load(ROOT/"experiments/round227/scores.npz")["held"]
    vs=np.load(ROOT/"experiments/round228/validation_227.npz")["scores"]
    oldh=strict_choice(bh,c["held_s1"],c["strict_nested"][held])
    val=vy=oldv=None
    records=[]
    for mode in ("raw","row_normalized"):
        addh=normalized(hs) if mode=="row_normalized" else hs-hs.mean(1,keepdims=True)
        addv=normalized(vs) if mode=="row_normalized" else vs-vs.mean(1,keepdims=True)
        for alpha in ALPHAS:
            newh=strict_choice(bh,c["held_s1"]+alpha*addh,c["strict_nested"][held])
            rec=dict(mode=mode,alpha=alpha,held=record(y[held],newh,oldh))
            if rec["held"]["correct"]>=1477:
                if val is None:
                    val,_,vy,_=load("validation")
                    oldv=strict_choice(val,c["val_s1"],c["strict_val"])
                newv=strict_choice(val,c["val_s1"]+alpha*addv,c["strict_val"])
                rec["validation"]=record(vy,newv,oldv)
            records.append(rec)
            print("CONFIG",rec,flush=True)
    eligible=[r for r in records if r.get("validation",{}).get("correct",0)>760]
    chosen=max(eligible,key=lambda r:(r["validation"]["correct"],r["held"]["correct"],-r["alpha"])) if eligible else None
    dump(OUT/"result.json",dict(records=records,selected=chosen,test_opened=False))
    print("SELECTED",chosen,flush=True)


if __name__=="__main__":
    main()
