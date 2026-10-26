"""Preserve the quadratic expert's confidence; select all settings on FIT."""
from round22_common import ROOT, load, read, dump, sha
from round224_target_conclusion import strict_choice
from round225_train import record
from threadpoolctl import threadpool_limits
from functools import lru_cache
import round4_gates
import numpy as np

OUT = ROOT / "experiments/round230"


def choices(bank, base_scores, strict, expert, mode, value):
    old = strict_choice(bank, base_scores, strict)
    if mode == "raw_blend":
        return strict_choice(bank, base_scores + value*(expert-expert.mean(1,keepdims=True)), strict)
    new = expert.argmax(1)
    rows = np.arange(len(bank))
    allowed = (np.argsort(-base_scores,axis=1)[:,:2] == new[:,None]).any(1)
    flip = allowed & (expert[rows,new] - expert[rows,old] > value)
    result = old.copy()
    result[flip] = new[flip]
    return result


def main():
    threadpool_limits(4)
    round4_gates.trace = lru_cache(maxsize=32768)(round4_gates.trace)
    OUT.mkdir(exist_ok=True)
    grids = dict(raw_blend=[0,.1,.25,.5,1.], top2_gap=[.5,1.,2.,3.])
    dump(OUT/"plan.json",dict(
        rationale="Per-row score normalization discards confidence. Test raw magnitude and a top2 confidence override.",
        grids=grids,selection="Within each family choose only by aggregate FIT design accuracy; no HELD/VAL grid.",
        thresholds_selected_from_fit_only=True,test_opened=False,source_hash=sha(__file__)))
    bank,_,y,ids=load("train")
    with np.load(ROOT/"experiments/round216/cache.npz") as z:
        c={k:z[k] for k in ("folds","held_s1","strict_nested","val_s1","strict_val")}
    with np.load(ROOT/"experiments/round227/scores.npz") as z:
        assert np.array_equal(ids,z["ids"])
        design,hs=z["design"],z["held"]
    base=np.load(ROOT/"experiments/round225/fit_only_s1.npz")["scores"]
    fit,held=c["folds"]!=2,c["folds"]==2
    bf=[b for b,k in zip(bank,fit) if k]
    bh=[b for b,k in zip(bank,held) if k]
    oldf=strict_choice(bf,base[fit],c["strict_nested"][fit])
    oldh=strict_choice(bh,c["held_s1"],c["strict_nested"][held])
    design_records=[]
    selected={}
    for mode,grid in grids.items():
        candidates=[]
        for value in grid:
            new=choices(bf,base[fit],c["strict_nested"][fit],design[fit],mode,value)
            r=dict(mode=mode,value=value,**record(y[fit],new,oldf))
            candidates.append(r)
            design_records.append(r)
            print("DESIGN",r,flush=True)
        selected[mode]=max(candidates,key=lambda r:(r["correct"],-r["changed"]))
    dump(OUT/"design_selection.json",dict(records=design_records,selected=selected))
    results={}
    for mode,s in selected.items():
        if s["correct"] <= s["baseline"]:
            results[mode]=dict(selected=s,rejected_on_design=True)
            continue
        newh=choices(bh,c["held_s1"],c["strict_nested"][held],hs,mode,s["value"])
        r=dict(selected=s,held=record(y[held],newh,oldh),validation_evaluated=False,test_opened=False)
        print("HELD",mode,r,flush=True)
        if r["held"]["correct"]>=1477:
            val,_,vy,_=load("validation")
            vs=np.load(ROOT/"experiments/round228/validation_227.npz")["scores"]
            oldv=strict_choice(val,c["val_s1"],c["strict_val"])
            newv=choices(val,c["val_s1"],c["strict_val"],vs,mode,s["value"])
            r.update(validation=record(vy,newv,oldv),validation_evaluated=True)
            np.savez(OUT/f"validation_{mode}.npz",selected=newv)
            print("VALIDATION",mode,r,flush=True)
        results[mode]=r
    dump(OUT/"result.json",results)


if __name__=="__main__":
    main()
