"""Prepare immutable candidate assets and cached VAL choices; no TEST access."""
from round22_common import ROOT, load, read, dump, sha
from round224_target_conclusion import strict_choice
import numpy as np
import shutil

OUT=ROOT/"experiments/round232"


def main():
    selected=read(ROOT/"experiments/round231/result.json")["selected"]
    assert selected["mode"]=="raw" and selected["alpha"]==.25
    assert selected["held"]["correct"]==1479 and selected["validation"]["correct"]==762
    assert not OUT.exists(), "Do not overwrite a frozen candidate."
    OUT.mkdir()
    bundle=OUT/"bundle"
    shutil.copytree(ROOT/"experiments/round214/bundle",bundle)
    shutil.copy2(ROOT/"experiments/round228/full_227.npz",bundle/"quadratic.npz")
    cfg=read(bundle/"config.json")
    cfg["quadratic_alpha"]=.25
    dump(bundle/"config.json",cfg)
    size=sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    assert size<=16*2**20
    val,_,vy,_=load("validation")
    with np.load(ROOT/"experiments/round216/cache.npz") as z:
        base,strict=z["val_s1"],z["strict_val"]
    aux=np.load(ROOT/"experiments/round228/validation_227.npz")["scores"]
    chosen=strict_choice(val,base+.25*(aux-aux.mean(1,keepdims=True)),strict)
    assert int(vy[np.arange(len(val)),chosen].sum())==762
    pred=read(ROOT/"experiments/round214/frozen/validation_predictions.json")
    pred["selected"]=chosen.tolist()
    pred.pop("prediction_seconds",None)
    dump(OUT/"validation_quadratic_raw025_predictions.json",pred)
    dump(OUT/"final_feature_selection.json",dict(
        best_new=dict(arm="quadratic_raw025",correct=762,bundle="experiments/round232/bundle"),
        selected="round232",test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/"freeze_plan.json",dict(
        method="round214 plus TRAIN-only whitened residual/context quadratic ranker; raw coefficient 0.25",
        held=1479,validation=762,asset_bytes=size,test_opened=False,test_authorized=False,
        source_hashes={n:sha(ROOT/"research"/n) for n in
                       ("round227_quadratic.py","round228_design_selection.py",
                        "round231_validation_selection.py","round232_selector.py",
                        "round232_quadratic_inference.py")}))
    print(dict(asset_bytes=size,asset_MiB=size/2**20,validation=762,held=1479,test_opened=False))


if __name__=="__main__":
    main()
