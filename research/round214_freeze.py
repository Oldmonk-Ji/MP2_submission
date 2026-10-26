"""Freeze the strict-gap rule method (VAL 760): bundle, cache predictions, selection json.

The bundle reuses round210/bundle_control (exact104 assets) plus strict.npz
(round210 full contrastive weight). The validation selection is recomputed
through the selector's own prediction path so the frozen cache matches the
feature_validation live run by construction.
"""
from round22_common import ROOT, load, read, dump, sha
from round214_selector import predict
import numpy as np, shutil, time

OUT = ROOT / 'experiments/round214'
ARM = 'strict_gap_t3'
if __name__ == '__main__':
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    bundle = OUT / 'bundle'
    if bundle.exists():
        shutil.rmtree(bundle)
    shutil.copytree(ROOT / 'experiments/round210/bundle_control', bundle)
    shutil.copy2(ROOT / 'experiments/round210/full.npz', bundle / 'strict.npz')
    weights = [m['weight'] for m in read(ROOT / 'experiments/round25/selected_bundle/config.json')['members']]
    dump(bundle / 'config.json', dict(arm='relations', weights=weights, structure_arm='local',
                                      strict_tau=3.0))
    assets = [p for p in bundle.rglob('*') if p.is_file()]
    size = sum(p.stat().st_size for p in assets)
    assert size <= 16 * 1024 ** 2, size
    v, va, vy, _ = load('validation')
    sel = np.asarray(predict(v, va, bundle), int)
    correct = int((vy[np.arange(len(v)), sel] == 1).sum())
    assert correct >= 760, correct
    pred = read(ROOT / 'experiments/round63/validation_sft_predictions.json')
    pred['selected'] = sel.tolist()
    dump(OUT / f'validation_{ARM}_predictions.json', pred)
    dump(OUT / 'final_feature_selection.json', dict(
        best_new=dict(arm=ARM, correct=correct, bundle='experiments/round214/bundle'),
        selected='round214' if correct > 759 else 'round104',
        test_target_fixed_before_test=True, test_used_for_selection=False))
    dump(OUT / 'freeze_plan.json', dict(
        method='exact104 fusion + gate, then strict-gap flip rule (tau=3.0, top-2 restricted, '
               'gate rows untouched). round213_scan design: fit oof design, held flat at 1477, '
               'VAL 760 (+1). Freeze candidate for the standard feature validation.',
        arm=ARM, correct=correct, asset_bytes=size, strict_tau=3.0,
        source_sha256=sha(__file__), test_opened=False))
    print('FREEZE correct', correct, 'asset_bytes', size, 'seconds', time.perf_counter() - tick, flush=True)
