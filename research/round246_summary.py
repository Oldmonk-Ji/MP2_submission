"""Audit and summarize the independent text-model continuation."""
from round22_common import ROOT,read,dump,sha
import numpy as np

def main():
    models=[]
    for source in (241,242,243,245):
        stage=ROOT/f'experiments/round{source}';plan=read(stage/'plan.json');result=read(stage/'result.json')
        for name,expected in plan['source_hashes'].items():assert sha(ROOT/'research'/name)==expected,(source,name)
        assert not result['test_opened']
        folder=stage/'held'
        with np.load(folder/'model.npz') as archive:
            parameters=sum(archive[k].size for k in archive.files)
        models.append(dict(source=source,parameters_including_quantization_scales=parameters,
            standalone_held=result['standalone_held'],standalone_role='correction diagnostic only' if source==243 else 'independent selector',
            best_initial_fused_held=max(r['held']['correct'] for r in result['records']),
            held_asset_bytes=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),
            source_hashes_verified=True))
    adjudication=read(ROOT/'experiments/round244/result.json')
    calibration=read(ROOT/'experiments/round246/result.json')
    assert not adjudication['test_opened'] and not calibration['test_opened']
    best=max(calibration['records'],key=lambda r:(r['held']['correct'],-r['held']['harms'],-r['held']['changed']))
    summary=dict(models=models,original_user_held_floor=1477,previous_extra_conservative_floor=1479,
        floor_amendment='Recorded before any VAL evaluation of the new neural models, using TRAIN-held results only.',
        neural_validation_records=adjudication['records']+calibration['advanced'],
        best_train_held_candidate=best,selected_new=calibration['selected'],
        selected_submission_round=235,baseline_validation=763,baseline_held=1479,
        profiles=calibration['profiles'],test_opened=False,
        conclusion='Raw-text held signal improves, but no new model has validation>763. Preserve frozen235; no TEST authorization requested.')
    assert summary['selected_new'] is None,'Update conclusion/selection before publishing an improved candidate.'
    dump(ROOT/'experiments/round246/continuation_summary.json',summary)
    print(summary,flush=True)

if __name__=='__main__':main()
