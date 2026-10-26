"""Check completed TRAIN/VAL work, then select one new method before TEST."""
from round22_common import ROOT,read,dump,sha
from pathlib import Path

if __name__=='__main__':
    target=ROOT/'experiments/round77_79_selection.json';assert not target.exists();checked={};manifests=[]
    for name in ['round46_50_completion_manifest.json','round51_54_completion_manifest.json']+[f'round{r}/completion_manifest.json' for r in range(55,76)]+['round64_corrected/completion_manifest.json']:
        p=ROOT/'experiments'/name
        if not p.exists():continue
        manifests.append(name)
        for source,h in read(p).get('hashes',{}).items():
            if source not in checked:checked[source]=sha(ROOT/source)
            assert checked[source]==h,source
    eligible=set(read(ROOT/'experiments/round22/plan.json')['training_ids']);rows=[]
    for r in (77,78,79):
        path=ROOT/f'experiments/round{r}';complete=read(path/'training_complete.json');assert not complete['test_opened']
        for plan in [path/'plan.json']+([path/'resume_plan.json'] if r==77 else []):
            for source,h in read(plan)['source_hashes'].items():assert sha(ROOT/source)==h,source
        if r==77:
            for arm in ('positionless','ordered'):
                t=read(path/'full'/arm/'training.json');assert set(t['training_ids'])==eligible;assert set(t['constraint_parents'])<=eligible
        else:assert set(complete['training_ids'])==eligible
        for rec in read(path/'validation_records.json'):
            if r==78 and rec['arm']=='control':continue
            rows.append(dict(round=r,**rec))
        files=[p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        sources=list((ROOT/'research').glob(f'round{r}*.py'))
        dump(path/'completion_manifest.json',dict(completed=True,test_used_for_tuning=False,hashes={str(p.relative_to(ROOT)):sha(p) for p in files+sources}))
    best=max(rows,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']));r=best['round'];bundle=f"experiments/round{r}/bundle_{best['arm']}"
    selection=dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=bundle),selected=f'round{r}' if best['correct']>753 else 'round75',test_target_fixed_before_test=True,test_used_for_selection=False,method=f'research/round{r}_selector.py',all_new_validation_results=rows,prior_freezes_checked=len(checked),prior_manifests=manifests,goal86_required_correct=402)
    dump(target,selection);print(selection,flush=True)
