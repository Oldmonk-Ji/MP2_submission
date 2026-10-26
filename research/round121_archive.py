from round22_common import ROOT,read,dump,sha
if __name__=='__main__':
    for n in (107,110,117):
        path=ROOT/f'experiments/round{n}'
        for stage in ['oof0','oof1','oof2','nested0','nested1','full']:
            r=read(path/('latent' if n==107 else '')/f'{stage}.json');assert r['optimizer_success']
    for stage in ['oof0','oof1','oof2','nested0','nested1','full']:
        assert max(read(ROOT/f'experiments/round120/{stage}.json')['iterations'])<700
        assert max(read(ROOT/f'experiments/round112/{stage}.json')['iterations'])<1500
    for n in (113,121):assert read(ROOT/f'experiments/round{n}/feature_checks.json')['test_opened']==False
    table=[]
    for n in (108,111,113,118,121):
        for r in read(ROOT/f'experiments/round{n}/validation_records.json'):
            if r['arm']=='control':assert r['correct']==759;continue
            assert r['correct']<=759;held=read(ROOT/f"experiments/round{n}/{r['arm']}_held.json")
            table.append(dict(round=n,**r,held_correct=held['correct'],held_n=held['n']))
    result=dict(current_recommendation='round104 local',validation_correct=759,validation_n=798,previous_frozen_test_correct=391,test_n=467,goal_required_test_correct=402,goal_complete=False,test_opened_in_this_phase=False,results=table,old_process_labels_have_lineage_defects=True,new_supervision_still_weak_not_human_gold=True,local_cpu_screen_seconds=33,official_reference_cpu_seconds=30,official_asset_bytes=16*1024**2)
    dump(ROOT/'experiments/round107_121_summary.json',result)
    for n in range(107,122):
        path=ROOT/f'experiments/round{n}';manifest=path/'completion_manifest.json';assert not manifest.exists()
        files=[p for p in path.rglob('*') if p.is_file()]+list((ROOT/'research').glob(f'round{n}_*.py'))
        dump(manifest,dict(completed=True,test_used_for_tuning=False,hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}))
    report=ROOT/'解释选择与监督来源审计_2026-10-26_第107至121轮.txt'
    dump(ROOT/'experiments/round107_121_report_manifest.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in [report,ROOT/'experiments/round107_121_summary.json']}))
    dump(ROOT/'experiments/next_after_round121.json',dict(goal_active=True,current_recommendation='round104 local',next_round=122,task='Repair demonstrable entity-identity information loss in structural question features.',evidence='TRAIN-fit question3100: swapping final target Robert to Tim leaves old question_tokens exactly equal despite different intended answer. Preserve stable within-question first-mention IDs, continue anonymizing actual names.',fixed_ablation='Same candidate programs, local structure tags, hash dimension65536, C1 pairwise training, 3OOF+2nested+full, fixed meta200/7. Only change entity representation.',test_opened=False,adoption='Full VAL strictly above759; ties retain104; only freeze a VAL winner for a new final TEST evaluation.'))
    print(result)
