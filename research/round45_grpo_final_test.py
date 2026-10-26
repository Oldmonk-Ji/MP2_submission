"""Requested fixed TEST comparison of an already trained GRPO checkpoint.

Select the GRPO arm by previously frozen validation results, then evaluate
without fitting, tuning, inspecting test errors, or changing the incumbent.
"""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'DASE7506-MP2-main/code'
BASE=ROOT/'experiments/final_evaluation_20261021'
OUT=ROOT/'experiments/round45_grpo_final_test_20261021'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    assert not OUT.exists(),'Do not silently repeat a completed TEST comparison.'
    verified={}
    for manifest in [ROOT/'experiments/round45/completion_manifest.json',BASE/'completion_manifest.json']:
        for name,h in read(manifest)['hashes'].items():
            if name not in verified:verified[name]=sha(ROOT/name)
            assert verified[name]==h,name
    records=read(ROOT/'experiments/round45/selection.json')['records']
    eligible=[r for r in records if r.get('arm') in ('grpo','grpo_relations') and r['asset_eligible']]
    chosen=max(eligible,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes_upper_bound']))
    assert chosen['name']=='candidate_grpo_fixed_members' and chosen['correct']==750
    OUT.mkdir();bundle=OUT/'bundle';methoddir=OUT/'method'
    shutil.copytree(BASE/'bundle',bundle)
    shutil.copytree(BASE/'method',methoddir,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    model_sources=[]
    for j,seed in zip((4,5),(25025,25026)):
        source=ROOT/f'experiments/round45/full/candidate_grpo_s{seed}'
        cfg=read(source/'config.json');assert cfg['arm']=='grpo' and cfg['training_n']==4743 and cfg['seed']==seed
        for f in ('config.json','model.npz'):
            shutil.copyfile(source/f,bundle/f'member{j}'/f);model_sources.append(source/f)
    cfg=read(bundle/'config.json');cfg.update(level='candidate',arm='grpo');dump(bundle/'config.json',cfg)
    assets=[p for p in bundle.rglob('*') if p.is_file()];size=sum(p.stat().st_size for p in assets);assert size<=16*1024**2
    frozen=assets+list(methoddir.glob('*.py'))+model_sources+[CODE/'predict.py',CODE/'score_predictions.py',CODE/'common.py',Path(__file__).resolve()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(frozen)}
    method=methoddir/'round25_selector.py'
    commands={split:[[str(CODE/'predict.py'),'--method',str(method),'--checkpoint',str(bundle),'--split',split,'--output',str(OUT/f'{split}_predictions.json')],
        [str(CODE/'score_predictions.py'),'--split',split,'--predictions',str(OUT/f'{split}_predictions.json'),'--output',str(OUT/f'{split}_score.json')]] for split in ('validation','test')}
    dump(OUT/'freeze_before_evaluation.json',dict(created_utc=now(),user_request='rlvr后的模型尝试一测试集',selected=chosen,
        selection_basis='Previously frozen full validation results among GRPO arms; no test-based checkpoint selection.',
        baseline_test_previously_known=True,checkpoint_trained_and_frozen_before_baseline_test=True,
        training_or_parameters_changed=False,incumbent_not_changed=True,comparison_only=True,
        prior_files_verified=len(verified),asset_bytes=size,hashes=hashes,commands=commands))
    print('FROZEN GRPO',dict(name=chosen['name'],validation_correct=750,asset_bytes=size),flush=True)
    env=dict(os.environ);env['PYTHONPATH']=str(methoddir)+os.pathsep+str(ROOT/'.venv/Lib/site-packages');env['PYTHONIOENCODING']='utf-8'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='4'
    def run(args,label):
        tick=time.perf_counter();p=subprocess.run([sys.executable,'-X','utf8']+args,env=env,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        elapsed=time.perf_counter()-tick
        (OUT/f'{label}_stdout.txt').write_text(p.stdout,encoding='utf-8');(OUT/f'{label}_stderr.txt').write_text(p.stderr,encoding='utf-8')
        assert p.returncode==0,(p.returncode,p.stdout,p.stderr)
        print(p.stdout.strip(),flush=True);return elapsed
    for j,args in enumerate(commands['validation']):run(args,f'validation_process{j}')
    vr=read(OUT/'validation_score.json');assert vr['correct']==750 and vr['n']==798
    import numpy as np
    with np.load(ROOT/'experiments/round45/validation_choices.npz') as z:assert np.array_equal(z[chosen['name']],read(OUT/'validation_predictions.json')['selected'])
    dump(OUT/'validation_equivalence.json',dict(n=798,correct=750,all_choices_match_frozen_cache=True,created_utc=now()))
    for name,h in hashes.items():assert sha(ROOT/name)==h,name
    times=[run(args,f'test_process{j}') for j,args in enumerate(commands['test'])]
    result=read(OUT/'test_score.json');assert result['n']==467 and result['split']=='test'
    baseline=read(BASE/'test_score.json');assert baseline['n']==467 and baseline['correct']==386
    for name,h in hashes.items():assert sha(ROOT/name)==h,name
    summary=dict(model=chosen['name'],validation_correct=750,validation_n=798,test_correct=result['correct'],test_n=467,test_accuracy=result['accuracy'],
        baseline_test_correct=baseline['correct'],baseline_test_accuracy=baseline['accuracy'],correct_difference=result['correct']-baseline['correct'],
        accuracy_percentage_point_difference=result['accuracy']-baseline['accuracy'],asset_bytes=size,asset_MiB=size/1024**2,
        prediction_process_seconds=times[0],scoring_process_seconds=times[1],total_fresh_process_seconds=sum(times),
        local_cpu='Intel Core i7-11800H',threads=4,local_under30=sum(times)<=30,reference_machine_timing_certified=False,
        frozen_hashes_unchanged=True,test_used_for_tuning_or_selection=False,completed_test_evaluations_for_this_model=1,
        goal_97_achieved=result['accuracy']>=97.,incumbent_changed=False,leaderboard_submitted=False,completed_utc=now())
    dump(OUT/'summary.json',summary)
    diff=summary['correct_difference'];pdiff=summary['accuracy_percentage_point_difference'];status='达到' if summary['goal_97_achieved'] else '未达到'
    report=f'''RLVR/GRPO冻结模型测试对照，2026-10-21

无RLVR第25轮：386/467 = {baseline['accuracy']:.6f}%。
RLVR/GRPO后：{result['correct']}/467 = {result['accuracy']:.6f}%。
变化：{diff:+d}题，{pdiff:+.6f}个百分点。{status}97%目标。

测试对象：第45轮candidate_grpo_fixed_members。两个ground成员用在线GRPO微调，其余成员及融合权重保留；种子25025/25026，每个最终模型使用4743道批准TRAIN题。它是已有GRPO方案中验证集最好的一个，750/798（93.984962%）；无RLVR模型验证752/798。
本次按用户明确请求作固定模型测试对照。基线测试分数已知；GRPO权重在基线测试之前已经训练并冻结，此次不重训、不调参、不按测试分数选checkpoint或替换现有推荐模型。未读取测试错题进行规则修补。
独立打包后先复现完整798题验证预测，与第45轮冻结缓存逐题一致，再用原版predict.py和score_predictions.py测试完整467题，仅一次成功测试评分。既有模型、源码和新包在测试前后哈希一致。未提交排行榜。

资产：{size:,}字节，{size/1024**2:.6f}MiB。本机Intel Core i7-11800H，4线程：新预测进程{times[0]:.3f}s，独立评分{times[1]:.3f}s，合计{sum(times):.3f}s。参考Xeon Platinum8457C的时限资格未认证。
官方成绩：experiments/round45_grpo_final_test_20261021/test_score.json
统计对照：experiments/round45_grpo_final_test_20261021/summary.json
测试前冻结：experiments/round45_grpo_final_test_20261021/freeze_before_evaluation.json
冻结推理源码：experiments/round45_grpo_final_test_20261021/method/round25_selector.py（只负责推理；GRPO权重在bundle/member4和member5）
模型资产：experiments/round45_grpo_final_test_20261021/bundle
训练实现：research/round45_rlvr.py；完整消融：RLVR在线采样消融_2026-10-21_第45轮.txt。
此记录单独补充此前仅基线最终测试的报告，不覆盖历史记录。
'''
    path=ROOT/'RLVR模型测试对照_2026-10-21.txt';path.write_text(report,encoding='utf-8')
    files=[p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[path,Path(__file__).resolve()]
    dump(OUT/'completion_manifest.json',dict(completed=True,test_used_for_tuning_or_selection=False,goal_97_achieved=summary['goal_97_achieved'],
        hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}))
    print('GRPO TEST COMPARISON',summary,flush=True)

if __name__=='__main__':main()
