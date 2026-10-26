"""Repair only a missing frozen source dependency, then finish final scoring."""
from final_evaluation_20261021 import ROOT,CODE,OUT,read,dump,sha,now
from pathlib import Path
import os,shutil,subprocess,sys,time

def main():
    freeze=read(OUT/'freeze_before_test.json');assert not (OUT/'test_score.json').exists() and not (OUT/'test_predictions.json').exists()
    for p,h in freeze['hashes'].items():assert sha(ROOT/p)==h,p
    source=ROOT/'starter/method.py';old=read(ROOT/'experiments/round25/completion_manifest.json')['hashes']['starter\\method.py']
    assert sha(source)==old
    dest=OUT/'method/method.py';assert not dest.exists();shutil.copyfile(source,dest)
    added={str(p.relative_to(ROOT)):sha(p) for p in (source,dest,Path(__file__).resolve())}
    dump(OUT/'packaging_amendment_before_scoring.json',dict(created_utc=now(),reason='Missing starter/method.py from copied source closure caused import failure.',
        model_or_feature_logic_changed=False,original_hashes_unchanged=True,failed_prediction_attempts=1,
        failed_attempt_loaded_test_inputs=True,test_reference_reads_before_amendment=0,completed_test_scores_before_amendment=0,
        dependency_verified_against='round25/completion_manifest.json',added_hashes=added))
    env=dict(os.environ);env['PYTHONPATH']=str(OUT/'method')+os.pathsep+str(ROOT/'.venv/Lib/site-packages');env['PYTHONIOENCODING']='utf-8'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='4'
    def run(args,label):
        tick=time.perf_counter();p=subprocess.run([sys.executable,'-X','utf8']+[str(a) for a in args],env=env,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        elapsed=time.perf_counter()-tick
        (OUT/f'{label}_stdout.txt').write_text(p.stdout,encoding='utf-8');(OUT/f'{label}_stderr.txt').write_text(p.stderr,encoding='utf-8')
        assert p.returncode==0,(p.returncode,p.stdout,p.stderr)
        print(p.stdout.strip(),flush=True);return elapsed
    method=OUT/'method/round25_selector.py';bundle=OUT/'bundle'
    # Full validation equivalence of the source package, before any test scoring.
    vp=OUT/'packaging_validation_predictions.json';vs=OUT/'packaging_validation_score.json'
    run([CODE/'predict.py','--method',method,'--checkpoint',bundle,'--split','validation','--output',vp],'packaging_validation_predict')
    run([CODE/'score_predictions.py','--split','validation','--predictions',vp,'--output',vs],'packaging_validation_score')
    assert read(vs)['correct']==752 and read(vs)['n']==798
    import numpy as np
    with np.load(ROOT/'experiments/round45/validation_choices.npz') as z:assert np.array_equal(read(vp)['selected'],z['control'])
    dump(OUT/'packaging_validation_equivalence.json',dict(created_utc=now(),n=798,correct=752,choices_identical_to_frozen_control=True,test_score_not_yet_read=True))
    hashes=dict(freeze['hashes'],**added)
    for p,h in hashes.items():assert sha(ROOT/p)==h,p
    started=now();times=[]
    for j,args in enumerate(freeze['commands']):times.append(run(args,f'successful_test_process{j}'))
    score=read(OUT/'test_score.json');assert score['split']=='test' and score['n']==467
    for p,h in hashes.items():assert sha(ROOT/p)==h,p
    total=sum(times);size=freeze['asset_bytes'];goal=score['accuracy']>=97.
    result=dict(method='round25',validation_correct=752,validation_n=798,test_correct=score['correct'],test_n=score['n'],test_accuracy=score['accuracy'],
        goal_97_achieved=goal,asset_bytes=size,asset_MiB=size/1024**2,prediction_process_seconds=times[0],scoring_process_seconds=times[1],
        total_fresh_process_seconds=total,local_cpu='Intel Core i7-11800H',threads=4,local_under30=total<=30,reference_machine_eligibility_not_certified=True,
        started_utc=started,completed_utc=now(),frozen_model_and_source_hashes_unchanged=True,packaging_dependency_added_without_modification=True,
        test_used_for_tuning=False,completed_test_evaluations=1,failed_prediction_import_attempts=1,leaderboard_submitted=False)
    dump(OUT/'summary.json',result)
    status='达到97%目标' if goal else '未达到97%目标'
    report=f'''最终冻结模型测试评估，2026-10-21

完整官方测试集：{score['correct']}/{score['n']} = {score['accuracy']:.6f}%。{status}。
采用第25轮冻结模型，按完整官方验证集752/798（94.2356%）选定。第45轮候选GRPO为750/798，关系GRPO固定融合739/798，加入关系约束740/798，均未替换原模型。
用户明确授权“最后一定用测试集试一下效果”。先完成验证选模和冻结，后执行最终测试评分；没有根据测试结果调参、比较模型或分析测试错题来改规则。完整467题，无删题、拒答或外部推理API。结果为本地原版官方评分，未提交排行榜。

过程记录：首次独立打包的预测进程因缺少starter/method.py而导入失败。该进程已加载测试输入，但未生成预测、未运行评分器、未读取测试参考答案。补齐第25轮清单哈希一致的原始依赖；模型权重和特征逻辑没有修改。补齐后先在完整798题验证集复现，与冻结对照逐题一致，再完成1次成功的测试预测和官方评分。原始失败日志和补充冻结记录均保留。

资产：{size:,}字节 = {size/1024**2:.6f} MiB。
本机Intel Core i7-11800H，4线程：新预测进程{times[0]:.3f}秒 + 独立评分进程{times[1]:.3f}秒 = {total:.3f}秒。本机低于30秒：{total<=30}。
作业参考Xeon Platinum8457C决定正式时限资格；本机测试不等价于参考机器认证。

预测：experiments/final_evaluation_20261021/test_predictions.json
官方成绩：experiments/final_evaluation_20261021/test_score.json
统计：experiments/final_evaluation_20261021/summary.json
冻结依据：freeze_before_test.json及packaging_amendment_before_scoring.json；测试前后全部模型和既有源码哈希一致。
独立源码：experiments/final_evaluation_20261021/method/round25_selector.py
资产：experiments/final_evaluation_20261021/bundle
命令（使用D:/anaconda3/envs/mp1/python.exe，PYTHONPATH含上述method及项目.venv/Lib/site-packages）：
python DASE7506-MP2-main/code/predict.py --method experiments/final_evaluation_20261021/method/round25_selector.py --checkpoint experiments/final_evaluation_20261021/bundle --split test --output experiments/final_evaluation_20261021/test_predictions.json
python DASE7506-MP2-main/code/score_predictions.py --split test --predictions experiments/final_evaluation_20261021/test_predictions.json --output experiments/final_evaluation_20261021/test_score.json
所有CPU库线程数设置4；研究入口data_access.py的测试防护保持原样。
RLVR消融见RLVR在线采样消融_2026-10-21_第45轮.txt；其中“未读测试”描述的是此前研究阶段。本次最终测试单独记录，不回流训练选模。
'''
    path=ROOT/'最终测试评估_2026-10-21.txt';path.write_text(report,encoding='utf-8')
    files=[p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[path,Path(__file__).resolve()]
    dump(OUT/'completion_manifest.json',dict(completed=True,goal_97_achieved=goal,test_used_for_tuning=False,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}))
    print('FINAL TEST',result,flush=True)

if __name__=='__main__':main()
