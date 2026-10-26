"""User-authorized final TEST evaluation, after validation-only selection freezes.

Deliberately separate from data_access: research keeps its TEST access guard.
This entrypoint never feeds test outcomes back into model selection or fitting.
"""
from pathlib import Path
import ast,datetime,hashlib,json,os,shutil,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'DASE7506-MP2-main/code';OUT=ROOT/'experiments/final_evaluation_20261021'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    assert not OUT.exists(),'Final evaluation already initialized; do not silently re-run or select using TEST.'
    selection=read(ROOT/'experiments/round45/selection.json')
    assert selection['selected']['name']=='control' and selection['selected']['correct']==752
    assert len(read(ROOT/'experiments/round45/verification.json'))==2
    # Verify all prior experiment freezes before accessing any TEST input.
    manifests=[ROOT/f'experiments/round{i}/completion_manifest.json' for i in range(22,38)]
    manifests += [ROOT/'experiments/round38_42_completion_manifest.json',ROOT/'experiments/round43_44_completion_manifest.json',ROOT/'experiments/round45/completion_manifest.json']
    checked={}
    for manifest in manifests:
        for p,h in read(manifest)['hashes'].items():
            if p not in checked:checked[p]=sha(ROOT/p)
            assert checked[p]==h,p
    OUT.mkdir();bundle=OUT/'bundle';shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle)
    methoddir=OUT/'method';methoddir.mkdir();queue=[ROOT/'research/round25_selector.py',CODE/'predict.py',CODE/'score_predictions.py'];sources=set()
    while queue:
        p=queue.pop()
        if p in sources:continue
        sources.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                for directory in (ROOT/'research',CODE):
                    q=directory/f'{name}.py'
                    if q.exists():queue.append(q);break
    for p in sources:
        if p.parent==ROOT/'research':shutil.copyfile(p,methoddir/p.name)
    assets=[p for p in bundle.rglob('*') if p.is_file()];assetbytes=sum(p.stat().st_size for p in assets);assert assetbytes==14358875 and assetbytes<=16*1024**2
    frozen=assets+list(methoddir.glob('*.py'))+[p for p in sources if p.parent==CODE]+[Path(__file__).resolve()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(frozen)}
    method=methoddir/'round25_selector.py';pred=OUT/'test_predictions.json';score=OUT/'test_score.json'
    commands=[[str(CODE/'predict.py'),'--method',str(method),'--checkpoint',str(bundle),'--split','test','--output',str(pred)],
              [str(CODE/'score_predictions.py'),'--split','test','--predictions',str(pred),'--output',str(score)]]
    dump(OUT/'freeze_before_test.json',dict(frozen_utc=now(),user_authorization='最后一定用测试集试一下效果',
        selection_basis='Full official validation only; round45 selected unchanged round25 control,752/798.',
        selected_method='round25',validation_correct=752,validation_n=798,expected_test_n=467,
        test_use='One final evaluation of already frozen method. No test-driven tuning, model comparison, per-error rule edits or score-based selection.',
        asset_bytes=assetbytes,source_files=len(sources),prior_files_verified=len(checked),hashes=hashes,commands=commands,
        local_cpu='Intel Core i7-11800H',threads=4,test_access_started=False))
    print('FROZEN BEFORE TEST',dict(method='round25',validation_correct=752,asset_bytes=assetbytes,prior_files_verified=len(checked)),flush=True)
    env=dict(os.environ);env['PYTHONPATH']=str(methoddir)+os.pathsep+str(ROOT/'.venv/Lib/site-packages');env['PYTHONIOENCODING']='utf-8'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='4'
    timing=[];started=now()
    for j,args in enumerate(commands):
        tick=time.perf_counter();p=subprocess.run([sys.executable,'-X','utf8']+args,env=env,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        timing.append(time.perf_counter()-tick)
        (OUT/f'process{j}_stdout.txt').write_text(p.stdout,encoding='utf-8');(OUT/f'process{j}_stderr.txt').write_text(p.stderr,encoding='utf-8')
        assert p.returncode==0,(p.returncode,p.stdout,p.stderr)
        print(p.stdout.strip(),flush=True)
    result=read(score);assert result['split']=='test' and result['n']==467
    for name,h in hashes.items():assert sha(ROOT/name)==h,name
    summary=dict(method='round25',test_correct=result['correct'],test_n=result['n'],test_accuracy=result['accuracy'],
        goal_97_achieved=result['accuracy']>=97.,validation_correct=752,validation_n=798,asset_bytes=assetbytes,asset_MiB=assetbytes/1024**2,
        prediction_process_seconds=timing[0],scoring_process_seconds=timing[1],total_fresh_process_seconds=sum(timing),
        local_cpu='Intel Core i7-11800H',threads=4,local_under30=sum(timing)<=30,reference_machine_eligibility_not_certified=True,
        started_utc=started,completed_utc=now(),frozen_hashes_unchanged=True,test_used_for_tuning=False,test_evaluations_this_stage=1)
    dump(OUT/'summary.json',summary)
    status='达到97%目标' if summary['goal_97_achieved'] else '未达到97%目标'
    report=f'''最终冻结模型测试评估，2026-10-21

用户授权：最后一定用测试集试一下效果。
先按完整官方验证集选择并冻结模型，之后才打开测试集。最终采用第25轮模型；第45轮RLVR/GRPO均未超过其752/798（94.2356%）验证成绩。没有根据测试结果选择任何模型或修改规则。

完整官方测试集：{result['correct']}/{result['n']} = {result['accuracy']:.6f}%。{status}。
使用原版predict.py和score_predictions.py，分母467，无删题或拒答，无外部推理调用。只执行一次最终模型测试，不比较各实验的测试成绩，不分析测试错题以继续调参。此结果是本地官方评分，未提交排行榜。

资产：{assetbytes:,}字节，{assetbytes/1024**2:.6f} MiB。
本机Intel Core i7-11800H，4线程：新预测进程{timing[0]:.3f}秒，独立评分进程{timing[1]:.3f}秒，总计{sum(timing):.3f}秒。本机是否低于30秒：{sum(timing)<=30}。
参考Xeon Platinum8457C决定正式时限资格；本机测试不等于参考机器认证。

冻结证据：experiments/final_evaluation_20261021/freeze_before_test.json，包含测试前时间、按验证选模依据、源代码/资产SHA256及命令。测试前后哈希一致。
预测：experiments/final_evaluation_20261021/test_predictions.json
官方成绩：experiments/final_evaluation_20261021/test_score.json
统计：experiments/final_evaluation_20261021/summary.json
冻结资产：experiments/final_evaluation_20261021/bundle
冻结推理源码：experiments/final_evaluation_20261021/method/round25_selector.py
RLVR完整消融另见RLVR在线采样消融_2026-10-21_第45轮.txt；其中“未读取测试”描述的是测试前的实验阶段，后续最终测试单独记录在本文件。
研究入口data_access.py的测试访问防护保持原样；本最终评估由独立脚本research/final_evaluation_20261021.py执行。
'''
    reportpath=ROOT/'最终测试评估_2026-10-21.txt';reportpath.write_text(report,encoding='utf-8')
    files=[p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[reportpath]
    dump(OUT/'completion_manifest.json',dict(completed=True,goal_97_achieved=summary['goal_97_achieved'],test_used_for_tuning=False,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}))
    print('FINAL TEST',summary,flush=True)

if __name__=='__main__':main()
