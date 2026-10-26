"""Fresh official validation run, resource accounting, and immutable history checks."""
from round22_common import ROOT,CODE,OUT,INCUMBENT,read,dump,sha,plan,load
from round22_selector import predict,component_scores,fused_scores,side_features
from round22_fusion import evaluate
from features import metadata
from data_access import forbid_test
from pathlib import Path
import datetime,json,os,subprocess,sys,time
import numpy as np


def main():
    f=read(OUT/'frozen.json')
    assert all(sha(ROOT/k)==v for k,v in f['hashes'].items())
    p=plan();sel=read(OUT/'selection.json');bundle=OUT/'selected_bundle'
    assert len(p['training_ids'])==4743 and len(p['excluded_training_ids'])==12
    assert set(p['training_ids'])|set(p['excluded_training_ids'])==set(range(4755))
    assert not set(p['training_ids'])&set(p['excluded_training_ids'])
    assert read(OUT/'full/training_ids.json')==p['training_ids']
    seen=[]
    from round14_prepare import family
    train,_,_,tids=load('train');groups={int(i):family(r['question']) for i,r in zip(tids,train)}
    for j in range(3):
        split=read(OUT/f'oof/fold{j}/partition.json');fit=set(split['fit_ids']);held=set(split['held_ids'])
        assert not fit&held and fit|held==set(p['training_ids'])
        assert not {groups[i] for i in fit}&{groups[i] for i in held}
        seen+=split['held_ids']
    assert sorted(seen)==p['training_ids']
    history={}
    for n in (18,19,20):
        old=read(ROOT/f'experiments/round{n}/frozen.json')
        assert all(sha(ROOT/k)==v for k,v in old['hashes'].items())
        history[str(n)]='unchanged'
    old=read(ROOT/'experiments/round21_audit_quality/completion_manifest.json')
    assert all(sha(ROOT/k)==v for k,v in old['hashes'].items());history['21']='unchanged'
    try:forbid_test('open',('gsm8k_test.npz',))
    except PermissionError:pass
    else:raise AssertionError('Test protection not active')
    b,a,y,ids=load('validation');cfg=read(bundle/'readout.json')
    components,extra=component_scores(b[:12],{'embeddings':a['embeddings'][:12]},bundle)
    cues,evidence=side_features(b[:12],extra)
    perm=[2,0,3,1]
    pb=[dict(question=r['question'],id='unrelated-id',candidates=[dict(index=j,text=r['candidates'][k]['text']) for j,k in enumerate(perm)]) for r in b[:12]]
    pc,pe=component_scores(pb,{'embeddings':a['embeddings'][:12,perm]},bundle)
    pq,pev=side_features(pb,pe)
    scores=fused_scores(b[:12],components,cfg,cues,evidence);ps=fused_scores(pb,pc,cfg,pq,pev)
    permutation_error=float(np.max(np.abs(ps-scores[:,perm])))
    assert np.allclose(ps,scores[:,perm],atol=5e-4,rtol=3e-4),permutation_error
    assert predict([],{'embeddings':np.zeros((0,4,1024),np.float32)},bundle)==[]
    model_files=[q for q in bundle.rglob('*') if q.is_file()]
    asset_bytes=sum(q.stat().st_size for q in model_files);assert asset_bytes<=16*1024**2
    assert not any(q.name in ('training_ids.json','eval.npz','eval_components.npz') for q in model_files)
    # One fresh prediction process, then a separate fresh official scoring process.
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
    predictions=OUT/'official_validation_predictions.json';scorefile=OUT/'official_validation_score.json'
    commands=[
        [sys.executable,str(CODE/'predict.py'),'--method',str(ROOT/'research/round22_selector.py'),'--checkpoint',str(bundle),
         '--split','validation','--output',str(predictions)],
        [sys.executable,str(CODE/'score_predictions.py'),'--split','validation','--predictions',str(predictions),'--output',str(scorefile)]]
    timing=[]
    for command in commands:
        tick=time.perf_counter();proc=subprocess.run(command,cwd=str(CODE),env=env,capture_output=True,text=True,encoding='utf-8')
        timing.append(time.perf_counter()-tick)
        assert proc.returncode==0,(proc.returncode,proc.stdout,proc.stderr)
        print(proc.stdout.strip(),flush=True)
    official=read(scorefile);new=np.asarray(read(predictions)['selected']);assert official['n']==798
    with np.load(OUT/'validation_comparison.npz') as z:
        old=z['incumbent_choice'];expected=z[sel['selected']['name']+'_choice']
    assert np.array_equal(new,expected),'Fresh official predictions differ from selection cache'
    assert official['correct']==sel['selected']['correct']
    meta=metadata(b);changed=[];errors=[]
    for i,(row,c,o) in enumerate(zip(b,new,old)):
        good=bool(y[i,c]);before=bool(y[i,o])
        entry=dict(validation_index=i,question=row['question'],selected_index=int(c),previous_index=int(o),
            selected_correct=good,previous_correct=before,answers=meta['answers'][i],
            reference_answer=next(meta['answers'][i][j] for j in range(4) if y[i,j]),
            candidate_texts=[q['text'] for q in row['candidates']])
        if not good:errors.append(entry)
        if c!=o:
            entry['change']='repaired' if good and not before else 'damaged' if before and not good else 'same_correctness'
            changed.append(entry)
    (OUT/'validation_remaining_errors.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in errors),encoding='utf-8')
    (OUT/'validation_changes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in changed),encoding='utf-8')
    training_seconds=read(OUT/'full/done.json')['seconds']
    crossfit_seconds=sum(read(OUT/f'oof/fold{j}/done.json')['seconds'] for j in range(3))
    readout_seconds=sum(r['seconds'] for r in read(OUT/'readouts/frozen.json')['records'])
    import sklearn,scipy,importlib.metadata
    environment=dict(python=sys.version,python_executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
        sklearn=sklearn.__version__,torch_training=importlib.metadata.version('torch'),threads=4,
        final_expert_training_seconds=training_seconds,crossfit_training_seconds=crossfit_seconds,readout_fitting_seconds=readout_seconds,
        training_device=read(OUT/'full/baseline/member3/training.json')['device'],
        training_gpu=read(OUT/'full/baseline/member3/training.json').get('gpu'))
    dump(OUT/'environment.json',environment)
    verification=dict(passed=True,official_validation=official,final_training_n=4743,no_permanent_internal_holdout=True,
        no_evaluation_deletion=True,oof_group_disjoint=True,final_model_contains_no_cached_predictions=True,
        fresh_official_predictions_equal_research_predictions=True,permutation_max_absolute_error=permutation_error,
        inference_asset_bytes=asset_bytes,inference_asset_MiB=asset_bytes/(1024**2),
        local_cpu='Intel Core i7-11800H',threads=4,prediction_fresh_process_seconds=timing[0],scoring_fresh_process_seconds=timing[1],
        local_complete_validation_seconds=sum(timing),
        timing_scope='Full 798-question official validation, separate fresh prediction and scoring processes; not reference-machine 467-question test qualification.',
        test_opened=False,validation_opened=True,historical_rounds=history,training_environment=environment)
    dump(OUT/'verification.json',verification)
    dump(OUT/'asset_manifest.json',{str(q.relative_to(bundle)):dict(bytes=q.stat().st_size,sha256=sha(q)) for q in model_files})
    write_report(sel,verification,errors,changed)
    sources=list((ROOT/'research').glob('round22_*.py'))
    products=[OUT/k for k in ('plan.json','training_exclusion_policy.json','selection.json','frozen.json','verification.json',
        'asset_manifest.json','environment.json','official_validation_score.json','official_validation_predictions.json','validation_changes.jsonl',
        'validation_remaining_errors.jsonl','全量训练与官方验证_第22轮报告.txt')]
    dump(OUT/'completion_manifest.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        completed=True,test_opened=False,validation_opened=True,
        hashes={str(q.relative_to(ROOT)):sha(q) for q in sources+products+model_files}))
    print('VERIFIED',verification,flush=True)


def write_report(sel,v,errors,changed):
    best=sel['selected'];old=sel['incumbent'];o=v['official_validation']
    lines=['第 22 轮：全量训练，完整官方验证', '',
        '已按用户最终更正执行：只在训练时禁用 12 道问题题；任何实际评测都不删题。',
        '官方训练 4755 道，排除已确认的 12 道后，4743 道全部参与每个最终模型的拟合。',
        '排除的 0-based 训练行号：'+', '.join(map(str,plan()['excluded_training_ids'])),
        '官方验证 798 道全部评估和选方案；官方测试 467 道未打开。',
        '之前的 938/950 内部成绩已退出本轮选模目标，不能拿来替代官方验证或提交成绩。',
        '', '正式提交指标',
        '提交官方 score_predictions.py 对完整测试集输出的 accuracy（0 到 100，越高越好）。',
        '每题必须选给定四候选中的一个。最终数值符合官方参考即得分；不要求每一步推理正确。',
        '不允许正式测评删题、拒答或更改数字解析器。测试集至少答对 444/467 才到 95%，463/467 才超过 99%。',
        '资源限制：额外推理资产不超过 16 MiB；参考 Xeon Platinum 8457C 上，全流程 CPU 评估不超过 30 秒，总共四线程。',
        '最终课程评分还包括不超过 10 页的报告、消融、可靠性分析、资源与可复现性，非单看榜单。',
        '官方来源：https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md （2026-10-15 核对）',
        '', '本轮改动',
        '1. 恢复此前因内部划分或不确定性筛选没有参与最终训练的正常题。',
        '2. 按现有五个专家与轻量核比较器的固定配方，重新拟合全部 4743 道。',
        '3. 另用三折、按题目模板分组的交叉预测学习小型融合层。临时折模型不参与部署；最终部署的每个专家都用完整 4743 道训练。',
        '4. 对比线性融合、随题目条件变化的融合、候选概率平均，以及先合并相同数值答案的概率平均。',
        '   融合参数只从官方训练的交叉预测学习；参数强度用完整官方验证选择。无外部数据、预训练权重或推理时 LLM 调用。',
        '5. 推理复用共同的特征提取，减少相同文本被反复解析的开销。',
        '', '完整官方验证结果',
        f"旧冻结方案：{old['correct']}/{old['n']} = {old['accuracy']:.4f}%。",
        f"本轮选定方案 {best['name']}：{o['correct']}/{o['n']} = {o['accuracy']:.4f}%。",
        f"相对旧方案修复 {best['repaired']}，误伤 {best['damaged']}，净增 {best['net']:+d}。",
        f"存在不同数值答案的题：{best['hard_correct']}/{best['hard_n']} = {best['hard_accuracy']:.4f}%；其他题也保留在全量分母中。",
        f"距验证集 95% 所需 759 道：还差 {max(0,759-o['correct'])} 道。",
        '以上是用于选择方案的验证成绩，不是独立测试成绩；所有试验配置都列在下方。',
        '', '资源与复现',
        f"推理资产：{v['inference_asset_bytes']:,} 字节 = {v['inference_asset_MiB']:.4f} MiB。",
        f"训练耗时：最终六专家 {v['training_environment']['final_expert_training_seconds']:.1f} 秒；三折辅助训练 {v['training_environment']['crossfit_training_seconds']:.1f} 秒；融合拟合 {v['training_environment']['readout_fitting_seconds']:.1f} 秒。依赖版本和 GPU 见 environment.json。",
        f"本地 CPU：{v['local_cpu']}，四线程。完整 798 题验证：新进程预测 {v['prediction_fresh_process_seconds']:.3f} 秒 + 新进程评分 {v['scoring_fresh_process_seconds']:.3f} 秒 = {v['local_complete_validation_seconds']:.3f} 秒。",
        '这项时间不是参考机器上的官方测试资格认证，未为测时间打开测试集。',
        '新进程官方预测逐项复现选模时的预测；参数资产大小、候选顺序置换、分组交叉隔离与历史文件哈希检查通过。',
        '原冻结提交保留，新候选独立保存；所有权重和配置位于 selected_bundle。',
        '', '复现命令（工作目录 Project2，Python 3.12 与现有依赖）',
        '训练阶段：python research/round22_common.py；python research/round22_train.py full；python research/round22_train.py oof；python research/round22_fusion.py fit；python research/round22_fusion.py select。',
        '已有检查点无需重训：',
        'python DASE7506-MP2-main/code/predict.py --method research/round22_selector.py --checkpoint experiments/round22/selected_bundle --split validation --output experiments/round22/reproduced_validation_predictions.json',
        'python DASE7506-MP2-main/code/score_predictions.py --split validation --predictions experiments/round22/reproduced_validation_predictions.json --output experiments/round22/reproduced_validation_score.json',
        '运行前将 OMP_NUM_THREADS、OPENBLAS_NUM_THREADS、MKL_NUM_THREADS、NUMEXPR_NUM_THREADS 均设为 4。',
        '数据默认路径由官方 common.py 解析；训练脚本禁止覆盖已冻结结果，复现实验请使用新的输出目录。',
        'AI 披露：本轮代码实现、实验编排与分析由 Codex 辅助；模型从零训练，仅使用课程输入／标签与给定表示。',
        '', '全部验证配置：name | correct/798 | repaired | damaged | net']
    for r in sel['records']:lines.append(f"{r['name']} | {r['correct']}/798 | {r['repaired']} | {r['damaged']} | {r['net']:+d}")
    lines+=['',f'剩余 {len(errors)} 道验证错题完整候选见 validation_remaining_errors.jsonl；变更明细见 validation_changes.jsonl。',
        '后续错误诊断仅帮助提出下一轮训练方法，不将验证题答案作为训练标签。']
    (OUT/'全量训练与官方验证_第22轮报告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
