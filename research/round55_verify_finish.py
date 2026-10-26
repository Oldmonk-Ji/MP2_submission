"""Validate experiment lineage before TEST, summarize after frozen evaluation."""
from pathlib import Path
import json,hashlib,sys,datetime
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/round55'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

if __name__=='__main__':
    checked={}
    for manifest in (ROOT/'experiments/round51_54_completion_manifest.json',OUT/'prepared_freeze.json',OUT/'warm_freeze.json'):
        for p,h in read(manifest)['hashes'].items():
            if p not in checked:checked[p]=sha(ROOT/p)
            assert checked[p]==h,p
    for p,h in read(OUT/'plan.json')['source_hashes'].items():assert sha(ROOT/p)==h,p
    with np.load(OUT/'train_targets.npz') as z:ids=z['ids'];folds=z['folds']
    heldparents={p['train_index'] for p in read(OUT/'held_source_review_before_reference.json')}
    parents=np.load(OUT/'constraint_parent_ids.npy')
    for stage in ('diagnostic','full'):
        expected=ids[folds!=2].tolist() if stage=='diagnostic' else ids.tolist()
        for path in (OUT/stage).glob('*/training.json'):
            data=read(path);assert data['training_ids']==expected,path
            assert not set(data.get('constraint_parents',[]))&heldparents,path
        for seed in (48048,48049):
            sample=read(OUT/f'{stage}_sampling_s{seed}.json');used=np.flatnonzero(sample['pair_draws'])
            assert set(used)<=set(sample['eligible_pairs'])
            assert set(parents[used])<=set(expected)
    if len(sys.argv)>1 and sys.argv[1]=='--before-test':
        dump(OUT/'pretest_verification.json',dict(checked_files=len(checked),prior_history_unchanged=True,all_training_ids_verified=True,held_process_excluded=True,test_opened=False))
        print('PRETEST VERIFIED',len(checked));sys.exit()
    testdir=ROOT/'experiments/feature_validation_20261025';result=read(testdir/'summary.json');val=read(OUT/'validation_result.json')[0];relation=read(OUT/'held_relation_results.json')
    diag=next(x for x in read(OUT/'diagnostic_result.json') if x['mode']=='fixed_residual');cost=read(OUT/'training_complete.json')
    control=read(ROOT/'experiments/round52/validation_result.json')[0]
    report=f'''论文启发的父题均衡难关系对消融，第55轮，2026-10-25

论文清单与方法建议：相关论文与优化路线_2026-10-25.txt。
本轮是Textual Enhanced Contrastive Learning思想的有限适配，不是原论文复现。未引入其数据、模型、外部标注或新候选生成。

固定对照：第52轮SFT。相同程序库、2048维关系特征、两固定种子、冻结20轮预热、10轮继续SFT、每批两次更新、KL0.02和关系权重0.2。唯一实验处理为“半批随机关系对+半批按父题均衡的最难已审查关系对”，难度每轮更新。它同时改变父题分布和难例分布，因此不能把收益单独归因于其中一个因素。未扫描比例或超参数。

结果：
训练历史留出最终候选：对照1466/1582，新方法{diag['correct']}/1582，修复{diag['repairs']}、误伤{diag['harms']}。
24父题209关系诊断：对照151/209，新方法{relation['correct']}/209；父题等权平均{100*relation['parent_mean_accuracy']:.2f}%。
关系分类：{relation['by_kind']}。
完整官方验证：对照{control['correct']}/798，新方法{val['correct']}/798（{val['accuracy']:.6f}%）；原模型752/798。
按验证选模推荐：{result['selected_for_submission']}。本轮新方法最终冻结测试：{result['test_correct']}/467（{result['test_accuracy']:.6f}%）；历史原模型386/467。97%是否达到：{result['goal_97_achieved']}。
测试仅在训练、验证和方法冻结后执行一次。未检查测试错题，测试不用于调参、选模或改变推荐。

解释：需要同时看关系留出与候选选择，不能用拟合奖励或局部关系分数替代最终效果。现有706对监督仅来自86父题，难例重采样没有增加跨对象、跨状态的真实关系多样性。更深入的MathWorld式对象—状态图仍是后续路线，不把本轮采样实验称作已完成世界模型。

训练使用全部4743道允许TRAIN；诊断仅3161题，held过程标签不参与挖掘/梯度；原12题排除不变。历史产物、预热权重和所有样本ID核验通过。
本轮继续训练耗时{cost['seconds']:.2f}秒，复用预热另有历史训练成本；{cost['device']}，PyTorch{cost['torch']}。
推理资产{result['asset_MiB']:.6f}MiB；本地i7-11800H四线程完整TEST新进程预测+评分{result['test_total_fresh_seconds']:.3f}秒；参考Xeon资格未认证。未提交排行榜。

主要文件：research/round55_hard_relations.py、round55_evaluate.py；experiments/round55/plan.json、diagnostic_result.json、held_relation_results.json、validation_result.json。
标准最终验证：experiments/feature_validation_20261025/freeze_before_evaluation.json、interface_checks.json、test_score.json、summary.json。
'''
    path=ROOT/'论文启发的难关系监督消融_2026-10-25_第55轮.txt';path.write_text(report,encoding='utf-8')
    dump(OUT/'summary.json',dict(validation=val,held_relations=relation,diagnostic=diag,final_test=result,test_used_for_tuning=False,goal_97_achieved=result['goal_97_achieved']))
    files=[path,ROOT/'相关论文与优化路线_2026-10-25.txt']+list((ROOT/'research').glob('round55*.py'))
    for d in (OUT,testdir):files += [p for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='completion_manifest.json']
    manifest=dict(completed=True,test_used_for_tuning=False,hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))});dump(OUT/'completion_manifest.json',manifest)
    for p,h in manifest['hashes'].items():assert sha(ROOT/p)==h,p
    print(dict(report=str(path),validation=val['correct'],test=result['test_correct'],relation=relation['correct']),flush=True)
