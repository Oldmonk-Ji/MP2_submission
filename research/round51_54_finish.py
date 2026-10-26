"""Summarize frozen feature experiments and final TEST report without tuning."""
from pathlib import Path
import json,hashlib,datetime

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    selection=read(ROOT/'experiments/round54/final_feature_selection.json');final=read(ROOT/'experiments/feature_validation_20261022/summary.json')
    coverage1=read(ROOT/'experiments/round51/preparation_result.json');coverage2=read(ROOT/'experiments/round54/preparation_result.json')
    diagnostics=read(ROOT/'experiments/round52/diagnostics.json');original38=read(ROOT/'experiments/round54/original38_audit.json')
    relations=read(ROOT/'experiments/round52/held_relation_results.json');lexical=read(ROOT/'experiments/round53/held_relation_results.json')
    training={str(i):read(ROOT/f'experiments/round{i}/training_complete.json') for i in (52,53)}
    new=selection['best_new'];rows=selection['records'];cov={r['group']:r for r in coverage2['records']}
    summary=dict(completed=True,goal_97_achieved=final['goal_97_achieved'],training_and_validation_test_access=False,
        final_test_evaluated=True,test_used_for_tuning_or_selection=False,selection=selection,final_test=final,
        coverage51=coverage1,coverage54=coverage2,original38=original38,relation_diagnostics=relations,
        lexical_relation_diagnostics=lexical,training_cost=training,diagnostics=diagnostics)
    dump(ROOT/'experiments/round54/summary.json',summary)
    table='\n'.join(f"{r['arm']}: {r['correct']}/798 = {100*r['correct']/798:.4f}%；冲突题{r['conflict_correct']}/355；修复{r['repairs']}、误伤{r['harms']}。" for r in rows)
    rel='\n'.join(f"{r['arm']}: {r['correct']}/{r['n']}；按父题等权平均{100*r['parent_mean_accuracy']:.2f}%；分类型{r['by_kind']}。" for r in relations['records'])
    policies='\n'.join(str(r) for r in diagnostics['policy'])
    report=f'''程序覆盖与关系泛化优化、标准feature最终测试流程（第51—54轮）
日期：2026-10-22

结论
本轮完成两次拟合侧程序规则修复、关系抽象特征与匹配SFT消融、精确奖励/GRPO消融、全量验证及冻结新方法的全量测试。97%目标是否达到：{final['goal_97_achieved']}。推荐模型按验证结果确定为{selection['selected']}；最终测试不改变该选择。
新方法测试对象：{new['arm']}，完整验证{new['correct']}/798，完整测试{final['test_correct']}/{final['test_n']}={final['test_accuracy']:.6f}%。历史原模型测试{final['historical_baseline_correct']}/467={final['historical_baseline_accuracy']:.6f}%；差{final['correct_difference']:+d}题。这是最终结果报告，不是根据测试选择模型。未提交排行榜。

1. 规则从哪里来
沿用固定历史TRAIN划分：folds0+1拟合，fold2诊断。第一阶段从拟合侧73道未覆盖错误按round51固定哈希排序读30题，并在73题检查规则；随后将规则和关系特征源码哈希冻结，才评估完整TRAIN的覆盖。第二阶段只读取拟合侧仍缺程序的20题，制定额外解析修复，再单独冻结并在留出部分验证。没有读取留出错误题目来写规则，没有读取测试错题，也没有新增数据集、预训练权重或训练删除。
新增类型包括：带单位计算链、单变量线性方程、解出变量后继续求目标量、上下文支持的时间换算、整数floor/ceil及严格边界、缺少一步的源数量运算，以及由已有中间量构成的最后求和/求差。修复了Markdown加粗残留被误作乘号。只能执行有界AST及Fraction运算，不执行候选任意代码，不改变官方解析器或评分器。
第51轮保持旧程序并追加到64上限；第54轮先给新连接程序最多16个位置，再补旧程序，总量仍64，因此同时记录覆盖损失，不隐瞒旧程序被挤出。

2. 程序覆盖：数值正确不等于关系正确
第50轮3987/4743 → 第51轮4588/4743 → 第54轮{cov['all']['correct_program']}/4743。
第54轮相对51轮，全体新增{cov['all']['gained']}、丢失{cov['all']['lost']}题覆盖。
109道历史训练留出错题：71 → 93 → {cov['held_baseline_errors']['correct_program']}道。最后阶段新增{cov['held_baseline_errors']['gained']}、丢失{cov['held_baseline_errors']['lost']}道。
针对用户指定的原38道缺程序错题：第51轮补回22道，最终补回{original38['recovered']}道，仍缺{original38['still_missing']}道；逐题状态在original38_audit.json。没有宣称全部补齐。
这里的覆盖定义是“存在能执行出训练正确候选数值的程序”，不是独立语义金标。补全隐式中间量可能产生数值巧合；源文本数量相同也可能绑定错对象。第51轮全体{diagnostics['leaf_only_numeric_coverage']}题仅有单个源数量叶子覆盖而没有非叶程序；原38题中非叶程序补回{diagnostics['recovered_nonleaf_from_initial38']}题。应将这些区别保留在报告里，不能把覆盖百分比当准确率或完整语义证明。

3. 关系泛化路线与消融
原1024维局部词法特征之外增加1024维关系抽象：对象/owner是否相同、数量是否对应问题目标、时间/金额/计数类别、rate的单位是否能相消、前后状态、是否对子集或剩余量运算、源数量重复使用等。特征不存题目ID，不用答案表。保留既有86父题706对源文本审查关系监督，包括10对同数值错对象关系；本轮未把自动补全程序转成新的过程金标。
第52轮为增强特征的SFT/exact/GRPO；第53轮仅移除新增1024维抽象特征，保留相同程序、训练题、种子48048/48049、20轮SFT预热、10轮继续训练、每批两更新和优化器。该对照可用于判断新增特征，而“仅覆盖修复、复用第50轮权重”只隔离推理解析变化，不能冒充重新训练的匹配对照。
学习率0.0003，batch32，weight_decay0.01，KL0.02，关系约束0.2；GRPO组16、clip0.2。所有最终模型使用4743道允许TRAIN，诊断模型仅fit3161题。24个新增关系留出父题的过程标签始终不进入训练；其普通TRAIN答案仍可用于最终全训练。仍是有限程序策略学习，不是外部语言模型微调。
第54轮只更换程序解析，复用在此前完整验证上选出的最佳新策略权重，不再训练、不扫门槛或融合系数。完整答案覆盖不足则回退原模型，固定残差0.25。

4. 官方完整验证结果
原模型752/798=94.2356%；冲突题309/355。
{table}
所有候选均在完整798题评分，无删除或拒答。feature_selection在TEST前冻结；测试对象是新方法中验证最好的一个，即使未超过原模型，也按用户要求做固定新方法测试对照。

5. 关系诊断与剩余瓶颈
24个父题209对关系诊断：
{rel}
相同程序和预算的词法SFT：{lexical['correct']}/{lexical['n']}。
原第50轮SFT/exact/GRPO为150/153/152；对照变化同时受新程序训练分布影响，不能仅凭跨轮变化归因到抽象特征。直接特征归因应比较第52轮SFT与第53轮SFT。标签由助手依源文本审查，不是独立人工金标；历史TRAIN也已被探索，结果不是全新盲测。
正确程序存在时策略是否会选：
{policies}
精确奖励上涨、训练关系拟合接近满分都不等于关系泛化。局部对象绑定、时间与分数作用范围仍需更强的源证据约束；当前集合大量同数值替代绑定，终局奖励无法区分。不能因为新程序库覆盖接近97%就声称模型正确率达到97%。
额外仅TRAIN诊断将同答案程序概率求和改为求平均，三个模型仅变化0—1题，未采用，也未另跑验证/测试搜索。

6. 标准feature验证流程已实现并执行
入口：research/feature_validation.py。流程为：
（1）核验历史和新训练冻结哈希；按完整验证选定新方法；
（2）独立打包推理依赖和资产，明确纳入动态导入的starter/method.py；
（3）冻结源码、权重、配置、选模记录和官方评估命令；
（4）新进程复现完整验证且逐题预测等于冻结缓存；
（5）检查未知ID、批处理、候选置换的答案不变性、空输入、重复候选，预测阶段拒绝读取参考答案；
（6）只对已冻结新方法运行一次完整官方TEST预测和独立评分；
（7）复核冻结哈希不变，保存官方JSON与资源信息；测试结果不回流训练、规则修改或模型选择。
研究data_access.py的TEST防护保持不变，最终评估使用独立入口。已完成输出目录不可静默重复测试。以后追加新feature需先完成训练与验证冻结，再使用新的输出目录；不能依据既有TEST错题开发feature。
本轮实际命令：
D:/anaconda3/envs/mp1/python.exe research/feature_validation.py --selection experiments/round54/final_feature_selection.json --method research/round54_selector.py --output experiments/feature_validation_20261022 --with-test
命令中的--with-test是最终评分开关，不是开发循环的评分指标。已完成的本轮目录禁止重复运行；复现时查看冻结commands和包。

7. 资源和审计
测试资产{final['asset_bytes']:,}字节（{final['asset_MiB']:.6f}MiB），低于16MiB。
本机Intel Core i7-11800H，四线程，完整467题新进程预测{final['test_prediction_seconds']:.3f}秒、独立评分{final['test_scoring_seconds']:.3f}秒、合计{final['test_total_fresh_seconds']:.3f}秒；本机是否低于30秒：{final['local_under30']}。参考Xeon8457C正式时限资格尚未认证，不能把本机时间当参考机器认证。
训练墙钟：增强三臂{training['52']['seconds']:.2f}秒；匹配词法SFT{training['53']['seconds']:.2f}秒。解析覆盖准备及训练/诊断有部分并行，不应相加当总墙钟；最后独立TEST计时在训练和其他重任务完成后执行。单张RTX A2000 Laptop GPU，PyTorch{training['52']['torch']}；推理CPU，不使用外部数据或权重。
32题、{diagnostics['executor_audit']['programs']}个程序独立执行/奖励/候选映射审计通过。旧实验及本轮准备冻结均核验。预测包装拒绝读取train/validation/test参考答案；独立官方评分器按协议读取评分参考。

8. 产物
experiments/round51：拟合发现列表、初始冻结、全TRAIN程序和增强特征。
experiments/round52：关系抽象三臂、完整验证、TRAIN关系和奖励诊断。
experiments/round53：同程序同预算的词法SFT对照。
experiments/round54：第二次拟合侧解析修复、原38题覆盖审计、最终feature选模与汇总。
experiments/feature_validation_20261022：冻结独立包、完整官方TEST成绩、资源与接口复核。
experiments/round51_54_completion_manifest.json：本轮源文件与产物哈希。

本轮没有达到的事项须明确保留：剩余程序缺口、关系泛化不足、97%测试目标，以及参考机器时限认证。后续应继续在拟合部分审查有正确数值却错绑定的程序，并用按对象/关系组合隔离的训练留出评估；不能把本轮TEST结果用作下一轮规则设计依据。
'''
    path=ROOT/'程序覆盖与关系泛化及最终测试_2026-10-22_第51至54轮.txt';path.write_text(report,encoding='utf-8')
    # Verify final models used all allowed rows, diagnostic process labels exclude held parents.
    import numpy as np
    with np.load(ROOT/'experiments/round51/train_targets.npz') as z:ids=z['ids'];folds=z['folds']
    heldparents={r['train_index'] for r in read(ROOT/'experiments/round51/held_source_review_before_reference.json')}
    for i in (52,53):
        for stage in ('diagnostic','full'):
            expected=ids[folds!=2].tolist() if stage=='diagnostic' else ids.tolist()
            for p in (ROOT/f'experiments/round{i}'/stage).glob('*/training.json'):
                r=read(p);assert r['training_ids']==expected,p
                assert not set(r.get('constraint_parents',[]))&heldparents,p
        for p,h in read(ROOT/f'experiments/round{i}/plan.json')['source_hashes'].items():assert sha(ROOT/p)==h,p
    for i in (51,52,53,54):
        for p,h in read(ROOT/f'experiments/round{i}/prepared_freeze.json')['hashes'].items():assert sha(ROOT/p)==h,p
    files=[path,Path(__file__).resolve(),ROOT/'research/feature_validation.py']
    files+=[p for p in (ROOT/'research').glob('round5[1-4]*.py')]
    for i in (51,52,53,54):files += [p for p in (ROOT/f'experiments/round{i}').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [p for p in (ROOT/'experiments/feature_validation_20261022').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    manifest=dict(completed=True,goal_97_achieved=final['goal_97_achieved'],test_used_for_tuning_or_selection=False,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))})
    output=ROOT/'experiments/round51_54_completion_manifest.json';dump(output,manifest)
    for p,h in read(output)['hashes'].items():assert sha(ROOT/p)==h,p
    print(dict(report=str(path),test_correct=final['test_correct'],test_accuracy=final['test_accuracy'],selected=selection['selected'],files=len(manifest['hashes'])),flush=True)

if __name__=='__main__':main()
