"""Archive factorization, label-quality and class-objective ablations."""
from round22_common import ROOT,read,dump,sha
import json
if __name__=='__main__':
    records=[]
    for n in (161,163,166,168):
        folder=ROOT/f'experiments/round{n}';assert (folder/'training_complete.json').exists();assert not read(folder/'feature_checks.json')['test_opened'];r=next(r for r in read(folder/'validation_records.json') if r['arm']=='literal');assert r['correct']<=759
        held=read(folder/'literal_held.json');records.append(dict(round=n,**r,held_correct=held['correct'],held_n=held['n']))
    summary=dict(current_recommendation='round104 local',validation_correct=759,validation_n=798,previous_frozen_test_correct=391,test_n=467,goal_required_test_correct=402,goal_active=True,goal_complete=False,test_opened_in_this_phase=False,results=records,
        factor_heads=read(ROOT/'experiments/round159/held_heads.json'),filtered_heads=read(ROOT/'experiments/round165/held_heads.json'),balanced_heads=read(ROOT/'experiments/round167/held_heads.json'),filtered_label_stats=read(ROOT/'experiments/round164/full_stats.json'),fit_semantic_review=read(ROOT/'experiments/round164/fit_semantic_review.json'))
    dump(ROOT/'experiments/round159_168_summary.json',summary)
    dump(ROOT/'experiments/next_after_round168.json',dict(goal_active=True,target_test_correct_at_least=402,current_recommendation='round104',next_start='Repair source-use supervision from TRAIN-fit question/reference provenance, not from arbitrary final-correct candidate graphs. First audit absence ambiguity due neutral/unit1 and implicit literal coefficients/unrolled counts; preserve reliable negatives instead of removing81% of constant-negative labels.',evidence='TRAIN-fit12-case AI review in round164; guide explicitly permits final-correct candidates with flawed reasoning. Broad candidate-based unknown filtering reduced source negatives1384->439 and constant negatives8742->1489, and did not improve selection.',rules='Design any mask only on TRAIN-fit. Never relabel absent sources as positive merely from numeric equality or final-correct candidate hypotheses. Document neutral/unit and implicit-count ambiguity; leave unsupported cases unknown. Freeze before held aggregate verification.',matched_plan='Keep original158 constant and conditional-scope supervision while changing source absence policy only. Fixed C1,16k hashed dimensions, original OOF/nested stages, fixed selector; avoid a parameter sweep. If a sufficiently clean source-negative set remains, evaluate source target/local-role interactions as a separate ablation rather than conflating supervision and features.',available='round159 factorized model; round162 separate40 decision features; round167 globally class-balanced source loss with unchanged total weight and bitwise unchanged constant/scope weights.',selection='Require fullVAL>759, fresh export/cache/interface checks and16MiB before freezing and standard feature_validation --with-test. TEST must not determine policy or selection.',resource_policy='Local laptop roughly33s screen; official reference30s and16MiB unchanged.',test_used_for_tuning=False))
    report=ROOT/'数量使用与关系决策消融_2026-10-26_第159至168轮.txt'
    report.write_text('''第159至168轮：数量使用、常数存在与关系决策消融

尚未达到86%目标，目标保持active。当前仍保留第104轮：验证759/798，之前冻结测试391/467（83.73%）。超过86%需要402/467。本阶段没有打开或评分测试集，没有把较弱模型换成推荐模型。

有效完整对照（训练留出1582；验证798）：
104基线：留出1477，验证759。
161，分解为数量使用、常数存在、条件运算关系，再组合概率：留出1472，验证757。
163，分别呈现三个中间决策的40项证据：留出1472，验证756。
166，对可疑“不使用/不存在”监督改为未知：留出1469，验证754。
168，仅对数量使用分类做全局类别平衡、总训练权重不变：留出1474，验证754。
四项均没有超过验证基线。每项控制组均逐题复现104结果。

分解决策的结果：
使用冻结158标签，数量使用有9902条，常数存在11093条，条件运算关系3835条。
独立条件运算分类在留出1296条真正存在关系中判对877条。组合数量使用/常数存在后，完整关系判对681条，较旧独立五分类664条增加17条；但全部14066条标签的正确数从12802降至12718。整体选择验证757，不能以单个中间任务的改善声称最终准确率提高。
分别呈现三项决策也未改善最终验证，结果756。

训练拟合错题的实际审查：
在严格nested预测中，有412个旧标签为“不使用”的数量被高概率预测为使用。只核查训练拟合部分的12个例子，没有读取留出文本制定规则。
其中有真实无关量：桥梁车道数、给定转移量之外的初始库存、对年龄差抵消的共同时间增量等。
也有表示遗漏：两杯各4元在参考中写作4+4，没有出现数量2的叶节点；三个对象的数量可由展开枚举隐含表示；比例/单位中的1可在公式中省略；总量与分量允许同题中的替代表示。因此“某参考程序没有某叶节点”不等于“该数量语义无关”。
12例解释已保存在round164/fit_semantic_review.json，审查者是AI助手，不是独立人工金标，也未据此批量翻转为正标签。

保守过滤的覆盖代价与失败：
中性1、正确答案候选程序的可能使用、缺少可执行正确候选证据，都会把负标签改为未知。没有将这些候选假设变成正语义标签。
全训练保留8957条数量使用标签，仅439条负例；常数存在保留3840条，仅1489条负例。原常数负例8742条被大幅削减，其中7180条因正确答案候选的某个程序包含该常数而留空。
这个过滤过于宽泛。作业guide明确指出正确最终答案也可能来自错误推理；候选程序还有数值同值绑定和省略中间量假设，不能当作可靠过程证据。过滤后的结果754，没有提高选择准确率。
下一步不继续把这种广泛候选假设当作监督来源，应回到题目/训练参考可追溯的单位、计数与关系证据，修复已确认的表示歧义并保留可靠负例。

类别偏置的独立对照：
只改变数量使用头的损失，把两类总权重各设为原总权重一半；C=1、特征、总权重、其他两头不变。所有阶段常数和条件关系权重与165逐位一致。
在同一过滤后留出集合，不使用判对38/137提高至77/137，但使用判对2814/2819降至2699/2819。最终留出1469提高到1474，完整验证仍为754。类别偏置确实存在，但消除它还不足以取得最终收益。
过滤前后中间标签集合不同，不能把它们的总体百分比直接横向比较；部分组合关系指标仍按旧参考标签计算，只是表示一致性诊断，不是新的独立语义金标。

执行与验证：
所有训练仅使用提供的TRAIN数据，原fold0/1拟合、fold2留出及严格OOF/nested父题隔离保持不变。无外部数据或预训练权重，所有输出仍是原四候选之一。
161、163、166、168都检查了全部798题新增特征与导出权重一致、实时小批抽取/预测一致、分批一致及零数量对输入；候选置换等性质有专项检查。没有把这些检查冒充全部正式135项接口验收。
额外资产约1609万字节，低于16MiB。没有因更慢的本机而改动作业资产上限。本机约33秒可作为筛选门槛，正式参考30秒不变。本阶段无方法过准确率门槛，未追加正式新进程测试耗时或测试评分。
源码、训练时长、标签父题隔离、结果、检查与哈希见experiments/round159_168_summary.json及各轮目录。
''',encoding='utf-8')
    for n in range(159,169):
        folder=ROOT/f'experiments/round{n}';paths=[p for p in folder.rglob('*') if p.is_file() and p.name not in ('completion_manifest.json','completion_manifest_final.json')]+list((ROOT/'research').glob(f'round{n}_*.py'))
        name='completion_manifest_final.json' if (folder/'completion_manifest.json').exists() else 'completion_manifest.json'
        dump(folder/name,dict(completed=True,test_used_for_tuning=False,hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}))
    dump(ROOT/'experiments/round159_168_report_manifest.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in (report,ROOT/'experiments/round159_168_summary.json',ROOT/'experiments/next_after_round168.json')}))
    print(json.dumps(dict(results=records,recommendation='round104',goal_active=True,test_opened=False),ensure_ascii=False),flush=True)
