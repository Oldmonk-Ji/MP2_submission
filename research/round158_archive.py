"""Archive completed experiments, including rejected and invalidated trials."""
from round22_common import ROOT,read,dump,sha
import json
if __name__=='__main__':
    records=[]
    for folder,arm in [('round141','literal'),('round151','literal'),('round154_corrected','markup'),('round157','literal')]:
        base=ROOT/'experiments'/folder;assert (base/'training_complete.json').exists();checks=read(base/'feature_checks.json');assert not checks['test_opened']
        record=next(r for r in read(base/'validation_records.json') if r['arm']==arm);held=read(base/f'{arm}_held.json');records.append(dict(experiment=folder,**record,held_correct=held['correct'],held_n=held['n']))
        assert record['correct']<=759
    summary=dict(current_recommendation='round104 local',validation_correct=759,validation_n=798,previous_frozen_test_correct=391,test_n=467,goal_required_test_correct=402,goal_active=True,goal_complete=False,test_opened_in_this_phase=False,results=records,
        initial_literal_head=read(ROOT/'experiments/round139/held_relations.json'),
        normalized_literal_head=read(ROOT/'experiments/round149/held_relations.json'),
        context_literal_head=read(ROOT/'experiments/round156/held_relations.json'),
        algebra_fit_audit=read(ROOT/'experiments/round147/fit_audit.json'),
        valid_markup_fit_audit=read(ROOT/'experiments/round152/fit_audit.json'),
        normalized_literal_coverage=read(ROOT/'experiments/round155/fit_audit.json'),
        factorized_supervision_fit=read(ROOT/'experiments/round158/fit_stats.json'),
        factorized_supervision_full=read(ROOT/'experiments/round158/full_stats.json'),
        invalidated=['round144','round145','round146','round154'],invalidation_explanations={n:read(ROOT/f'experiments/{n}/invalidation.json') for n in ('round144','round145','round146','round154')})
    dump(ROOT/'experiments/round139_158_summary.json',summary)
    dump(ROOT/'experiments/next_after_round158.json',dict(goal_active=True,target_test_correct_at_least=402,current_recommendation='round104',next_start='Implement three separately supervised heads from frozen round158 labels: source use, literal presence, conditional scope. Keep fixed C1 and parent-balanced weighting; do not sweep parameters. Compose P(absent)=1-P(source used)*P(literal present), and P(scope k)=P(source used)*P(literal present)*P(scope k | present).',supervision='round158/{fit,full}_{source,constant,scope}_labels.json',scope_classes=4,feature_plan='Source: existing round139 source_features. Literal: whole-question anonymous lexical/typed features crossed with canonical literal value, without source occurrence fields. Scope: existing round139 pair_features, trained only on agreed present relations. Start lexical-only to isolate factorization; no new pretrained weights.',fit_only_diagnostic='Audit frozen-fit coverage and hard decision coherence; labels with any reference disagreement remain unknown. Strict original OOF/nested stages; all auxiliary fit/centering must exclude held parents.',integration='Keep exact round104 base and fixed meta settings; reuse round150 normalized root packets and27 summary features for a matched comparison. All inference caches must be reproduced from new inputs and checked against exported models.',selection='Adopt only when full VAL >759 and fresh feature/interface checks pass. Then freeze and run official aggregate TEST once through standard feature_validation.',resource_policy='16MiB official assets unchanged; local laptop screen roughly33s, official reference30s remains authoritative.',test_used_for_tuning=False))
    report=ROOT/'常数语义监督与程序覆盖审计_2026-10-26_第139至158轮.txt'
    report.write_text('''第139至158轮：常数语义监督、程序覆盖与一致性审计

结论：尚未达到86%测试目标，继续保留第104轮模型。
现行模型验证集759/798（95.11%），此前冻结测试391/467（83.73%）。超过86%至少需要402/467，仍差11题。本阶段没有读取测试题、错误案例或测试参考来调参，也没有对较弱模型追加测试。

本阶段有效对照（训练留出1582题；完整验证798题）：
原104：训练留出1477，验证759。
141，加入初版常数关系监督：训练留出1472，验证756。
151，先统一等价代数表达再监督：训练留出1468，验证758。
154_corrected，修复加粗乘法公式抽取：训练留出1469，验证753。
157，给归一化后的关系预测增加所提供固定表示的题目上下文：训练留出1469，验证756。
所有有效新增方法都低于基线，不进入测试选型。原基线在各对照中逐题复现。

根因证据：
1. 初版常数关系监督在训练拟合部分有292题找不到满足全部已知关系约束的正确候选程序。但精确代数审计发现，其中50题已经有等价正确程序，不能把292全称为“程序缺失”。
2. 有界精确Laurent展开统一了上述50题中的49题：百分数分母、乘除链、分配律、减法链不再轻易产生不同关系标签。通过24926次随机有理数精确执行检查，10次未定义取值跳过。非单项分母或规模超限回退旧的精确表示。仍不是完整通用计算机代数系统，也不把题目中数值相等的不同对象当成同一来源。
3. 修正后，全训练参考产生42888条一致关系标签，覆盖3010个父题；其中39053条为ABSENT。真正存在的关系在训练留出的预测为664/1296，整体12802/14066的高正确数不能解释为已掌握语义。
4. 增加所提供的整题上下文后，整体关系预测12916/14066，但真实存在关系只到669/1296；父题/存在性组平衡正确率未提高。最终选择也没有提升。
5. 新归一化监督在1028道有正确、错误候选且受监督的训练拟合题中：770题覆盖某个正确程序，503题能排除全部错误候选，258题仍无满足已知约束的正确程序。由于监督集合改变，这些数字不能直接与旧集合相减当作净修复数。
6. 加粗公式内含乘号时，旧归一化顺序可能残留Markdown标记并漏掉整段运算。修复只处理适当配对的加粗标记、保留幂运算和官方候选答案，并保留全部原程序。有效审计新增315个程序，涉及84道训练拟合题；新增程序全部执行到所属候选原答案。6道题首次恢复与参考代数等价的正确程序，但整体选择准确率仍下降。

质量检查发现并撤回的试验：
144—146的合并未统一JSON列表与实时元组，导致重复程序；覆盖数字和模型结果已明确标为无效。152—154重新统一树结构并重跑。
154首次重跑又被全验证集导出检查发现：缓存将float32概率提升为float64后再聚合，与实时推理不同。保留原失败记录，修正为154_corrected并重新训练。最终通过全部798题辅助特征与导出权重一致性、16道包含实际变化题的实时抽取/预测检查、分批一致性及零数量对输入检查。
141、151、157也通过对应全验证辅助特征一致性、实时预测、分批、候选置换和空数量对检查。不能把这些专项检查声称为每个新增模型都重新跑过完整135项正式接口验收。

下一条已冻结的训练监督：
将“不存在数量—常数关系”拆成“该数量未被使用”或“该常数不应出现”，再在二者都存在时学习运算范围。全训练得到9902条数量使用标签、11093条常数存在标签、3835条条件关系标签；3025项数量使用与855项常数存在因参考备选有分歧而留空。
只在训练拟合部分作结构审计：1129个完整监督矩阵都满足来源使用集合与常数集合的笛卡尔积结构；原独立分类器有41个父题的逐项最大概率决策不能同时满足这个结构。这是点决策一致性诊断，不表示其软概率非法，也不表示41题已被修复。
下一步训练三个共享决策，再组合为关系概率；不增加外部模型，不作参数扫描，保持现行基线与固定选择器设置，继续用留出和验证衡量真实收益。

作业与资源：
仅给定训练输入/参考用于拟合；验证用于开发，测试只供冻结后的正式评分。标签是保守参考程序弱监督，不是独立人工语义金标。新增规则只由训练拟合部分制定。
仍输出原四候选之一，官方解析器与评分器未改。新增候选程序只是内部解释，不生成新的候选答案。
本机耗时按用户许可可用约33秒筛选，正式参考机30秒及16MiB限制不变。本阶段模型都未过准确率门槛，没有将并行训练时耗时冒充正式CPU评估。现行104此前完整新进程测试及独立评分23.3023秒，额外资产15562541字节。
有效新增模型资产约15.56—15.93百万字节，均在16MiB内。具体字节数、训练耗时、种子、父题隔离记录和源码/产物哈希见experiments/round139_158_summary.json及各轮目录。

目标保持active，不声称已达到86%，也不因本阶段阴性结果而停止后续优化。
''',encoding='utf-8')
    for n in range(139,159):
        folder=ROOT/f'experiments/round{n}';paths=[p for p in folder.rglob('*') if p.is_file() and p.name!='completion_manifest.json']+list((ROOT/'research').glob(f'round{n}_*.py'))
        dump(folder/'completion_manifest.json',dict(completed=True,valid_for_model_selection=n not in (144,145,146,154),test_used_for_tuning=False,hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}))
    folder=ROOT/'experiments/round154_corrected';paths=[p for p in folder.rglob('*') if p.is_file() and p.name!='completion_manifest.json']+list((ROOT/'research').glob('round154_*.py'))
    dump(folder/'completion_manifest.json',dict(completed=True,valid_for_model_selection=True,test_used_for_tuning=False,hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}))
    dump(ROOT/'experiments/round139_158_report_manifest.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in [report,ROOT/'experiments/round139_158_summary.json',ROOT/'experiments/next_after_round158.json']}))
    print(json.dumps(dict(results=records,current='round104',goal_active=True,test_opened=False),ensure_ascii=False),flush=True)
