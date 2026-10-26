from round22_common import ROOT,read,dump,sha
import json
if __name__=='__main__':
    records=[]
    for n,arms in [(171,('balanced_control','reference_mask')),(174,('literal',))]:
        folder=ROOT/f'experiments/round{n}';assert (folder/'training_complete.json').exists();assert not read(folder/'feature_checks.json')['test_opened']
        for arm in arms:
            r=next(r for r in read(folder/'validation_records.json') if r['arm']==arm);held=read(folder/f'{arm}_held.json');assert r['correct']<=759;records.append(dict(round=n,**r,held_correct=held['correct'],held_n=held['n']))
    dump(ROOT/'experiments/round169/superseded.json',dict(superseded_before_held=True,replacement='round169_refined',reason='TRAIN-fit review revealed implicit count2 bound to ordinal second. Initial source preserved as source_initial.py; final rules fixed before full TRAIN use.',test_opened=False))
    summary=dict(current_recommendation='round104 local',validation_correct=759,validation_n=798,previous_frozen_test_correct=391,test_n=467,goal_required_test_correct=402,goal_active=True,goal_complete=False,test_opened_in_this_phase=False,results=records,reference_mask_stats=read(ROOT/'experiments/round169_refined/full_stats.json'),balanced_control_heads=read(ROOT/'experiments/round170/balanced_control/held_source.json'),reference_mask_heads=read(ROOT/'experiments/round170/reference_mask/held_source.json'),target_interaction_heads=read(ROOT/'experiments/round173/interaction/held_source.json'),fraction_type_fit=read(ROOT/'experiments/round175/fit_stats.json'),fraction_type_full=read(ROOT/'experiments/round175/full_stats.json'),fraction_type_model_trained=False)
    dump(ROOT/'experiments/round169_175_summary.json',summary)
    dump(ROOT/'experiments/next_after_round175.json',dict(goal_active=True,target_test_correct_at_least=402,current_recommendation='round104',next_start='Run matched corrected-type ablation using frozen175 source labels, unchanged158 constant/scope labels and fixed C1 models. Two arms: corrected_labels with original139 source/pair features; corrected_types with175 source_features and a new pure pair_features wrapper applying the same kind/family correction plus literal-kind/family cross-feature replacement.',reason='Existing46 graph recognizes a third only case-sensitively and misclassifies one-third/one fourth and every third as ordinals. TRAIN-fit58 typed positions in53 parents were inspected and are fraction/interval expressions.175 restores45 clear positive source labels from169 ordinal abstention. This is not yet a measured selection gain.',training='Original OOF/nested stage parent isolation; total-preserving class balance for source only as170; constant binary head and scope4-class per-parent weighting as159. Keep constant weights bitwise identical159; refit conditional scope only in corrected_types because its source/type feature changes. No new weights/data outside suppliedTRAIN.',inference='Use same162 separate40 decision summary. Candidate root extraction and baseline104 remain unchanged for first matched trial. New pure packet must emit corrected source and corrected source/literal pair blocks; cache vs fresh features, export probabilities, batching, empty-pair and candidate permutation checks required.',caution='Do not mutate frozen graph/roles caches or ordinary source modules in place. Do not treat final-correct candidate arithmetic hypotheses as process gold. Positive true-ordinal arithmetic bindings remain uncertain; explicit fractions and periodic intervals must not be discarded as ordinals.',followup_if_no_gain='Prioritize TRAIN-fit audit of a strict candidate trace parser: explicit correct numeric equalities, source/verified-intermediate lineage, no free constant shortcut, no right-hand claimed-number root, conservative ambiguity overflow. Use improved fraction typing. Compare correct-program coverage and noise before a new selector; preserve original candidates/final answers. This targets program coverage rather than another parameter search.',selection='Only fullVAL>759 plus fresh validation/interface/resource checks can advance to frozen official TEST through standard feature_validation. Never inspect TEST errors or use its score to select.',resource_policy='Local laptop roughly33s screen; official reference30s and16MiB unchanged.',test_used_for_tuning=False))
    report=ROOT/'序数误绑定与分数类型审计_2026-10-26_第169至175轮.txt'
    report.write_text('''第169至175轮：参考来源歧义、目标条件关系与分数类型审计

当前目标仍未达成。保留104模型：验证759/798；此前冻结测试391/467（83.73%）。超过86%需402/467，本阶段没有读取或评分测试集，目标保持active。

完整候选选择结果：
171 balanced_control：训练留出1472/1582，验证756/798。
171 reference_mask：训练留出1471/1582，验证756/798。
174 target/source interaction：训练留出1467/1582，验证755/798。
基线为留出1477、验证759。以上新增方法均未超过基线，不采用、不进入测试选型。每次控制组逐题复现104结果。

参考来源审计与冻结规则：
不再使用“最终正确候选的某个程序包含该数量/常数”来大范围删除负监督。正确答案并不保证推理正确，上一阶段该办法丢失过多负例。
169只按训练参考和题目判断表示歧义：单位/中性1、可能由参考字面常数编码的数量、明确第一至第k项枚举中的隐含计数。只改为未知，不凭数值相等翻转正标签；常数及条件关系监督与158逐位一致。
训练拟合题58进一步发现：参考中的乘数2被绑定到了序数词second。这个2代表两个其余拼图，并非算术使用“第二个”的序号。新增序数正绑定未知规则在使用留出之前冻结；初始fit版本和源码保留在round169，实际全训练版本为round169_refined。
该版本保留8962条数量标签，其中507条负例；不再删去大量常数负例。使用相同全局类别平衡损失进行匹配控制，最终验证仍均为756，修复表示歧义尚未带来选择收益。

目标与数量局部角色的交互：
原数量特征把题目目标与局部词分别相加，缺少它们的直接交互。例如“收到多少”和“剩余多少”对初始库存、转移量有不同要求。
172在相同16384维哈希中加入现有目标词×现有局部/类型特征乘积。只改变数量使用头；常数及条件关系模型保留159，C=1与训练权重不变。
在共同保留的2958条留出标签上，正确数2700到2758，但“不使用”判对64/166降到54/166；最终验证755。不能把中间整体正确数上升称为最终收益。

更上游的类型错误：
继续核查仅TRAIN拟合部分的序数正绑定时，发现one-third、one fourth等明确分数也被46图标为ordinal。旧规则只识别小写a后面的third/fourth/fifth，还漏掉句首A third等情况。“every third shopper”则是周期计数，不能与“third puzzle”同样处理。
175新增纯输入类型解释器：显式分数分母、every第几的间隔分母与真正序号分开；数量数值和槽位完全不变。未修改旧46图、52角色缓存或104推理源码。
TRAIN拟合中改正58个位置、覆盖53父题，其局部原文逐项经AI助手检查；这是助手审查，不是独立人工金标。全TRAIN中恢复45条此前因ordinal而留空的明确分数/间隔正标签，仅18条真正序号的正算术绑定继续未知。
最终175数量监督9007条，其中507负例；常数11093、条件范围3835条与158完全相同。该规则在full应用前冻结。只是监督与输入表示准备，尚未用175训练新模型，不能报告分类或选择准确率提升，也没有声称已验证所有语言形式的泛化。

下一阶段：
以相同175标签比较“只修监督”与“同时修数量/数量—常数关系输入类型”。分数类型不仅影响数量使用，也影响除法方向和常数作用关系，所以类型输入修复应覆盖条件关系特征；保持模型维数、C与选择器设置，避免把收益混同参数扫描。
若仍无收益，转向严格候选算术过程抽取：只使用正确数值等式、源数量和经验证中间量；拒绝凭右侧声称答案或任意短常数补全依赖。先在TRAIN拟合部分核查正确程序覆盖和噪声，再冻结并验证。保留原候选答案，不能把过程一致性当作最终答案正确的充要条件。

验证与作业边界：
171两个新臂、174均已检查全部798题新增特征与导出模型一致；实时小批抽取/预测、分批、零数量对及候选置换性质有专项检查。未用专项检查冒充整套正式接口验收。
额外资产约1609万字节，均低于16MiB。本机耗时约33秒筛选许可保留，正式参考30秒和16MiB没有修改。本阶段没有方法通过准确率门槛，因此未追加测试新进程耗时或评分。
仅给定TRAIN用于拟合，训练留出/VAL用于开发验证；无外部数据、新预训练权重、答案索引或测试调参。新方法继续只选原四个候选之一。
机器结果、训练隔离、源码与产物哈希见experiments/round169_175_summary.json和各轮目录。
''',encoding='utf-8')
    for n in range(169,176):
        folder=ROOT/f'experiments/round{n}';paths=[p for p in folder.rglob('*') if p.is_file() and p.name!='completion_manifest.json']+list((ROOT/'research').glob(f'round{n}_*.py'));dump(folder/'completion_manifest.json',dict(completed=True,fit_audit_superseded=n==169,test_used_for_tuning=False,hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}))
    folder=ROOT/'experiments/round169_refined';paths=[p for p in folder.rglob('*') if p.is_file() and p.name!='completion_manifest.json']+[ROOT/'research/round169_reference_mask.py'];dump(folder/'completion_manifest.json',dict(completed=True,test_used_for_tuning=False,hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    dump(ROOT/'experiments/round169_175_report_manifest.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in (report,ROOT/'experiments/round169_175_summary.json',ROOT/'experiments/next_after_round175.json')}));print(json.dumps(dict(results=records,goal_active=True,recommendation='round104',test_opened=False),ensure_ascii=False),flush=True)
