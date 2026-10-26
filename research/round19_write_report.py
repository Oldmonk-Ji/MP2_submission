"""Archive retained candidates and summarize the completed TRAIN-only round."""
from round19_common import OUT,ROOT,dump,sha
import datetime,json


def read(path):return json.loads((OUT/path).read_text(encoding='utf-8'))


def main():
    f=read('frozen.json');a=read('audit_result.json');d=read('posthoc_diagnostics.json');p=read('plan.json')
    assert all(sha(ROOT/k)==v for k,v in f['hashes'].items())
    records=f['secondary'];primary=f['primary'];res=a['results'];names={
      'legacy_short':'原结构＋按开发损失早停', 'regularized_short':'小模型＋强正则＋短文本',
      'regularized_long':'小模型＋强正则＋完整候选', 'regularized_independent':'完整候选、关闭候选交互',
      'anchored_long':'完整候选＋词语/数值位置对齐', 'anchored_independent':'位置对齐、关闭候选交互',
      primary['name']:'核模型（预先选定的主实验）','linear_C0.1':'匹配的线性对照'}
    table=[]
    for r in records:
        v=res[r['name']]
        table.append(f"{names[r['name']]}：开发 {r['selected']['correct']}/650；审查 {v['correct']}/950（{v['accuracy']*100:.2f}%）；修复{v['repaired']}、误伤{v['damaged']}、净增{v['net']:+d}。")
    registry=dict(status='candidates_retained_for_replication',incumbent=str(ROOT/'experiments/round4/selected_bundle'),
        incumbent_historical_validation=dict(correct=750,n=798,accuracy_percent=100*750/798),
        promoted=False,discarded=False,validation_opened=False,test_opened=False,
        distinction='All model files are retained. Failure of the preregistered promotion gate is insufficient evidence for replacing the incumbent, not a reason to delete a candidate.',
        primary=dict(name=primary['name'],path=str(OUT/primary['path']),audit=res[primary['name']],status='retained_primary_candidate'),
        exploratory_long=dict(path=str(OUT/'neural/regularized_long'),audit=res['regularized_long'],status='retained_secondary_candidate_not_promoted_after_audit'),
        rule_origin='Agent-chosen preregistered round19 promotion policy, not a course requirement.',
        all_trained_variants_retained=True)
    dump(OUT/'candidate_status.json',registry)
    lines=[]
    for name,v in d['pair_quality'].items():
        lines.append(f"{name}：拟合 {v['fit']['accuracy']*100:.2f}%，开发 {v['development']['accuracy']*100:.2f}%，审查 {v['audit']['accuracy']*100:.2f}%。")
    main_result=res[primary['name']];ci=main_result['bootstrap_95_ci_percentage_points']
    neural=read('neural/selection.json');kernel=read('kernel/selection.json')
    base=read('baseline/done.json');training_seconds=sum(x['seconds'] for x in neural)+kernel['seconds']+sum(x['seconds'] for x in base['ledger'])
    text=f'''第19轮：向95%推进的真实候选实验
2026-10-15，UTC+8

结论与候选状态
本轮完成22种配置：6种神经版本、12种RBF核模型配置、4种线性对照。
所有权重、配置、词表与结果均已保存；有小幅提升的版本保留为候选，没有删除。
本轮没有覆盖现有主模型，没有进入官方验证集，未读取测试集。
原主模型历史官方验证成绩仍为750/798（93.98496%）。95%至少需要759/798，即净增9题；本轮不能声称已达到。

“未过门槛”的准确含义
门槛是实验前由实现者设定的晋级规则，不是课程规定，也不是用户指定的数值标准。
本轮要求：开发净增>0、封存审查净增≥5、精确单侧配对检验p≤0.05、95%分组bootstrap区间下界>0。
它判断的是是否已有足够证据推进当前主实验，不决定是否保存或继续研究一个模型。
核模型在这批审查题上确实修复1题且没有误伤，这个观察是真实的。
但是950题中仅1题改变了正确/错误判分结果，另22次候选切换都是正确答案之间的切换，稳定收益仍不确定。
因此“尚未证实稳定提升”不等于“没有价值”，更不等于“应丢弃”。保留候选与替换主模型是不同决定。
对当前95%阶段，这个晋级标准较保守。后续可以另行预先规定固定配置复验流程；不能在看到本轮审查后篡改原门槛并把失败写成通过。
candidate_status.json明确记录主候选、探索候选及原主模型路径。

一、隔离与评估
官方4755道训练题重新按归一化题干模板分组：3155拟合、650开发调参、950封存审查。
同题四份候选以及相同归一化模板不能跨组。
上一轮950道已评审题全部只允许进入本轮拟合区，不进入新的开发区或审查区；已知人工标注题也排除在新审查区之外。
旧的全训练集基线权重未用作本轮审查预测。现有五个基线成员、词表、标准化和融合系数全部按本轮划分重训。
项目已有长期训练集探索，所以新划分是本轮拟合与评估隔离，不是全新外部基准；不宣称消除了历史选择影响。
参考解仅用于解析最终数值标签，没有使用参考解步骤监督，也没有生成题、外部语料或额外预训练权重。

二、实现变化
神经分支将宽度32缩至16，词表4096缩至2048，dropout从0.15增至0.35，权重衰减从0.01增至0.15，加入0.95 EMA和0.05配对标签平滑。
检查点按开发区未平滑的配对损失选择，避免用少数改答的偶然涨分挑训练后期模型。
短版约62513参数，长版68145参数，原结构197089参数。所有可训练网络权重随机初始化，仍使用官方固定候选嵌入作为输入。
候选长度上限160增至512；本轮拟合、开发、审查的所有候选都完整保留。审查候选token覆盖从85.14%升至100%。
长版审查题目覆盖也是100%，但小词表的题目未知词比例为10.53%，完整读入并不等于完整理解。
追加的词语对齐保留同一人物名、物品词和精确数值的位置身份，包括词表外的人名；这些身份给注意力加软偏置，不推断人物归属、单位角色或时间相等。
词语对齐是在第一阶段开发比较后登记的追加实验，见amendment_01.json；当时未读取新审查标签。开发过程有适应性，最终只设一个主实验。
核分支仅用拟合数据学习96维PCA，结合题内差分、乘积及算术/文本特征，使用256个拟合数据基点训练带L2正则的核排序器。
基点是数值特征，不保存题目ID、问题文本或答案表。匹配线性对照使用相同特征空间。
两类方法均对同题正误候选训练，多正确候选不会互相当负例。

三、开发选型与一次封存审查
基线开发600/650，审查880/950（92.63%）；审查冲突子集341/411。
所有融合系数只在开发区从0.1、0.25、0.5、1.0中选择，并共享已有固定算术门控。
神经检查点先按开发损失固定，再选融合系数。核参数只在开发区筛选。
按事先规定的并列规则，rbf_g3.0_C0.1被选为唯一主实验：gamma因子3、C=0.1、融合系数0.1。
主实验与所有对照先冻结模型、配置、代码及输入哈希，然后一次性评估950道封存题。

{chr(10).join(table)}

主实验准确率增益{main_result['gain_percentage_points']:.3f}个百分点，95%分组bootstrap区间[{ci[0]:.3f}, {ci[1]:.3f}]个百分点，配对p={main_result['paired_one_sided_p']:.3f}。
长文本联合版是预先登记的次要对照：修复3、误伤1，净增2，p=0.3125，95%增益区间[-0.211,0.632]个百分点。
不能因为次要对照审查分数最高，就事后替换主实验并宣布通过。它保留为下一阶段复验候选。
这些审查结果与上一轮的875/950使用不同题目及拟合划分，不能直接相减表示迭代提升。

四、过拟合与覆盖
下面统计的是同题正确候选相对错误候选的聚合排序分数，不是关系字段抽取正确率：
{chr(10).join(lines)}
核模型的拟合与审查差距较小，但仍未显著改善强基线。
长文本神经模型完整读取候选后，审查排序表现仍不足；因此本轮证据不支持把截断视为唯一瓶颈。
位置对齐版本最终净增0，不能把“能匹配人名/词语位置”写成“已可靠掌握人物指代、单价作用域和时间关系”。
原结构本轮第2轮早停，长文本版第6轮早停；与上轮第14轮检查点的数据划分、目标及设置不同，不能把跨轮差值归因于某一项正则化。

五、主实验修复的真实案例
训练索引3521：Kayla摘20个苹果，Caleb少5个，Suraya比Caleb多12个，问Suraya比Kayla多多少。
基线选择27，把Suraya的总量当成所求差值；核模型改选7，符合官方最终答案。
这是实际候选上的一项修复，但所选候选正文没有完整写出最后一次减法，因此不能据此证明模型真正计算了27-20。
所有23次候选变化的原题、前后候选、答案与判分保存于audit_changes.jsonl，其中22次不影响正确数。
审查题不注入训练复核页面，也不根据此例追加硬编码规则。

六、资源与检查
主实验推理资产合计{f['total_primary_inference_bytes']} bytes，小于16MiB；尚未进行最终发布打包。
主核模块734169 bytes；神经长版约322411 bytes。文件全部保留。
本机4线程，950题，进程与输入已加载：基线{a['local_warm_cpu_timing']['baseline_warm_seconds']:.2f}秒，核模块含门控额外{a['local_warm_cpu_timing'][primary['name']]:.2f}秒，长文本模块额外{a['local_warm_cpu_timing']['regularized_long']:.2f}秒。
这是热运行局部计时，不是包含冷启动和独立评分的官方资源认证。
各训练记录计时合计约{training_seconds:.1f}秒；包含部分评估与预处理，两条路线曾并行运行，因此不能当作总墙钟时间。
训练使用既有RTX A2000 Laptop GPU/PyTorch环境；审查预测全部在CPU、4线程运行。
分组隔离、旧审查不复用、CPU导出、候选重排、相同候选零差值及零对齐偏置退化对照均通过。
原主模型16个资产以及第18轮所有冻结内容哈希均保持不变。详见checks.json与frozen.json。

七、文件与复现
训练和推理源码位于Project2/research/round19_*.py。主要入口：
  round19_common.py：划分登记与数据访问保护。
  round19_baseline.py：匹配基线重训。
  round19_neural.py：神经对照；会读取已登记的amendment_01.json。
  round19_kernel.py：核与线性对照。
  round19_evaluate.py freeze / audit：先冻结，再做一次审查。
  round19_diagnostics.py：冻结后的描述性诊断，不更新模型。
  round19_write_report.py：保存报告及候选登记。
当前实验目录已封存，训练、冻结、审查有重跑保护。复现应在独立项目副本进行，保留历史依赖、官方TRAIN文件、原plan.json与amendment_01.json，不删除或覆盖本目录。
已有可用解释器是D:\\anaconda3\\envs\\mp1\\python.exe；设置PYTHONPATH为对应Project2\\.venv\\Lib\\site-packages，PYTHONIOENCODING=utf-8后依次执行上述训练入口。
副本重建时先只放原plan.json与amendment_01.json，再运行baseline、neural、kernel、freeze、audit、diagnostics、write_report。
数据访问保护拒绝官方validation和test。当前入口未启用官方验证通道，不应改成运行旧的全流程脚本。
具体Python、NumPy、PyTorch与scikit-learn版本见posthoc_diagnostics.json；划分、种子、配置和SHA256见plan.json及冻结文件。

八、下一阶段决策依据
保留两类有小幅信号的候选：预选核模型，以及完整候选的联合比较器。它们是研究候选，不是新主模型。
后续适合固定配置、按题目交叉拟合复验修复/误伤方向，再决定是否安排有限的官方验证比较；不能反复用本轮950题选最好的版本。
本轮尚未获得通向95%的充分证据，也没有证明95%不可能。
代码由Codex协助实现；使用官方训练材料及普通开源数值/机器学习库，没有下载额外预训练模型、外部题库或调用推理API。
本文件是实验记录，不是最终十页课程报告或提交结果。
'''
    report=OUT/'95%阶段优化_第19轮报告.txt';report.write_text(text,encoding='utf-8',newline='\n')
    paths=[q for q in OUT.rglob('*') if q.is_file() and q.name!='completion_manifest.json']+list((ROOT/'research').glob('round19_*.py'))
    completion=dict(status='completed_candidates_retained_no_validation',gate_passed=a['train_side_gate_passed'],
        primary_candidate_retained=True,secondary_candidates_retained=True,incumbent_unchanged=True,
        validation_opened=False,test_opened=False,report=str(report.relative_to(ROOT)),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hashes={str(q.relative_to(ROOT)):sha(q) for q in sorted(paths)})
    dump(OUT/'completion_manifest.json',completion);print(str(report));print('Candidates retained; no model promoted or discarded.')


if __name__=='__main__':main()
