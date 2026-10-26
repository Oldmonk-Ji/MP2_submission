"""Final RLVR experiment handoff and immutable source/artifact manifest."""
from data_access import ROOT,CODE
from round22_common import read,dump,sha
import ast,datetime

OUT=ROOT/'experiments/round45'
def main():
    selection=read(OUT/'selection.json');verification=read(OUT/'verification.json');diag=read(OUT/'reward_diagnostics.json');integrity=read(OUT/'history_integrity.json')
    chosen=selection['selected'];train=read(OUT/'training_complete.json');plan=read(OUT/'plan.json');coverage=diag['training_reward_coverage']
    audit=read(OUT/'reward_audit.json')
    table='\n'.join(f"  {r['name']}: {r['correct']}/798 ({r['accuracy']:.4f}%)；修复{r['repairs']}，误伤{r['harms']}" for r in selection['records'])
    timing='\n'.join(f"  {r['name']}: {r['asset_MiB']:.4f} MiB；新进程预测{r['prediction_seconds']:.3f}s + 评分{r['scoring_seconds']:.3f}s = {r['total_fresh_process_seconds']:.3f}s；本机低于30秒={r['local_under30']}；接口抽查{r['checked_cases']}题通过。" for r in verification)
    held='\n'.join(f"  {r['level']}/{r['arm']}/{r['mode']}: {r['correct']}/1582；修复{r['repairs']}，误伤{r['harms']}" for r in read(OUT/'diagnostic_result.json'))
    signals=[]
    for r in diag['rollout_training']:
        if r['stage']!='full':continue
        f,l=r['first'],r['last'];signals.append(f"  {r['level']}/{r['arm']}: 两种奖励都出现的采样组占比 {100*f['mixed_group_fraction']:.2f}%→{100*l['mixed_group_fraction']:.2f}%；训练期望奖励 {f['expected_reward']:.4f}→{l['expected_reward']:.4f}；最后KL={l['kl']:.5f}，clip比例={l['clip_fraction']:.5f}。")
    cover='\n'.join(f"  {r['level']}/{r['arm']}: 原模型错题{r['n']}道，动作空间含正确解{r['oracle']}道；策略首选动作正确{r['top_action_correct']}道；期望奖励{r['expected_verified_reward']:.4f}。" for r in diag['held_reward'] if r['group']=='baseline_errors')
    current='保留第25轮冻结模型' if chosen['name']=='control' else f"验证集选择{chosen['name']}，实验包为experiments/round45/selected_bundle；参考机器时限尚待认证"
    report=f'''RLVR与在线GRPO消融：第45轮，2026-10-21

结论
{current}。选中方案完整官方验证集为{chosen['correct']}/798（{chosen['accuracy']:.4f}%）。没有读取测试集或提交排行榜，没有证据声称最终测试准确率达到97%。
本轮实际新增当前策略在线采样、组内相对优势、重要性比例裁剪的GRPO训练。上轮第44轮reward已经属于可验证奖励优化，只是有限空间上的精确期望奖励；本轮不把旧结果重新命名为新实验。

完整验证集（所有预注册方案均报告，未筛减评测题）
{table}
候选层直接替换原第25轮两个ground成员，权重保持0.125，其余六成员不变。关系层两种固定部署为首选程序执行及0.25候选概率残差。程序覆盖和执行器沿用第44轮，未根据新验证结果修改规则。

算法与对照
两层均有exact和grpo；关系层另有grpo_relations。exact直接计算所有合法动作的期望正确性奖励；grpo每题从当前批次更新前策略采样16个动作，按组内奖励均值/标准差计算优势。采样行为策略每批刷新；参考策略固定为共同SFT起点。两者不是同一个对象。
每个采样批次进行2次更新，重要性比例clip到[0.8,1.2]；KL系数0.02，使用有限类别分布的精确KL，不使用语言token的采样KL估计。每组10个训练轮，batch64，AdamW weight_decay=0.01，梯度裁剪5。候选层学习率0.0001，关系层0.0003；两种种子分别25025/25026和44044/44045。
同层各臂共享起点、分组、batch顺序、优化器预算与学习率；继续训练阶段全部关闭dropout，避免随机掩码污染重要性比。exact臂也采样以记录相同类型的诊断，但更新使用精确期望。无参数扫描、没有按验证选择中间epoch。
同组全部成功或失败时，优势为0，不伪造负例、不动态剔除问题；仍保留KL约束。有些训练题因此没有该步策略梯度，这是奖励信息不足，不是额外删题。
候选层完整训练共同起点是第25轮epoch35；诊断起点在fit折重新训练35轮SFT/EMA并保存。关系层共同起点是第44轮sft30轮模型，诊断/full使用各自对应分组。这是对该已训练策略的进一步RLVR，并非与第44轮10轮起点的直接等预算比较。
这是一动作有限决策问题上的GRPO式RLVR，不是自回归大语言模型微调，也没有自主生成新推理文本或扩展程序空间。GRPO来源：https://arxiv.org/abs/2402.03300

奖励和关系监督
候选动作奖励：官方解析的候选最终数值等于官方TRAIN参考答案则1，否则0。关系动作奖励：Fraction执行结果等于TRAIN答案且能选择原候选则1，否则0。参考答案只用于训练奖励，不进入推理特征。
grpo_relations额外使用系数0.2的来源明确关系偏好约束。沿用第44轮32道审查题、249组关系对，包括最终数值相同但数量对象绑定错误的对照；两者正确性奖励相同，关系约束分别处理语义是否一致。监督为助手来源审查和程序化负例，不宣称独立人工金标。
所有最终模型使用4,743道批准训练题；仅沿用此前批准排除的12题。诊断按父题与模板分组，fit3,161，held1,582；held父题的所有关系对不参与诊断模型拟合。分组及历史关系模式曾用于研究，诊断不是完全未触碰的研究样本。官方验证798题已多次复用，只用于开发与选方案，不能当作无偏泛化估计。

训练内留出效果
{held}
候选层诊断沿用既有OOF基线加固定0.25归一化候选评分；完整验证采用固定成员替换，两者应分别解读。诊断共同SFT起点为1475/1582，GRPO为1476，故相对共同起点只多1题；不能把对OOF基线的3题差额都归因于RL。
留出关系检查只有3道父题的21组相关约束，不是21道独立题；其局部偏好正确率不应替代完整候选选择准确率。
原模型109道留出错题上的动作覆盖与奖励（不是验证集调参）：
{cover}
候选动作空间始终包含正确候选；关系动作空间仍受解析和32程序上限约束。在线采样只能在已有动作内重新分配概率，不能凭奖励创造缺失的正确程序。
4,743道训练题中，{coverage['all_four_correct']}道四个候选全部正确，{coverage['candidate_mixed_correctness']}道同时有正确/错误候选。关系层有{coverage['relation_has_no_correct_action']}道题在固定动作空间中没有正确程序。这些结构性零信号组与模型过于确信造成的零信号应当区分。

在线采样信号（完整训练，两种种子均值；首轮→末轮）
{chr(10).join(signals)}
混合奖励比例接近零时，GRPO主目标没有区分信号。比例下降也可能来自策略趋于确定，不能单独解释为准确率提升。必须结合完整验证结果和留出错题改善判断。

接口与资源
独立抽查{audit['train_questions_checked']}道TRAIN题并重新执行{audit['programs_reexecuted']}个程序，确认候选奖励与官方解析一致、关系奖励与确定性执行结果及TRAIN参考一致。
所有8个实验部署均由未修改官方score_predictions.py独立评分。候选层及关系层各选一个最佳实验包做完整新进程predict.py加score_predictions.py复现；与缓存逐题一致。批量/分块、未知ID、候选置换后的数值答案、空输入和相同候选检查通过。
{timing}
本机Intel Core i7-11800H，所有CPU库合计设置4线程。作业参考Xeon Platinum8457C上的完整测试预测加评分才决定30秒资格；本机验证计时不等价于参考机器认证。资产低于16MiB不意味着已满足时间资格。
本轮训练耗时{train['seconds']:.2f}秒，设备{train['device']}；仅一张RTX A2000 Laptop GPU，无新预训练权重/外部训练集/推理API调用。普通Python库、NumPy/SciPy/PyTorch为现有环境。
规则来源：https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md

复现与冻结
工作目录E:/New Life/HKU/7506/Project2，Python D:/anaconda3/envs/mp1/python.exe；PYTHONPATH含research与.venv/Lib/site-packages，PYTHONIOENCODING=utf-8。
顺序：python research/round45_rlvr.py prepare；python research/round45_rlvr.py run；python research/round45_reward_audit.py；python research/round45_evaluate.py；python research/round45_analyze.py；python research/round45_verify.py；python research/round45_report.py。
prepare有防覆盖检查，不应直接在已有产物目录重新训练；重现实验应在副本/干净输出目录进行。
关键输出：experiments/round45/plan.json、training_complete.json、selection.json、diagnostic_result.json、reward_diagnostics.json、verification.json、environment.json（实际Python/NumPy/SciPy/PyTorch/CUDA版本）；每个模型保留训练ID、每轮损失/奖励/组内方差/KL日志。
历史{integrity['prior_manifests']}个完成清单的{integrity['distinct_prior_files']}个不同文件均保持原哈希；本轮训练前源码和输入哈希仍一致。完成清单同时冻结本轮产物及本地Python源码依赖闭包。
本轮代码、实验设计、文档由AI辅助完成；最终报告需披露此实质帮助并由提交者理解、复核。
'''
    path=ROOT/'RLVR在线采样消融_2026-10-21_第45轮.txt';path.write_text(report,encoding='utf-8')
    dump(OUT/'summary.json',dict(completed=True,goal_97_achieved=False,test_opened=False,selected=chosen,
        report=str(path.relative_to(ROOT)),reference_machine_certified=False))
    files={path};queue=list((ROOT/'research').glob('round45*.py'))+[CODE/'predict.py',CODE/'score_predictions.py'];seen=set()
    while queue:
        p=queue.pop()
        if p in seen:continue
        seen.add(p);files.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[v.name.split('.')[0] for v in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                for directory in (ROOT/'research',CODE):
                    q=directory/f'{name}.py'
                    if q.exists():queue.append(q);break
    files.update(p for p in OUT.rglob('*') if p.is_file() and p.name!='completion_manifest.json')
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    dump(OUT/'completion_manifest.json',dict(completed=True,goal_97_achieved=False,test_opened=False,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_closure_files=len(seen),hashes=hashes))
    print(dict(report=str(path),frozen_files=len(files),selected=chosen),flush=True)

if __name__=='__main__':main()
