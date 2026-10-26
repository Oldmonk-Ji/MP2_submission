"""Archive completed TRAIN/VAL experiments; never read TEST inputs."""
from data_access import ROOT, CODE
from round22_common import read, dump, sha
from pathlib import Path
import datetime

def main():
    results={str(i):read(ROOT/f'experiments/round{i}/validation_result.json') for i in (47,48,50)}
    verify={str(i):read(ROOT/f'experiments/round{i}/verification.json') for i in (47,48,50)}
    guard=read(ROOT/'experiments/round50/unknown_guard_ablation.json')
    best=max((dict(r,round=int(i)) for i,rs in results.items() for r in rs),key=lambda r:r['correct'])
    assert best['correct']<752
    summary=dict(experiments_completed=True,goal_97_achieved=False,test_opened=False,
        selected='experiments/round25/selected_bundle',selection_basis='Full official VAL only; retain 752/798 incumbent',
        incumbent=dict(correct=752,n=798,conflict_correct=309,conflict_n=355),best_new=best,
        validation=results,verification=verify,unknown_guard_ablation=guard,
        coverage=dict(all_n=4743,original=2842,graph=3603,parser_fixed=3987,
                      held_error_n=109,original_errors_covered=47,graph_errors_covered=56,parser_fixed_errors_covered=71),
        supervision=dict(parents=86,pairs=706,same_value_pairs=10,new_held_parents=24,new_held_pairs=209,
                         provenance='Assistant source review; not independent human gold'),
        resource_limitations='Local full-VAL timings are not reference-Xeon full-TEST certification. Round48 timing overlapped a TRAIN diagnostic.')
    dump(ROOT/'experiments/round50/summary.json',summary)
    rows=[]
    for i,rs in results.items():
        for r in rs:
            rows.append(f"第{i}轮 {r['arm']:5s}: {r['correct']}/798 ({r['accuracy']:.4f}%), 冲突题 {r['conflict_correct']}/355, 修复{r['repairs']} / 误伤{r['harms']}")
    times='\n'.join(f"第{i}轮: 资产{v['asset_MiB']:.4f}MiB；完整798题新进程预测+评分{v['total_fresh_process_seconds']:.3f}秒。" for i,v in verify.items())
    report=f'''关系监督、关系图执行器与 SFT / RLVR 消融：第46—50轮
日期：2026-10-21

结论
已完成本轮补充监督、程序覆盖修复、关系图执行和三组匹配消融。未达到97%，新模型没有超过原模型，保留 round25/selected_bundle（完整验证集752/798，94.2356%；冲突题309/355，87.0423%）。本轮没有打开测试集、没有产生新的测试成绩，也没有根据测试错题制定规则。历史测试成绩不作为本轮选择依据。

1. 真实训练错题与监督来源
在历史训练OOF错误的fit folds0+1内，用固定哈希顺序抽取60题，先依据题目逐句写数量来源、表达式和证据，再运行训练参考答案比对。54条表达式全部得到训练参考数值；另6题存在离散整除、舍入、指代、日历或比较语义歧义，不强行编造过程标签。这6题仍参加普通训练。
新增54个过程监督父题，加旧32个，共86题706对关系约束：对象绑定340、运算方向285、分数范围22、时间状态59。保留10对最终数值相同但对象绑定错误的约束，不能用终局奖励替代它们。第46轮试运行曾漏掉旧249对中的2对同值约束，审计后中止该试运行，恢复全部旧监督并增加8对源文本审查的同值约束，第47轮重新正式训练；未将46轮当正式验证实验。
另在历史fold2固定抽24题（12个原错误、12个正确对照），得到209对关系诊断，24条表达式均匹配训练答案。这24题的新增过程标签不进入任何诊断或最终模型的过程梯度；最终模型仍按要求使用全部4743题的普通答案监督。标签由助手依题目审查，不是独立人工金标；训练资料历史上已探索过，不能称完全盲测。

2. 关系图与确定性执行
数量保留源文本出现位置，区分对象、单位、百分数/分数、前后状态，建立候选算式依赖图。精确有理数执行器执行程序，再映射到原四个候选。没有用参考答案向候选程序库插入正确程序，没有生成提交用的新答案。图边仍是词法及结构假设，并非已解决自然语言理解的解析器。
正确程序覆盖从2842/4743提升到3603/4743，再到3987/4743（84.0607%）。历史训练留出1582题中的109道基线错题，覆盖47→56→71；仍有38道缺失。固定32动作时新图已覆盖3593题，64动作仅增加10题且没有多覆盖留出错题，主要收益不是扩大动作预算。
从fit部分8道未覆盖错题继续追查，发现特征抽取器把“3 * 12 = 36 cans”一类带空白分隔单位的等式丢弃，还存在Markdown列表负号、纯单位LaTex分数和显式百分比比较处理问题。第49轮仅修复候选算式特征抽取，官方数值答案解析器和评分器完全未改。补回384题正确程序，其中留出错题15题。新增程序保留全部旧程序，64动作上限不变。第49轮只做覆盖准备，无额外训练或官方验证试验。

3. 学习目标与公平对照
第47轮整程序关系表示：2048维哈希特征、64/32隐藏层；第48轮改为局部关系1024维、32隐藏层，用固定soft-min聚合程序各节点，并在实际变更的关系位置施加过程约束。706对中705对局部表示可区分，剩余1对未丢弃。
每轮两固定种子，共享20轮SFT预热，然后比较10轮继续SFT、精确期望可验证奖励、GRPO。批32，学习率0.0003，权重衰减0.01，KL0.02，关系约束权重0.2，GRPO每组16采样、clip0.2、每批两次更新。这里是有限程序动作空间上的策略学习，不是预训练语言模型微调。终局奖励由执行数值与训练参考答案决定；相同终局数值的错误关系靠过程约束区分。
第50轮共享第48轮已冻结的SFT预热权重，在修正后的合法动作集合重算参考分布。仅保留能执行并映射到原候选的动作：73729个程序中19959个可选，532题没有合法程序，仍保留题目及适用的过程监督；无动作的策略项不产生虚假训练信号。部署时若任一可解析候选答案没有程序覆盖，则回退基线。缺失解释表示未知，不当成反对该答案的证据。固定残差系数0.25，不扫描门槛或权重。RL优化程序终局奖励，并非直接优化部署融合后的argmax。

4. 完整官方验证集（每次798题，未删除题目）
原模型: 752/798 (94.2356%), 冲突题309/355。
{chr(10).join(rows)}
三个正式轮次均未胜出，因此不替换原模型。第50轮最好749/798，93.8596%，冲突题306/355，86.1972%，修复0、误伤3。

5. TRAIN留出诊断与失败原因
固定残差下，原基线1473/1582；第47轮三臂均1463，第48轮SFT1464、exact1462、GRPO1462。第47轮额外关闭过程监督的匹配SFT为1462，有监督1463，仅多1题，不能声称已证明强过程监督收益。
第50轮同一权重只切换“未知回退”保护：SFT1462→1469、exact1463→1470、GRPO1460→1467。误伤分别20→8、20→7、21→8，同时也损失部分纠错机会。保护减少错误干预，但并未提升到基线1473。
24个新增过程留出题的209对关系，第47轮SFT/exact/GRPO正确147/149/146，第48轮158/151/151，第50轮150/153/152。第50轮fit关系训练已接近满分，而留出关系只有约72%—73%，说明局部词法关系特征仍有明显泛化不足。尤其时间状态仅11—13/25。4个同值不同对象的小挑战，第47轮各模型4/4，第48轮2/3/3，第50轮2/2/2；样本极小，但不能据此宣称新局部模型已学会稳定的对象绑定。
第50轮留出原109错题虽有71题存在正确程序，SFT/exact/GRPO首选正确程序仅38/40/39。覆盖改善与正确排序是两个独立问题。程序输出恰好等于参考数值也不是语义过程正确的充分条件。
对“只能选择已执行程序，或回退基线”的当前路线，即使oracle选对，训练留出冲突子集也最多659/697=94.5481%；97%至少需要677题，仍缺18题可纠正覆盖。加完整覆盖保护后oracle只有631/697。此界限仅针对当前程序选择/回退架构及这个历史TRAIN子集，不是测试集上界，不适用于未来任意候选模型。

6. 后续优化应由哪些证据驱动
优先补齐剩余未覆盖错题的可验证程序类型：符号一元方程、整数/舍入语义、隐含单位和时间转换；先记录源文本许可的假设，对歧义题不把数值匹配当语义标签。新增变换先在fit题制定，再到固定训练留出检查覆盖与误伤。
关系监督应补充跨题对象绑定、时间状态、分数作用范围的结构化标注，尤其增加“同答案、错关系”的对照；在模板/关系组合隔离的训练留出上验证，不仅看同类词法匹配。监督质量和独立审查仍需提升。
目前不支持继续加大GRPO步数或扫描混合系数：训练奖励增加未转化为验证纠错。应先让正确程序覆盖和关系排序在训练留出超过当前基线，再进行相同预算SFT/RLVR比较。若考虑局部覆盖下的干预，需要从训练构造“有证据的反驳”判据，而不是再次把缺失程序当负证据。

7. 合规、复核和资源
最终模型使用全部4743个允许TRAIN样本，仅沿用此前获准的12题排除；没有新增训练删除，没有外部数据或预训练权重，使用作业提供的特征。每题返回原候选之一。新增held过程标签仅诊断，不用于过程训练。官方评分器未改，TEST访问审计钩子禁止打开测试原始数据和特征。
各轮均完成独立新进程官方验证复现、未知ID、分批不变性、候选置换答案不变性、空输入和重复候选检查，接口检查期间禁止读取参考答案。历史19个完成清单的文件哈希保持不变。
{times}
第48轮计时与TRAIN诊断有重叠，不能作为无争用性能认证。以上均是本地i7-11800H四线程的完整验证集计时，不是467题测试集计时，也不是参考Xeon8457C的30秒认证。新模型资产均低于16MiB；未胜出的模型不发布为最终提交方案。
正式训练进程墙钟时间：第47轮约552秒，第48轮约561秒，第50轮约370秒（复用48轮预热）；额外无过程监督对照约126秒，与其他任务部分并行，不能相加当总耗时。第46轮中止的试运行仍有成本，未纳入这些正式轮次时长。

8. 复现与证据路径
research/round46_annotations.py、round46_graph.py：监督与关系图。
research/round47_prepare.py、round47_run.py：完整监督修复与整程序消融。
research/round48_local.py、round48_train.py：局部关系决策消融。
research/round49_programs.py、round49_prepare.py：候选算式特征解析修复。
research/round50_train.py、round50_evaluate.py、round50_diagnostics.py、round50_verify.py：合法动作、未知保护及最终消融。
experiments/round50/summary.json：汇总；各轮validation_result.json及verification.json：官方分数和独立验证。
experiments/round46_50_completion_manifest.json：本轮产物和源文件SHA256。
可重复运行研究脚本需使用D:/anaconda3/envs/mp1/python.exe，PYTHONPATH包含本项目research和.venv/Lib/site-packages。单独运行research/round50_verify.py可重复官方完整验证和接口检查，且不访问TEST。训练与准备脚本有冻结保护，不应覆盖已完成目录。
'''
    report_path=ROOT/'关系监督与确定性执行器消融_2026-10-21_第46至50轮.txt'
    report_path.write_text(report,encoding='utf-8')
    # Verify frozen inputs and initial source hashes before sealing artifacts.
    for i in range(46,51):
        folder=ROOT/f'experiments/round{i}'
        for name in ('prepared_freeze.json','warm_start_freeze.json'):
            path=folder/name
            if path.exists():
                obj=read(path)
                for p,h in obj.get('hashes',{}).items():assert sha(ROOT/p)==h,p
        for p,h in read(folder/'plan.json').get('source_hashes',{}).items():assert sha(ROOT/p)==h,p
    # Snapshot all research source, including dynamic starter import, not datasets.
    files=list((ROOT/'research').glob('*.py'))+[report_path]
    files+=list((ROOT/'starter').glob('*.py'))+list((CODE/'starter').glob('*.py'))+list(CODE.glob('*.py'))
    for i in range(46,51):files += [p for p in (ROOT/f'experiments/round{i}').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files=sorted(set(files))
    manifest=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=True,
        goal_97_achieved=False,test_opened=False,selected='round25',
        hashes={str(p.relative_to(ROOT)):sha(p) for p in files})
    target=ROOT/'experiments/round46_50_completion_manifest.json';dump(target,manifest)
    for p,h in read(target)['hashes'].items():assert sha(ROOT/p)==h,p
    print(dict(report=str(report_path),files=len(files),selected='round25',best_new=best,goal_97_achieved=False),flush=True)

if __name__=='__main__':main()
