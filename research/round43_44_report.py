"""Write reproducible experiment handoff after all official checks pass."""
from data_access import ROOT,CODE
from round22_common import read,dump,sha
import ast,datetime

def main():
    r43=ROOT/'experiments/round43';r44=ROOT/'experiments/round44'
    a=read(r43/'selection.json');b=read(r44/'selection.json');v=read(r44/'verification.json');d=read(r44/'postexperiment_train_diagnostics.json')
    assert v['official_prediction_matches_cache']
    table43='\n'.join(f"  {r['arm']}: {r['correct']}/798 = {r['accuracy']:.4f}%；修复{r['repairs']}，损伤{r['harms']}" for r in a['records'])
    table44='\n'.join(f"  {r['name']}: {r['correct']}/798 = {r['accuracy']:.4f}%；修复{r['repairs']}，损伤{r['harms']}" for r in b['records'])
    rel='\n'.join(f"  {r['stage']}/{r['arm']}/{r['group']}: {r['correct']}/{r['pairs']}" for r in d['relation_pair_accuracy'])
    cov='\n'.join(f"  {r['arm']}/{r['group']}: {r['n']}题，可构造程序{r['proposal_n']}，程序中有正确解{r['correct_program_available']}，策略首选程序正确{r['top_program_correct']}" for r in d['held_program_coverage'])
    report=f'''DPO与关系策略消融：第43至44轮，2026-10-21

一、结论和数据边界
完整验证集结果见下表；第44轮选择为 {b['selected']['name']}。当前验证集最优仍为第25轮752/798（94.2356%），本轮没有证明最终测试集达到97%。没有打开测试输入或参考答案，也没有提交排行榜。
用户要求测试集不参与调参，因此开发阶段使用训练集、训练内分组留出和官方验证集；最终方案冻结后才进行测试评估。验证集已多轮复用，其分数是开发证据，不是无偏泛化估计。此前的94.2356%是验证集分数，不应写成当前测试成绩。
最终模型使用全部4,743道允许训练题（原4,755题减去此前批准的12题），没有另行剔除困难题。诊断按既有模板分组：3,161题拟合，1,582题留出；派生关系样本跟随父题分组。既有训练题和关系模式曾被研究，不能将这次留出称为完全未见的研究数据。

二、第43轮：DPO消融
{table43}
相同ground特征、相同MLP架构、种子25025/25026、相同初始模型、批次及20轮更新。完整训练以第25轮35轮SFT模型为共同起点；诊断起点重新在3,161题上训练，排除留出题。学习率0.0001、weight decay 0.01，无温度/融合权重/轮数扫描。
SFT为正确候选均匀交叉熵加0.25 BCE；pairwise使用正负候选分数差的logistic损失；DPO减去冻结参考模型的同一正负分数差，beta=1。正确/错误对在题内归一化；全正确候选题三组均回退相同SFT损失，不造假负例。只替换原模型中的两个ground成员，原权重均为0.125。
诊断融合正确数：共同起点1475，SFT1474，pairwise1476，DPO1475；原OOF对照1473/1582。诊断的小幅变化没有转化为完整验证集收益。
这是有限候选选择器上的DPO目标消融，不是对外部预训练大语言模型做DPO。方法来源：https://arxiv.org/abs/2305.18290

三、第44轮：学习中间关系，确定性执行后选择原候选
{table44}
把题干数量绑定到其具体出现位置；策略为最多32个可执行程序评分，程序包含对象绑定、分数范围、时间先后及运算方向。Fraction精确执行器计算结果，再映射到原四个候选索引。无新答案文本生成，无外部预训练权重，无ID查表。程序提案只依赖输入题干与候选，不使用参考答案；模型特征不包括参考值、执行最终数值或题目ID。
程序由候选算式及有限关系编辑构造，因此不是任意程序合成器。训练集中3,185/4,743题能构造程序，原程序正确解覆盖2,624题，编辑后2,842题。覆盖增加218是oracle上限变化，不是准确率提升。其余题可能没有程序，或程序空间中没有正确解。
固定1024维哈希题干/数量上下文/AST特征，小型1024→64→32→1策略MLP，种子44044/44045。共同10轮SFT预热，然后20轮：继续SFT、最终奖励、最终奖励加关系约束。所有模型学习率0.0003、weight decay 0.01，KL系数0.02，关系系数0.2；没有参数扫描。
最终奖励为程序执行值等于官方训练参考答案且可映射到原候选时取1，否则0。对有限动作空间精确计算期望奖励及梯度，不使用采样REINFORCE估计。不把候选参考答案注入推理阶段。
原审查事实编译成关系程序；训练前的检查修复了3处题干明确的单数a=1绑定（4537、201、1095），保留历史失败记录及冻结修订记录。最终32道审查题提供249组约束：对象126、方向108、范围9、时间6。它们来自有来源的助手审查及程序化负例，不是249条独立人工金标。
670和3635各含一组“最终数值相同但对象绑定错误”的对照：最终正确性奖励保持正值，只有关系约束惩罚错误绑定。这样不把数值偶合误当作语义正确。未知关系未被无条件补成标签。
两种预注册部署：execute_top执行首选程序，不能映射则回退原模型；fixed_residual把正确候选映射概率质量以固定0.25残差加入原模型。两者均保留原候选索引接口。

四、失败边界：训练留出诊断（没有据此再改规则或超参数）
{cov}
关系偏好正确数：
{rel}
留出关系仅3道父题、21组相关约束，三种策略都是19/21；不足以证明关系泛化提高。额外关系损失能在已有监督处改变分数，但当前证据没有显示它改善新题决策。数值奖励也不能恢复不在提案空间内的正确关系。
本轮区分了三个限制：程序覆盖不足；局部上下文/AST特征对语义角色的判别不足；关系监督规模及类型不均衡。不能把问题简化为DPO beta或RL学习率没扫够。若继续研究，优先扩展训练题的来源明确、按父题留出的实体/状态/范围监督，以及可验证的程序覆盖；不以测试错题补规则。

五、官方接口、资源与冻结校验
所有3个DPO消融和6个关系部署均由未修改官方score_predictions.py在完整798题验证集复核。第44轮独立新进程运行实际predict.py，输出与缓存逐题一致；批量/分块、未知ID、候选置换后的数值答案、空输入和重复候选检查通过（{v['checked_cases']}个抽查/变化案例）。
被复现的实验包：{b['evaluation_bundle']}，方案 {v['evaluated_name']}；它只是实验复现包，不代表替代当前最优。
资产大小{v['asset_bytes']:,}字节 = {v['asset_MiB']:.4f} MiB，低于16 MiB。
本机Intel Core i7-11800H，4线程，独立预测进程{v['prediction_process_seconds']:.3f}秒，独立官方评分{v['scoring_process_seconds']:.3f}秒，合计{v['total_fresh_process_seconds']:.3f}秒。本机是否低于30秒：{v['local_under_30_seconds']}。这不等于在作业指定Xeon 8457C上通过30秒认证；不能宣称已获得参考机器时限资格。
官方作业要求来源：https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md
历史17个完成清单中的775个不同文件哈希仍一致；第43轮计划源码及第44轮训练前冻结哈希一致。新增代码和产物写入独立完成清单。第25轮最优模型保持原样。

六、复现入口与产物
Python: D:/anaconda3/envs/mp1/python.exe
工作目录: E:/New Life/HKU/7506/Project2
PYTHONPATH: research及.venv/Lib/site-packages；所有CPU库线程数4。
训练脚本（已运行，部分会拒绝覆盖既有冻结产物）：research/round43_dpo.py；round44_prepare.py；round44_checks.py；round44_train.py。
验证：python research/round43_44_verify.py
训练错误诊断：python research/round44_analyze.py
关键清单：experiments/round43/selection.json、experiments/round44/selection.json、verification.json、postexperiment_train_diagnostics.json、training_freeze.json。
当前仍应使用research/round25_selector.py + experiments/round25/selected_bundle。
'''
    path=ROOT/'DPO与关系策略消融_2026-10-21_第43至44轮.txt';path.write_text(report,encoding='utf-8')
    summary=dict(completed_ablation=True,goal_97_achieved=False,test_opened=False,incumbent='round25',validation_correct=752,validation_n=798,
        round43=a,round44=b,verification=v,report=str(path.relative_to(ROOT)))
    summarypath=ROOT/'experiments/round43_44_summary.json';dump(summarypath,summary)
    # Freeze the transitive local Python source closure, not only entrypoints.
    files={path,summarypath};queue=list((ROOT/'research').glob('round43*.py'))+list((ROOT/'research').glob('round44*.py'))+[CODE/'predict.py',CODE/'score_predictions.py'];seen=set()
    while queue:
        p=queue.pop()
        if p in seen:continue
        seen.add(p);files.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                for directory in (ROOT/'research',CODE):
                    q=directory/f'{name}.py'
                    if q.exists():queue.append(q);break
    for out in (r43,r44):files.update(p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    dump(ROOT/'experiments/round43_44_completion_manifest.json',dict(completed=True,goal_97_achieved=False,test_opened=False,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_closure_files=len(seen),hashes=hashes))
    print(dict(report=str(path),frozen_files=len(hashes),source_closure_files=len(seen)),flush=True)

if __name__=='__main__':main()
