"""Record the reviewed scope, measured failure, deployment and reproduction."""
from data_access import ROOT
import hashlib
import json
import platform
import shutil
import sys
from importlib.metadata import version

OUT=ROOT/'experiments/round14'


def main():
    training=json.loads((OUT/'training.json').read_text())
    audit=json.loads((OUT/'audit_result.json').read_text())
    smoke=json.loads((OUT/'local_smoke.json').read_text())
    plan=json.loads((OUT/'plan.json').read_text())
    historical=json.loads((ROOT/'experiments/round12/validation_result.json').read_text())
    sources=[dict(title='CogComp arithmetic',url='https://github.com/CogComp/arithmetic',use='Research reference: relation/relevance and structured inference; no code, data or weights imported.'),
             dict(title='FinQA evaluator',url='https://github.com/czyssrs/FinQA/blob/main/code/evaluate/evaluate.py',use='Research reference: executable program evaluation; official MP2 parser remains unchanged.'),
             dict(title='iGSM',url='https://github.com/facebookresearch/iGSM',use='Research reference only. Generative augmentation deferred; no vocabulary, templates, data or model imported.')]
    (OUT/'research_sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    source=ROOT/'research'
    environment=dict(python=sys.version,platform=platform.platform(),packages={k:version(k) for k in ('numpy','scipy','scikit-learn')},seed=plan['seed'],threads=4)
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    text=f'''第14轮：交接方案审查、局部证据试点与本地部署
完成日期：2026-10-15；分组种子：20261014。

结论
已完成首轮可执行试点和部署检查，尚未实现交接文档的完整研究路线。
本轮新模块未获得自然题提分证据，部署为诊断模式；最终选择仍由原 round4 基线输出。
历史最佳验证记录 {historical['baseline']['correct']}/{historical['baseline']['n']} = {historical['baseline']['accuracy']:.4f}%，冲突子集307/355。
这些是既有日志中的成绩，本轮未重新读取或评测验证集。测试集始终未访问。
99%以上需要至少791/798，距历史最佳仍差41题；本轮不能声称缩小了差距。

方案审查
1. 赞同“先定位候选分歧、核对原题局部证据、不确定时弃权”的主假设。
   它是待验证路线，开源项目本身不提供本题库99%的保证。
2. 原仓库已有 round5_age_graph.py 年龄全图求解器；直接扩规则不构成新的学习能力。
   本试点改成局部候选声明核验，允许部分系统，不要求整题所有实体均能求解。
3. 原文字符位置正确不等于语义提取正确。最终数值正确也不等于候选中间推理全部正确。
   两类监督和指标已分开；未知或欠定关系不当作错误。
4. 新分组仅隔离新模块。原基线已用全部训练数据训练，旧轮次也查看过部分训练题。
   本轮不是端到端独立泛化检验；没有拿历史OOF冒充新的嵌套外层评估。
5. 从零训练生成器原则上属于可研究路线，但应先验证语义和自然题收益。
   本轮先做确定性关系变化，不启动GRU/Transformer自回归生成；无需新增依赖或下载权重。
6. 仅核查参考项目的公开代码页面，不复制外部题库、外部模板、模型、词表或代码。
   旧人工复核中的争议继续保留；没有把参考答案自动当作语义裁判。
7. 本轮只实现显式人物年龄、比较关系和时间端点的窄试点。
   跨句人物指代、单价批次、总量/新增量的通用学习提取器仍未实现。

实际改了什么
round14_local.py：原文span、实体独立身份、关系两端时间、有理数线性约束、最小支持子集。
对给定候选中的局部年龄声明及可绑定目标的最终数值，返回entailed/contradicted/insufficient/ambiguous/inconsistent_source。
每题最多6个候选对；每对保留一个对齐分歧。关系端点有顺序，不合并不同人物的同值数量。
这里只检查被抽取的关系；缺失约束、生日条件和未知指代等情况会使证据不足。

round14_run.py：从零拟合逻辑回归，输入是已经抽取的结构匹配字段。
这是弱监督结构匹配对照，不是从原始文本学习人物/时间绑定的神经语义解析器。
训练正负标签来自规则提取的原题关系及受控变换；每个变换重新计算蕴含结果。
改变人物或时间不必然产生矛盾，因此二分类为“获支持/未获证明”，后者不能用作否定证据。
所有同源变体跟随原题分区；不用改写文本对应的旧embedding。

训练与审查结果
训练原题分组：拟合2883、开发934、审查938。按名字/数值归一化的完全相同模板分组。
该归一化不能保证识别所有同义近重复，故不声称完全消除模板泄漏。
拟合实际提取到可生成监督的45道原题，共454条关系变化样本。
开发14个原题产生177条变化样本；审查13个原题产生141条变化样本。

方法                           开发平衡准确率   审查平衡准确率
有原题关系证据的结构匹配器       100.00%          100.00%
去掉原题证据的匹配控制组          55.56%           54.18%
确定性规则教师                   定义这些弱标签，不能作为独立语义准确率对照

上表只测量对规则产出的结构标签的学习。输入本身有字段相等性，任务较容易，100%不代表自然语言理解或最终选择正确率。
两个模型在审查样本上都没有概率>=0.99的输出；概率未校准，不能把它当作可靠性保证。
局部模型资产650字节，无题目ID特征、答案查表或外部权重。

真实候选诊断（审查分区938题）
含已提取关系或明确未知情况：14题；已提取原题关系16条。
原文span校验32/32通过，仅代表字符串位置完整。
候选声明共45条：25条在抽取后的方程系统中被支持，20条证据不足。
可对齐分歧：仅1题、3个候选对，题目覆盖1/938 = {100/938:.3f}%。
其中12条最终目标声明全部证据不足；最终数值支持/冲突判定数量为0。
故最终判断与官方标签的一致率为未定义，不是100%；独立人工语义精确率也未定义。
开发侧仅3个最终判断、拟合侧8个最终判断与官方标签一致，样本极少且不能替代审查。

为什么停止扩大这一分支
可对齐自然候选分歧仅1题，低于预先规定的30题；新schema的独立人工语义标注为0。
没有证据支持重新训练昂贵的完整外层基线，也没有触发验证集比较条件。
主要瓶颈是自然文本覆盖和关系抽取，不是结构匹配器容量或训练次数。
本窄试点没有证成，也没有全面否定交接文档的主假设。扩大同一种规则变化只会增加容易样本。
下一步需要真实候选局部声明与原文证据的标注，覆盖人物别名、单价作用范围和时间两端；先提高覆盖和核验质量，再考虑排名训练。
即使新增2道人审完成，也不等于本轮语义门槛已通过，不能据此打开覆盖开关。

部署与运行
入口：research/round14_selector.py
配置及基线资产：experiments/round14/deployed_bundle
配置固定 diagnostic_only / override_enabled=false；若手动启用覆盖，入口直接报错，避免未经检验改答。
离线诊断入口：round14_selector.diagnose(bank)。学习对照模型仅离线实验使用，不是正式预测依赖。
原基线16个文件共15,625,842字节，逐文件SHA-256一致。
部署包（原基线+配置）共{smoke['assets_bytes']:,}字节，即{smoke['assets_mib']:.6f} MiB，低于16 MiB。
本地467道训练题冷启动进程耗时{smoke['cold_process_seconds']:.3f}秒；i7-11800H，CPU，总线程配置4。
该检查加载全训练输入档案再取467题；不是官方完整测试预测+评分，也不是参考CPU合规认证。
协议、与原基线一致、批大小一致、未知ID、空输入、候选重排数值不变等检查通过。
单元不变量覆盖精确分数、欠定系统、矛盾原约束、双端时间、歧义弃权、原文span与测试文件保护。
本轮没有验证集新分数，没有测试集分数，没有启用改答，因此r/d/w/c净增不作泛化估计。

复核页面
第三批新增2题，仅来自本轮开放的fit/development分区，不含audit问题。
15个开放待分析案例中，6个由助手作研究解释、6个倍数歧义继续mask、1个已有人工结果，剩2个交给用户。
助手研究解释没有写入本轮训练标签，独立性和来源记录在open_train_triage.json。
页面无标准答案、模型预测或推荐选项；前两批页面、用户答案和归档记录保留。
地址：http://127.0.0.1:8767/复核工作台.html

复现（PowerShell，在Project2目录）
已有冻结产物的检查与重新部署：
  .\\.venv\\Scripts\\python.exe research/round14_checks.py
  .\\.venv\\Scripts\\python.exe research/round14_deploy.py
  .\\.venv\\Scripts\\python.exe research/round14_report.py
本轮从干净checkout第一次运行顺序：
  .\\.venv\\Scripts\\python.exe research/round14_prepare.py
  .\\.venv\\Scripts\\python.exe research/round14_checks.py
  .\\.venv\\Scripts\\python.exe research/round14_run.py fit
  .\\.venv\\Scripts\\python.exe research/round14_run.py audit
  .\\.venv\\Scripts\\python.exe research/round14_deploy.py
  .\\.venv\\Scripts\\python.exe research/round14_review_pack.py
  .\\.venv\\Scripts\\python.exe research/round14_report.py
已有frozen.json时fit故意拒绝重训；下一轮应使用新目录和新计划，不能静默复用已打开的审查分区。
本轮不提供或运行验证/测试命令，因为语义门槛失败。

资料与AI披露
官方规则：DASE7506-MP2-main/guide/GUIDE.md（本地内容核查）。
CogComp arithmetic：https://github.com/CogComp/arithmetic
FinQA evaluator：https://github.com/czyssrs/FinQA/blob/main/code/evaluate/evaluate.py
iGSM：https://github.com/facebookresearch/iGSM
这些页面为研究参照，不是本轮性能证据。没有引入其代码、数据或权重，故没有声称锁定或部署其commit。
代码、实验设计、中文题目说明和报告由Codex辅助完成。无外部LLM教师API调用；规则产生的弱标签明确标记。
最终课程报告仍需限制10页；本文件为研究运行记录，不替代提交报告。
'''
    (OUT/'方案审查与运行报告.txt').write_text(text,encoding='utf-8')
    files=sorted(source.glob('round14_*.py'))
    frozen={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT/'delivery_source_hashes.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    handoff=__import__('pathlib').Path('C:/Users/besti/Downloads/DASE7506_MP2_Research_GPT_Handoff_ZH.txt')
    if handoff.exists() and not (OUT/'research_handoff_source.txt').exists():shutil.copy2(handoff,OUT/'research_handoff_source.txt')
    print('Saved report, source hashes, environment and provenance.')


if __name__=='__main__':main()
