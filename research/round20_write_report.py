"""Report deletion evidence, paired outcomes and retained artifacts without selective omission."""
from round20_common import ROOT,OUT,plan,read,dump,sha,paired
from round20_evaluate import NAMES,HEADS,integrity
from collections import Counter
import datetime,json
import numpy as np

LABELS=dict(full='完整训练数据',evidence_only='仅移除有证据问题/歧义题',quality_05='有证据问题＋高不确定性，最多5%',quality_10='有证据问题＋高不确定性，最多10%',random_matched='按样本量、正确候选数与难度匹配的随机删除')


def main():
    p=plan();f=read(OUT/'frozen.json');a=read(OUT/'audit_result.json');s=read(OUT/'selection.json')
    assert all(sha(ROOT/k)==v for k,v in f['hashes'].items())
    q=read(OUT/'quality/summary.json');profile=read(OUT/'quality/deletion_profile.json');review=read(OUT/'quality/adjudication.json')
    runs={n:read(OUT/f'runs/{n}/done.json') for n in NAMES}
    rows=[]
    for n in NAMES:
        v=a['results'][n]['long'];r=runs[n]
        rows.append(f"{LABELS[n]}：保留{r['n_fit']}、移除{r['removed']}（{100*r['removed_fraction']:.3f}%）；开发{r['development']['long']['correct']}/650；审查{v['correct']}/950={100*v['accuracy']:.3f}%；修复{v['repaired']}、误伤{v['damaged']}、净增{v['net']:+d}。")
    heads=[]
    for n in NAMES:
        entries=[]
        for h in HEADS:
            v=a['results'][n][h]
            entries.append(f"{h}: {v['correct']}/950，修复{v['repaired']}、误伤{v['damaged']}、净增{v['net']:+d}")
        heads.append(LABELS[n]+'；'+'；'.join(entries))
    primary=f['primary'];v=a['results'][primary]['long'];r=a['primary_vs_matched_random'];ci=v['bootstrap_95_ci_percentage_points']
    if a['positive_development_and_audit_gain']:
        conclusion='预先选定的清理版本在开发与审查两侧均有正净增，可以保留为后续固定配置复验候选。'
    else:
        conclusion='预先选定的清理版本没有在开发与审查两侧同时取得正净增。本轮不能据此把清理训练替换为默认方案。'
    random_note='本次审查也优于匹配随机删除。' if r['net']>0 else '本次审查没有优于匹配随机删除，不能将变化归因为质量筛选的独有收益。'
    text=f'''第20轮：训练数据质量筛选与同方法重训
日期：{datetime.datetime.now().date()}；目标是检查少量剔除是否有助于接近95%，不是保证达到95%。

结果
{conclusion}
{random_note}
所有模型与筛选清单均保留，原主模型未覆盖。官方验证集与测试集均未打开。
原主模型的历史官方验证结果仍是750/798（93.98496%）；本轮训练侧审查分数不能冒充官方验证分数。

实验边界
官方4755道训练题按归一化题干模板分为3155拟合、650开发、950封存审查。
同题四个候选保留在同一分区，相同归一化模板不跨分区。第18、19轮已经审查过的1900题只进入本轮拟合区。
这保证本轮训练隔离，不能消除项目长期探索过训练语料的历史影响，也不等同于新的外部基准。
开发与审查各题完整保留，不删除错误题、不改标签、不调整官方解析器或数据文件。
最多删除拟合区10%，即315题。实际最大删除314题（{314/3155*100:.3f}%），没有为了凑10%继续删题。

筛选方法
仅在3155道拟合题内部做三折按题目模板分组的交叉预测。
三种现有固定模型（表示MLP、词语与表示联合模型、词语排序器）分别用另外两折训练，词表、标准化、参数也只拟合那两折。
依据未见过该题的模型分歧、正误候选排序和标准化分差计算风险；风险分数不是经校准的错误概率。
普通错误候选是任务必需的负例，不因候选算错就删题。多正确候选不被互相当成负例。
额外审阅风险最高60题的题干、参考解与所有数值正确候选，并合并当前拟合区内既有人工复核7题。
67条检查记录中：18条有具体缺陷或歧义证据，46条确认有可用正确推理并保护保留，3条文字冲突尚不确定。
新检查由助手完成，不能冒充独立人工金标。既有人类回答原文未修改。amendment_01.json记录了在比较筛选版本前加入保护清楚难题的规则。
最终314条可疑/不确定题中仅18条有明确记录的缺陷或歧义证据，其余296条只是模型不确定，不能称为296条错标签。
参考解步骤只用于质量核查，没有作为预测器输入或新增训练目标，没有生成题、外部数据或额外预训练权重。

具体证据
train 1341：题干6个四年级班，参考解按4个班算；还隐含第三周保持相同收集速度。作为题意/参考解不一致排除，不擅自改标签。
train 217：唯一数值正确候选把逐年累计存钱当成7枚硬币，输出7碰巧命中美元总数；数值标签符合官方评分，推理却不支持它。
train 3421：唯一数值正确候选声称周末5天，用24/10=2.4后写向上取整为2；终值碰巧正确。
train 2102：唯一数值正确候选把124-58+24算成100，又漏加10名机组，错误抵消得到67。
train 3696：三个数值正确候选都写7+12+15-2=30；实际应去除4次重复，官方终值30并不证明候选推理正确。
train 2259：所有正例都把Alice的3600颗橡果绑定给Bob，却碰巧得到官方数字15。
相反，train 2502、947、70等清楚的难例即使三模型都选错，仍被保护保留。
这18条混合了参考解冲突、正例推理错误和题意歧义；它们不是18个已证实错误的官方数值标签。

删除规模与难度分布
原拟合区有1374道不同有效答案相互冲突的题。
只删18道组删除17道冲突题；最多5%组删除156道；最多10%组删除311道，占原冲突题的{311/1374*100:.2f}%。
因此删除总样本约10%会使关键难题减少约22.6%，不能把这个实验简单理解为只去掉了少量无用数据。
随机对照与开发选定的清理组在删除总数、正确候选数分布、输入冲突难度上完全匹配。

训练与公平比较
每一组均从头训练原来的五个基线成员，再训练第19轮保留的长文本联合比较器和固定核模型。
所有词表、PCA、标准化、基点和权重只从该组保留题学习，绝未沿用看过新审查题的旧全量权重。
同一组的五成员融合权重用同一个650题开发区、同一固定正则训练。
主比较是五成员基线＋长文本比较器，固定融合系数0.1；核模型gamma=3、C=0.1、融合0.1，作为已登记的另一条方法对照。
神经结构、优化器、种子、最大轮数、按开发损失早停、算术门控全部相同；删样本后每轮更新数会减少，随机对照用于识别数量/正则化影响。
只在开发区从三个清理版本选择一个主实验，并先固定选择，再训练匹配随机对照。没有看到审查结果后重选主实验。
本轮只用一个训练种子组合；删除样本会改变优化路径，若小幅提升仍需重复验证。

主方法：长文本融合
{chr(10).join(rows)}

预先选定的主实验为{primary}。
相对完整数据重训：净增{v['net']:+d}，变化{v['gain_percentage_points']:+.3f}个百分点；模板分组bootstrap 95%区间[{ci[0]:.3f}, {ci[1]:.3f}]；精确单侧配对检验p={v['paired_one_sided_p']:.5f}。
相对匹配随机删除：修复{r['repaired']}、误伤{r['damaged']}、净增{r['net']:+d}。
这些统计量说明不确定性，不作为删除实验成果的理由；本轮没有沿用“至少净增5题才保留”的门槛。

其他方法的完整对照（各自与完整数据的同方法相比）
{chr(10).join(heads)}
这些是描述性次要比较；不能事后挑最高分当成预先注册主实验通过。

复核与复现
quality/adjudication.json：67题的具体判断与证据；quality/ranking.json：3155题交叉预测风险与类型。
quality/evidence_only.json、quality/quality_05.json、quality/quality_10.json、quality/random_matched.json：删除与保留索引，均仅用于训练。
quality/removed_training_review.json：最大筛选组原题、四候选、筛选理由与内部模型诊断，可核查但不是盲测页面。
runs/各版本/：每个基线、长文本、核模型的可复现权重、词表、配置、训练记录和开发结果。
audit_result.json、audit_predictions.npz、audit_changes.jsonl：完整950题配对结果及四种改动版本的全部改答，含修复、误伤和不影响得分的改选。
checks.json：分组隔离、OOF未见题、10%上限、有效难题保护、随机分层匹配、推理导出与候选顺序等变性均检查通过。
frozen.json：审查前冻结代码、模型、筛选表、官方训练数据哈希；旧主模型、第18/19轮和原始数据哈希不变。
所有单个部署方案的推理资产均低于16MiB。审查时使用本机CPU四线程；记录的是热运行各模块计时，并非完整冷启动的官方资源认证。

复现顺序（已完成的输出应保留；不要覆盖冻结目录）
使用现有D:\\anaconda3\\envs\\mp1\\python.exe，并把Project2\\.venv\\Lib\\site-packages加到PYTHONPATH。
round20_common.py；round20_quality.py oof / scan / preview；round20_adjudicate.py；round20_quality.py rank；
round20_train.py full / quality / select / random；round20_evaluate.py freeze / audit；round20_write_report.py。
复核决定已记录于训练代码与adjudication.json；重新运行不需要让模型临时猜测筛选清单。
本轮实质性代码、筛选复核和分析使用AI辅助，需随最终课程报告披露。
'''
    report=OUT/'训练数据质量筛选_第20轮报告.txt';report.write_text(text,encoding='utf-8')
    registry=dict(status='completed_all_candidates_retained',primary=primary,primary_head='long',
        train_side_primary_result=v,vs_random=r,positive_development_and_audit_gain=a['positive_development_and_audit_gain'],
        promoted=False,incumbent=str(ROOT/'experiments/round4/selected_bundle'),validation_opened=False,test_opened=False,
        runs={n:dict(path=str(OUT/'runs'/n),removed=runs[n]['removed'],audit=a['results'][n]) for n in NAMES})
    dump(OUT/'candidate_status.json',registry)
    artifacts=[report,OUT/'audit_result.json',OUT/'audit_predictions.npz',OUT/'audit_changes.jsonl',OUT/'candidate_status.json',OUT/'frozen.json',OUT/'checks.json']
    dump(OUT/'completion_manifest.json',dict(status='complete_train_only_data_filtering_comparison',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_candidates_retained=True,
        validation_opened=False,test_opened=False,integrity=integrity(),artifacts={str(p.relative_to(OUT)):sha(p) for p in artifacts}))
    print(report);print(conclusion);print(random_note)


if __name__=='__main__':main()
