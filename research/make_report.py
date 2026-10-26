"""Build an offline research report and static, exportable figures from records."""
from data_access import ROOT
import html
import json
from pathlib import Path
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'experiments'/'round2'


def table(headers,rows):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    report=OUT/'report';report.mkdir(exist_ok=True)
    reliability=json.loads((OUT/'reliability/results.json').read_text())
    records={p.parent.name:json.loads(p.read_text()) for p in OUT.glob('*/result.json')}
    # Fixed-budget ablations do not use an internal early-stopping subset.
    for name,r in records.items():
        if r['config'].get('fixed_epochs'):
            r['internal_early_stopping_questions']=0
            (OUT/name/'result.json').write_text(json.dumps(r,indent=2))
    acc=lambda name:100*records[name]['validation']['correct']/798
    correct=lambda name:records[name]['validation']['correct']
    comparison=json.loads((OUT/'comparison.json').read_text())
    trials={}
    for p in OUT.glob('comparison*.json'):
        data=json.loads(p.read_text())
        for e in data.get('ensemble',[]):
            signature=json.dumps([e['members'],e['aggregation'],e.get('weights')])
            trials[signature]=e
    total_train=sum(r['training_seconds'] for r in records.values())
    oof_train=sum(r['seconds'] for r in reliability['oof_training_ledger'])
    old_calibration=OUT/'reliability/results_35epochs.json'
    if old_calibration.exists():oof_train+=sum(r['seconds'] for r in json.loads(old_calibration.read_text())['oof_training_ledger'])
    selected=reliability['members'][0]
    fullacc=100*reliability['official_validation_score']['correct']/798
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    labels=['Majority','Linear hybrid','LightGBM rank','MLP (early)','MLP + checks','Best tested ensemble']
    vals=[86.2155388471,90.1002506266,acc('lgbm_rank'),acc('mlp_rich'),acc(selected),
          max(e['validation']['correct']/798*100 for e in trials.values())]
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    ax.barh(labels,vals,color=['#9ba6b5','#9ba6b5','#5d83a6','#5d83a6','#14695f','#5d83a6'])
    ax.invert_yaxis();ax.set_xlim(80,101);ax.axvline(99,color='#b0433e',linestyle='--',label='99% target')
    for i,v in enumerate(vals):ax.text(v+.2,i,f'{v:.2f}%',va='center')
    ax.set_xlabel('Validation selection accuracy (%)');ax.set_title('All 798 validation questions; test untouched')
    ax.legend(loc='lower right');fig.savefig(report/'comparison.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for key,label,color in [('uncalibrated_sigmoid_score','Uncalibrated','#ad7358'),('calibrated','Train-only OOF calibration','#14695f')]:
        bins=reliability[key]['bins'];axs[0].plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],'o-',label=label,color=color)
        curve=reliability[key]['risk_coverage'];axs[2].plot([100*r['coverage'] for r in curve],[100*(1-r['accuracy']) for r in curve],'o-',label=label,color=color)
    for key,label,color in [('conflict_uncalibrated','Uncalibrated','#ad7358'),('conflict_calibrated','Calibrated','#14695f')]:
        bins=reliability[key]['bins'];axs[1].plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],'o-',label=label,color=color)
    for ax in axs[:2]:
        ax.plot([0,1],[0,1],'--',color='#999');ax.set(xlim=(0,1.01),ylim=(0,1.01),xlabel='Mean confidence',ylabel='Observed accuracy')
    axs[0].set_title('Reliability: all 798 questions');axs[1].set_title('Reliability: 355 conflicting-answer questions')
    axs[2].set(xlabel='Coverage (%)',ylabel='Error rate (%)',title='Selective risk (analysis only)');axs[2].legend(fontsize=8)
    fig.savefig(report/'reliability.png',dpi=170);plt.close(fig)
    ablation_reference='mlp_checked_s0'
    ablations=[('Full model (35 epochs)',ablation_reference),('No group context','fixed_no_group'),('No answer votes','fixed_no_vote'),
               ('No surface style','fixed_no_style'),('No arithmetic checks','fixed_no_arithmetic'),
               ('No embeddings','fixed_no_embedding'),('BCE only','fixed_bce')]
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    ax.barh([x[0] for x in ablations],[acc(x[1]) for x in ablations],color='#5d83a6');ax.invert_yaxis();ax.set_xlim(89,95)
    for i,(_,name) in enumerate(ablations):ax.text(acc(name)+.08,i,f'{acc(name):.2f}%  ({correct(name)}/798)',va='center')
    ax.set(xlabel='Validation selection accuracy (%)',title='Matched training budget: 35 epochs, same seed / hidden width / optimizer')
    fig.savefig(report/'ablations.png',dpi=170);plt.close(fig)
    c=reliability['calibrated'];u=reliability['uncalibrated_sigmoid_score'];hc=reliability['conflict_calibrated'];hu=reliability['conflict_uncalibrated']
    seeds=[acc(f'mlp_checked_s{i}') for i in range(3)]
    times=reliability['local_fresh_prediction_plus_score_seconds']
    ablation_rows=[(label,correct(name),f'{acc(name):.2f}%',f'{acc(name)-acc(ablation_reference):+.2f} pp') for label,name in ablations]
    factorial_rows=[('原始组','0',acc('mp1_real_d0')),('原始组','0.15',acc('mp1_real_d15')),
                    ('训练内候选重采样','0',acc('mp1_aug_d0')),('训练内候选重采样','0.15',acc('mp1_aug_d15'))]
    patterns=[(p,v['n'],v['correct'],f"{100*v['accuracy']:.2f}%") for p,v in reliability['by_answer_pattern'].items()]
    all_rows=[(name,r['config']['family'],r['config']['features'],r['config'].get('epochs',r['config'].get('rounds')),
               r['validation']['correct'],f"{100*r['validation']['correct']/798:.2f}%",f"{r['training_seconds']:.1f}")
              for name,r in sorted(records.items(),key=lambda x:x[1]['validation']['correct'],reverse=True)]
    selected_errors=798-reliability['official_validation_score']['correct']
    body=f'''<h1>MP2 第二轮实验与 MP1 复盘迁移</h1>
<p class="meta">2026-10-14 · DASE7506 · 只用训练与验证 · 未读取 MP2 测试输入或参考答案</p>
<p><strong>本轮达到 {fullacc:.2f}%（{reliability['official_validation_score']['correct']}/798），尚未达到 99%。</strong>选择了带组内上下文、数值特征与通用算术检查的小型 MLP。它略高于本轮最好集成，资产更小。相较原混合线性基线 719/798，多选对 {reliability['official_validation_score']['correct']-719} 题；冲突子集准确率为 {reliability['validation']['hard_accuracy']:.2f}%。达到 99% 至少需要 791/798，仍须净修正 {791-reliability['official_validation_score']['correct']} 题。</p>
<img src="comparison.png" alt="模型比较">
<h2>实验协议与可复现性</h2>
<p>共完成 {len(records)} 个独立模型配置及 {len(trials)} 个去重后的集成配方。本轮将训练集按题划分为 3,804/951，用内部留出集选轮数，再在全部 4,755 道训练题上重训；官方 798 题验证集仅做选型与分析。后续固定预算消融直接训练指定轮数。标准化、模型参数和校准器均只在训练数据上拟合。</p>
<p>置信度使用训练集三折交叉预测，每个折内再使用自己的训练子集选轮数；每题的校准训练预测来自没见过该题的模型。测试文件访问有程序拦截。官方评分、解析器和数据未修改。所有方法均返回每题一个候选，没有通过拒答缩小评分范围。</p>
<p>本机 CPU 为 i7-11800H；训练使用 CPU 四线程。训练复用了本机已有的 Python 3.12 / PyTorch 2.7.1+cu126 软件运行时，没有使用 MP1 权重或数据。推理只用 NumPy，不导入 PyTorch。独立环境中的 NumPy 为 2.5.3；全部依赖和命令见运行说明。</p>
<h2>MP1 的经验，哪些迁移有效</h2>
<p>依据用户提供的总结及本地师姐报告：MP1 的主要差距来自更强教师、训练派生输入与更充分的蒸馏训练；5-gram 的最终贡献较小。不能把“教师数量多”“模型更宽”或“去掉 dropout”当成收益保证。本轮先测核心模型、互补性和正则化，再判断是否值得蒸馏。</p>
<ul><li><strong>教师质量优先：</strong>宽 256、较强正则的 MLP 为 {acc('mlp_teacher'):.2f}%，未超过宽 128 的 {acc(selected):.2f}%。因此没有为了模仿 MP1 而蒸馏一个更弱教师。</li>
<li><strong>概率平均与分数平均实测区分：</strong>比较了原始分数、组内标准化分数、sigmoid 概率平均。初期三模型的最佳是标准化分数平均；最终单模型略高于最好集成。这里的 sigmoid 未逐成员做温度校准，不能把此结果解读为“校准后的概率集成一定较差”。</li>
<li><strong>覆盖范围而非样本数量：</strong>从同一训练题的四个候选重采样，生成 1 正 3 负、2 正 2 负组合，文本与嵌入始终同步。这保留训练标签，但没有创造新的数学语义；实际未见收益。与 MP1 教师生成新上下文不是同一种增强。</li>
<li><strong>训练长度要有对照：</strong>同一配置的 10 / 35 / 100 轮结果为 {acc('fixed_short'):.2f}% / {acc('mlp_checked_s0'):.2f}% / {acc('fixed_long'):.2f}%。35 到 100 轮仅多选对 1 题，尚不能证明这一小差异稳定。更长训练不能替代有效的表示与泛化控制。</li>
<li><strong>预算允许直接部署：</strong>当前单模型仅 {reliability['asset_bytes']/2**20:.2f} MiB，距离 16 MiB 上限很远，暂无强制蒸馏压缩的必要。</li></ul>
{table(['输入','dropout','验证准确率'],[(a,b,f'{v:.2f}%') for a,b,v in factorial_rows])}
<p>这个 2×2 比较共享最大轮数、优化器和网络宽度，但各自使用内部早停，实际训练目标量不完全一致；不能把差值解释成纯粹的因果贡献。</p>
<h2>相同训练预算下的关键消融</h2>
<p>下表均固定 35 轮、相同种子、隐藏层 128→32、学习率、EMA、dropout 与权重衰减。组内特征为候选嵌入、候选减组均值、候选与组均值的逐元素乘积。混合损失为组内正确候选概率质量的负对数，加 0.25 倍候选级 BCE；允许多个正确候选。</p>
{table(['消融','正确题数','准确率','相对完整模型'],ablation_rows)}
<img src="ablations.png" alt="固定预算消融">
<p>算术检查的消融差距最大，但它只核验可解析的数值等式，不能证明题意理解正确。BCE 与混合损失仅相差 1 题，证据不足以宣称后者普遍更好。旧混合线性模型加“过滤无法解析的候选”后仍为 719 题，因此本轮提升不是由这一合法性过滤单独带来的。</p>
<p>同一新检查配置的三个种子为 {', '.join(f'{v:.2f}%' for v in seeds)}，均值 {np.mean(seeds):.2f}%，样本标准差 {np.std(seeds,ddof=1):.2f} 个百分点。它们分别由训练内部早停选轮数；这是整条训练流程的重复，不是固定 checkpoint 的抖动。</p>
<h2>可靠性，不仅是选对多少题</h2>
{table(['范围','指标','未校准','训练 OOF 校准'],[
('全部 798 题','Brier',f"{u['brier']:.4f}",f"{c['brier']:.4f}"),
('全部 798 题','ECE（10 等频箱）',f"{100*u['ece_equal_count_10']:.2f}%",f"{100*c['ece_equal_count_10']:.2f}%"),
('冲突 355 题','Brier',f"{hu['brier']:.4f}",f"{hc['brier']:.4f}"),
('冲突 355 题','ECE（10 等频箱）',f"{100*hu['ece_equal_count_10']:.2f}%",f"{100*hc['ece_equal_count_10']:.2f}%"),
('冲突 355 题','正确性 AUROC',f"{hu['auroc']:.3f}",f"{hc['auroc']:.3f}")])}
<img src="reliability.png" alt="可靠性与风险覆盖曲线">
<p>全量 ECE 较低部分来自 443 道容易题，因此另列冲突子集。校准没有改变任何候选选择，也没有提高全量准确率。交叉拟合基础模型每次只见训练集的 2/3，最终模型见全训练集，这会带来置信度分布差异。</p>
{table(['保留的高置信度样本数','覆盖率','该子集准确率','错误数'],[(r['n'],f"{100*r['coverage']:.2f}%",f"{100*r['accuracy']:.2f}%",r['errors']) for r in c['risk_coverage']])}
<p><strong>50% 覆盖时的 99%+ 不是完成目标。</strong>排行榜要求对全部题作答；我们的全量结果仍为 {fullacc:.2f}%。以上覆盖率点只用于分析，阈值未作为拒答策略提交。</p>
<h2>统计证据与失败分布</h2>
<p>按题做 5,000 次配对 bootstrap，相对旧混合线性基线的提升区间为 {reliability['paired_gain_over_original_hybrid_95ci_pp'][0]:.2f}～{reliability['paired_gain_over_original_hybrid_95ci_pp'][1]:.2f} 个百分点；新方法修正 {reliability['new_correct_old_wrong']} 道旧错误，同时新增 {reliability['new_wrong_old_correct']} 道错误。全量准确率区间为 {reliability['accuracy_95ci_percent'][0]:.2f}%～{reliability['accuracy_95ci_percent'][1]:.2f}%。这些区间以已选模型为条件，没有校正反复使用验证集选型的乐观偏差。</p>
{table(['有效答案票数模式','题数','正确数','准确率'],patterns)}
<p>模式只统计官方解析器能识别的答案，例如 2–1–1 表示两个相同答案和两个不同答案；合计少于 4 表示有解析失败候选。当前 {selected_errors} 道失败的全文、分数和正确候选索引保存在 <a href="../reliability/failures.json">failures.json</a>，按置信度由高到低排列。</p>
<p>已人工抽查的难例包括：把收入和支出都当减项；追赶问题中 0.6/9 被错误化简为 2/15；忽略第二层涂料；以及“1/4 more”语义与提供的参考答案可能存在歧义。某些候选最终数值正确但中间步骤仍有错误。因此“最终答案匹配”与“推理过程可靠”必须分开，未据个别题修改参考答案、评分器或加入查表规则。</p>
<h2>99% 目标的当前距离</h2>
<p>目前没有达到 99%，也没有证据保证继续增加轮数或集成成员就能达到。达到目标需要把 {selected_errors} 个错误减少到最多 7 个，即再净修正 {selected_errors-7} 个。所有已试基础模型若事后知道答案、只在它们各自最终选择之间择优，准确率为 {comparison['diagnostic']['any_model_correct_oracle_accuracy']:.2f}%；这仅是诊断值，不是可部署算法，也不是所有未来方法或分数融合的理论上限。</p>
<p>下一阶段应围绕剩余错误中的题意约束和算术链展开：在训练集上开发更完整的式子／单位检查，或训练读取候选全文的轻量文本选择器，并按题交叉验证其新增能力。训练派生增强需要增加有效语义覆盖，而不是重复候选。任何继续开发仍应保留官方验证的复用记录；本轮不声称未知测试／audit 准确率。</p>
<h2>资源与交付</h2>
<p>最终候选包：<a href="../selected_bundle/config.json">selected_bundle</a>；完整资产含校准器共 {reliability['asset_bytes']:,} bytes（{reliability['asset_bytes']/2**20:.2f} MiB）。三次本机新进程预测加独立评分为 {', '.join(f'{v:.3f}' for v in times)} 秒，中位数 {statistics.median(times):.3f} 秒。这里测的是完整验证集；按用户要求未测测试集，因此尚未认证课程参考机上的最终测试限时。</p>
<p>模型训练记录合计约 {total_train:.1f} 秒，所选模型的交叉拟合训练约 {oof_train:.1f} 秒；不含依赖下载、独立特征准备、分析及暂停时间。实验结果文件保留了每个配置与训练轮数。<a href="../reliability/results.json">可靠性与官方评估记录</a> · <a href="../reliability/asset_manifest.json">资产 SHA-256</a> · <a href="../../../research/运行说明.txt">复现说明</a></p>
<details><summary>全部 {len(records)} 个模型结果</summary>{table(['配置','模型','特征','轮数／树数','正确题数','准确率','训练秒数'],all_rows)}</details>
<h2>披露与来源</h2><p>Codex 协助设计实验、实现代码、运行分析并撰写本报告。使用课程官方数据与评分接口、PyTorch、NumPy、scikit-learn 和 LightGBM；没有额外数据或预训练权重。</p>
<p>MP1 复盘依据：用户粘贴的双方对照，以及本地 audit_rank2/code/report/REPORT.md。MP2 规则：<a href="https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md">官方 Guide</a>。本文件为开发阶段研究报告，不是已经提交的课程作业。</p>'''
    css='body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#23313e;line-height:1.75}h1{font-size:30px;line-height:1.35}h2{margin-top:38px;color:#145f59}p,li{font-size:15px}.meta{color:#667788}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{border:1px solid #dce2e7;padding:9px;text-align:left}th{background:#edf3f5}tr:nth-child(even){background:#f7f9fa}img{max-width:100%;height:auto;margin:15px 0}a{color:#126f8b}summary{cursor:pointer;font-weight:600}@media print{body{margin:0;max-width:none}h2{break-after:avoid}tr,img{break-inside:avoid}}'
    (report/'实验报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MP2 第二轮实验与可靠性分析</title><style>'+css+'</style><body>'+body+'</body></html>',encoding='utf-8')
    ledger={'model_configurations':len(records),'unique_ensemble_recipes':len(trials),'training_seconds_recorded':total_train,
            'oof_training_seconds':oof_train,'selected':selected,'test_access':False,'goal_99_percent_met':False,
            'validation_correct':reliability['official_validation_score']['correct'],'target_correct':791,
            'seed_scores':seeds,'ensemble_recipes':list(trials.values())}
    (OUT/'research_ledger.json').write_text(json.dumps(ledger,indent=2))
    print('Created',report/'实验报告.html');print(json.dumps({k:v for k,v in ledger.items() if k!='ensemble_recipes'},indent=2))


if __name__=='__main__':main()
