"""Offline evidence report with an auditable review of every remaining error."""
from data_access import ROOT
import hashlib,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'experiments/round4'


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    r=read('results.json');audit=read('error_audit.json');models=read('validation_comparison.json')
    relation=read('relation_selection.json');balanced=read('balanced_relation_selection.json');pair=read('pair_selection.json')
    gates=read('rare_gate_comparison.json');gate_choice=read('rare_gate_selection.json')['chosen']
    old=json.loads((ROOT/'experiments/round3/reliability.json').read_text())
    pred=np.load(OUT/'selected_predictions.npz');new_choice=pred['selected'];new_correct=pred['correct']
    report=OUT/'report';report.mkdir(exist_ok=True)
    english={'实体与关系方向':'Entity/relation direction','遗漏或重复计算范围':'Missing/double-counted scope',
        '回答了不同的所求量':'Wrong target quantity','单位时段或百分比基数':'Units/time/percentage base',
        '局部数值计算':'Local arithmetic','题目措辞与参考解不一致':'Question/reference mismatch',
        '官方解析标记与展示答案不一致':'Official/display answer mismatch'}
    counts=audit['category_counts'];ordered=sorted(counts,key=lambda k:counts[k],reverse=True)
    fig,ax=plt.subplots(figsize=(10,4.4));ax.barh([english[k] for k in ordered][::-1],[counts[k] for k in ordered][::-1],color='#397d91')
    for i,k in enumerate(ordered[::-1]):ax.text(counts[k]+.15,i,str(counts[k]),va='center')
    ax.set(xlim=(0,20),xlabel='Questions among the previous 50 errors',title='Manual primary-cause audit (multiple causes can coexist)')
    fig.tight_layout();fig.savefig(report/'error_categories.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,['calibrated','conflict_calibrated'],['All validation questions','Conflicting-answer questions']):
        for data,label in [(old,'Round 3'),(r,'Round 4 gate')]:
            bins=data[key]['bins'];ax.plot([v['confidence'] for v in bins],[v['accuracy'] for v in bins],'o-',label=label)
        ax.plot([.4,1],[.4,1],'--',color='gray');ax.set(xlim=(.4,1.01),ylim=(.4,1.01),xlabel='Calibrated confidence',ylabel='Observed accuracy',title=title);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(report/'reliability.png',dpi=160);plt.close(fig)
    modelrows=''.join(f'<tr><td>{m["name"]}</td><td>{m["validation"]["correct"]}/798</td><td>{m["validation"]["accuracy"]:.2f}%</td></tr>' for m in models)
    gaterows=''.join(f'<tr><td>{g["kind"]}</td><td>{g["margin_limit"]:g}</td><td>{g["correct"]}/798</td><td>{g["fixed_old_errors"]}</td><td>{g["lost_old_correct"]}</td></tr>' for g in gates)
    metrics=''.join(f'<tr><td>{scope}</td><td>{label}</td><td>{old[key][field]:.4f}</td><td>{r[key][field]:.4f}</td></tr>'
        for scope,key in [('全部题','calibrated'),('答案冲突题','conflict_calibrated')]
        for label,field in [('Brier ↓','brier'),('ECE ↓','ece_equal_count_10'),('AUROC ↑','auroc')])
    diagnostics=[]
    for row in audit['rows']:
        i=row['index'];status='已修正' if new_correct[i] else ('换了候选但仍错' if new_choice[i]!=row['selected'] else '仍错')
        diagnostics.append(dict(index=i,category=row['primary_category'],status=status,note=row['review_note']))
    (OUT/'remaining_error_status.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=''
    for row,diagnostic in zip(audit['rows'],diagnostics):
        i=row['index'];category=html.escape(row['primary_category']);status=diagnostic['status']
        cards+=f'''<details class="case" data-category="{category}"><summary><b>#{i}</b> · {category} · <span class="status">{status}</span><br>{html.escape(row['question'])}</summary><p><b>判读：</b>{html.escape(row['review_note'])}</p><p>官方解析值：{html.escape(str(row['answers']))}；旧候选 {row['selected']} → 新候选 {int(new_choice[i])}；标签正确候选 {row['correct_candidates']}。</p><div class="columns"><section><h4>旧选择的候选全文</h4><pre>{html.escape(row['selected_text'])}</pre></section><section><h4>一个标签正确候选的全文</h4><pre>{html.escape(row['one_correct_text'])}</pre></section></div><details><summary>官方验证参考解（仅供分析，不进入推理）</summary><pre>{html.escape(row['reference'])}</pre></details></details>'''
    opts=''.join(f'<option>{html.escape(k)}</option>' for k in ordered)
    v=r['validation'];p=r['paired'];bestbalanced=max(balanced['records'],key=lambda q:q['internal']['macro_recall']);bestpair=min(pair['records'],key=lambda q:q['internal']['nll'])
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MP2 第四轮 · 从错题到约束检验</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;color:#20313e;max-width:1120px;margin:36px auto;padding:0 22px;background:#fbfcfc}}h1,h2{{color:#195c6a}}h2{{margin-top:36px}}table{{width:100%;border-collapse:collapse;margin:18px 0}}th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #d2dfe3}}.card{{padding:20px;background:#eaf3f5;border-radius:12px}}.warning{{border-left:4px solid #b17c37;padding-left:16px}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f6;padding:14px;font:13px/1.5 ui-monospace,monospace}}summary{{cursor:pointer}}.case{{padding:14px;border:1px solid #d3e0e5;border-radius:8px;margin:12px 0;background:white}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.status{{color:#a66327}}input,select{{padding:10px;margin:4px;border:1px solid #a9c2cb;border-radius:6px}}small{{color:#5d6b73}}@media(max-width:750px){{.columns{{display:block}}}}</style>
<h1>第四轮：题意约束在哪里失效？</h1><p>2026-10-14 · 只用训练集和验证集 · 测试输入与参考答案未读取</p>
<div class="card"><b>当前验证集：{v['correct']}/798 = {v['correct']/798*100:.2f}%；冲突题：{v['hard_correct']}/355 = {v['hard_correct']/355*100:.2f}%。</b><p>比上一轮净增 2 题，仍有 48 道错误；99% 需要至少 791/798，尚需净增 {r['additional_correct_needed_for_99']} 题。新增收益来自可核验的算术与格式一致性，尚未证明题意理解能力改善。</p></div>
<h2>50 道旧错题的逐题审查</h2><img src="error_categories.png" alt="旧错题主要原因统计"><p>39 道涉及实体关系、计数范围或所求目标，4 道涉及单位/时段/百分比基数，4 道主要是局部算错，2 道存在题目措辞与参考解不一致，1 道是展示答案与官方解析值不一致。分类为人工/AI 审查的主要原因，不代表只有一种错误；全部原文在页末可展开。</p>
<p class="warning">#32 的“1/4 more”与参考的“四分之一”不同；#661 的“at 20% of”与参考的“20% off”不同。保留原题、原标签和评分器，不据此改标签或写逐题特判。#733 展示 boxed 250，却被官方优先 #### 规则解析为标题里的 3；修复选择策略，不修改解析器。</p>
<h2>真正针对题意的新监督</h2><p><b>数量作用：</b>从训练参考解提取算术依赖图，对所求结果关于题目数量求导，学习正向、负向或未使用三类关系。它试图区分“卖出收入应增加余额”与“买入支出应减少余额”，并识别未进入最终结果的数量。排除带明显数值冲突、重复数量绑定歧义的训练标签；词表、IDF 和分类器只在训练数据上拟合。</p>
<p>普通损失最优配置只识别出约 4% 的负向关系。类别平衡后，负向召回为 {bestbalanced['internal']['by_class']['2']['recall']*100:.1f}%，但宏平均召回仅 {bestbalanced['internal']['macro_recall']*100:.1f}%。这能提供弱关系证据，尚不足以可靠核验完整题意。只判断方向还区分不了少乘一次、重复计数和总量/单份混淆。</p>
<p><b>数量对运算关系：</b>进一步从训练解的表达式树学习两个数量最近共同祖先的运算：加、减/反向减、乘、除/反向除。用两个数量的上下文、之间词语和问题目标预测运算，再检查候选依赖图。训练内部留出集共 {bestpair['internal']['n']} 个数量对，准确率 {bestpair['internal']['accuracy']*100:.2f}%。这一路也未带来可靠的最终选择收益。</p>
<h2>对照与失败结果</h2><p>以下四种 MLP 同样使用 35 轮训练。parser 仅加入依赖图可解析性、数值检查及答案标记等诊断；semantic 仅加入学到的数量作用关系；all 同时使用。所有训练期语义特征均由训练折外模型生成，避免把同题参考解直接喂给选择器。</p><table><tr><th>方法</th><th>验证正确数</th><th>准确率</th></tr>{modelrows}</table>
<p>新增单模型最高 747/798，未超过原融合的 748/798。7 组小型线性纠错消融分别检验数量作用、运算关系、诊断、算术证据及其组合；在训练内部筛选中都选择了“零修正”。没有强行加入未获支持的语义分支，也没有新增部署模型成员。</p>
<h2>保守一致性纠错为何有小幅收益</h2><p>只有在已有候选明确矛盾且另一个候选提供对应答案时才提出修正：① 官方解析值与最后展示 boxed 值不同；② 安全重算算术依赖链后得到另一个候选的答案，且该候选自身计算一致。只从现有四个候选中选择，不生成答案、不查询参考解。</p>
<p>原 951 题训练内部留出集没有触发解析/修复事件，因此初次选择停用。改用全部 4,755 题的训练折外预测拟合罕见事件阈值；选出的分数差上限为 4。训练 OOF 上改动 {gate_choice['best']['changes']} 题，修正 {gate_choice['best']['fixed']}、误改 {gate_choice['best']['lost']}；这是阈值拟合表现，不能当作独立测试结果。此前停用的记录仍完整保留。</p>
<table><tr><th>规则</th><th>训练选择的阈值</th><th>验证正确数</th><th>修正旧错题</th><th>误改旧对题</th></tr>{gaterows}</table>
<p>组合规则修正 #52 的分数错误和 #733 的解析不一致。#739 被修成了数值一致的复利答案，仍违反题目“按原始本金”计息的约束，因此仍错。这是算术一致性不能替代题意理解的直接反例。剩余 48 题中，原审查的 43 道语义/范围/单位问题全部仍在。</p>
<h2>可靠性、资源与复现边界</h2><p>相对上一轮，修正 {p['fixed_old_errors']} 道、误改 {p['lost_old_correct']} 道；配对 bootstrap 的准确率变化 95% 区间为 [{p['gain_95ci_pp'][0]:.2f}, {p['gain_95ci_pp'][1]:.2f}] 个百分点。样本很少，且区间没有校正反复使用验证集开发带来的选择偏差。</p><table><tr><th>范围</th><th>指标</th><th>上一轮</th><th>本轮</th></tr>{metrics}</table><img src="reliability.png" alt="校准曲线比较"><p>置信度仅用训练 OOF 分数和训练拟合后的规则输出校准，非完全嵌套估计。不能把高置信子集的准确率当作全量 99%。</p>
<p>完整推理资产 {r['asset_bytes']/2**20:.2f} MiB；本机 i7-11800H 四线程的新进程预测＋独立评分为 {', '.join(f'{x:.2f}' for x in r['local_fresh_prediction_plus_score_seconds'])} 秒。3 次官方验证评分一致。未运行测试集，因此没有声称已完成参考 Xeon 上的正式测试预算认证。保留一个仅含单 MLP 的紧凑备选：747/798，资产 {r['compact_asset_bytes']/2**20:.2f} MiB。</p>
<p>实现检查包含小数/混合分数、求导方向、分数中间值传播、Markdown 列表符号、单位等式、数量对方向、候选换序、未知 ID 和不同批量大小。严格纠错解析器专门避免把列表前缀“-”当成负号；前期语义特征实验保留原版本行为以保持可复现。</p>
<h2>下一步应解决的瓶颈</h2><p>当前仍是弱文本表示与不完整数量绑定：同一个数字可能属于不同实体，代词与时间阶段没有可靠消歧；求导符号不能表达作用次数；数字正确也可能使用了错误单位或百分比基数。下一步应让实体—数量—单位—时间阶段形成统一约束图，并先用训练参考解检查图的绑定精度与覆盖率。不能把当前 53% 左右的局部关系分类器当成强教师，继续堆叠或延长训练没有证据能逼近 99%。</p>
<h2>可搜索逐题审查</h2><input id="query" placeholder="搜索题目、编号或判读" size="38"><select id="category"><option value="">全部类别</option>{opts}</select><span id="count"></span>{cards}
<h2>文件与披露</h2><p>复现命令见 research/第四轮运行说明.txt；源码、配置、模型、折外记录、全部失败对照、官方验证评分与哈希保存在 experiments/round4。原始数据、解析器、评分器均保持不变。没有读取测试输入或参考答案，没有外部训练数据、额外预训练权重、LLM 推理 API 或逐题答案查表。AI 对分析、实现、运行与文稿提供了实质性帮助。</p>
<script>const cases=[...document.querySelectorAll('.case')];function filter(){{const q=document.querySelector('#query').value.toLowerCase(),c=document.querySelector('#category').value;let n=0;for(const el of cases){{const ok=(!c||el.dataset.category===c)&&el.textContent.toLowerCase().includes(q);el.hidden=!ok;if(ok)n++;}}document.querySelector('#count').textContent=` ${{n}} / ${{cases.length}} 题`;}}document.querySelector('#query').addEventListener('input',filter);document.querySelector('#category').addEventListener('change',filter);filter();</script></html>'''
    (report/'实验报告.html').write_text(content,encoding='utf-8')
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'research').glob('round4_*.py')}
    (OUT/'source_hashes.json').write_text(json.dumps(sources,indent=2))
    ledger=dict(internal_mlp_configurations=8,full_mlp_refits=4,quantity_relation_configurations=6,
                quantity_relation_crossfit_models=9,operation_pair_configurations=3,operation_pair_crossfit_models=3,
                residual_families=7,residual_penalties_each=3,initial_gate_families=4,rare_gate_families=4,
                gate_thresholds_each=7,test_access=False,
                scope_note='Counts separate training selections, crossfits, matched validation ablations, and rare-event threshold adaptation. No semantic gain established.')
    (OUT/'ledger.json').write_text(json.dumps(ledger,indent=2));print('Round 4 report and 50-case searchable audit saved.')


if __name__=='__main__':main()
