from data_access import ROOT
import json,html,hashlib,math

OUT=ROOT/'experiments/round6'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def esc(s):return html.escape(str(s))
def table(headers,rows):return '<table><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
def main():
    dev=read('development_evaluation.json');audit=read('audit_evaluation.json');plan=read('plan.json');frozen=read('source_freeze.json')
    for n,h in frozen['sha256'].items():assert hashlib.sha256((ROOT/'research'/n).read_bytes()).hexdigest()==h
    original=json.loads((ROOT/'experiments/round4/asset_manifest.json').read_text())
    for n,info in original.items():assert hashlib.sha256((ROOT/'experiments/round4/selected_bundle'/n).read_bytes()).hexdigest()==info['sha256']
    assert not audit['extraction_gate_passed']
    decision=dict(selected='round4/selected_bundle',previous_validation_correct=750,previous_validation_n=798,round6_validation_read=False,test_access=False,
        reason='Independent TRAIN semantic audit failed preregistered extraction gate; do not compare validation or deploy.',development=dev['overall'],audit=audit['overall'],thresholds=plan['gates'],old_model_assets_unchanged=True,
        independence='16 TRAIN audit questions withheld from extractor development until source freeze. Same assistant annotated; no external human adjudication. Not independent of historical base-model training.',annotation_counts=dict(development=67,audit=60),next_round_policy='These audit questions are now exposed; any tuning on them requires a new sealed audit split for a fresh independence claim.')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'research').glob('round6_*.py')};(OUT/'source_hashes.json').write_text(json.dumps(manifest,indent=2))
    d=dev['overall'];a=audit['overall']
    body='<h1>第六轮：先核对“题目读对没有”</h1><p>本轮完成训练集独立约束标注、实体账本与时间方程原型。提取质量未通过预先固定的门槛，因此<b>没有读取验证集或测试集</b>，没有替换模型。历史最佳仍为750/798（93.98%），不是本轮新测结果。</p>'
    body+='<h2>每一步到底改了什么？</h2><p>不是调整提示词让大模型重新回答。四份候选解答是固定输入；改的是Python代码、选择器的特征、训练目标与参数，以及可解释的算术/关系检查。</p>'
    body+=table(['阶段','改动','历史验证结果'],[['基线','多数投票；线性分类器','688 → 719 / 798'],['第二轮','MLP与LightGBM、候选间差异特征、正则化与训练时长','MLP745；LightGBM734'],['第三轮','增加互补特征与模型，用训练OOF分数学习组合权重','748'],['第四轮','重算部分分数算式、修复显示答案与官方解析答案不一致','750'],['第五轮','数量节点、单位与阶段图；严格单位链和年龄方程探索','图特征749，保留750'],['第六轮','独立语义标注，严格关系评估，账本与时间方程','质量门槛未过，未读取验证集']])
    body+='<h2>为什么需要关系绑定</h2><p>“买3支笔，每支1元”应表示为购买者→笔→数量3，以及笔→单价1元/支，才得到支出3元。“3支笔共1元”则是另一条约束。又如 Ann 四年后是 Bob 四年后的两倍，方程是 A+4=2(B+4)；若题目说 Bob 现在的年龄，方程应为 A+4=2B。数字相同，绑定不同，答案就不同。</p><p>本轮账本把初始余额、增量、支出、单价、总价和付款分开。求解器实现“初始量＋带正负号的变化”“付款－各项数量×单价”“折后总价÷数量”。时间求解器联立人物与阶段方程。无法归属的数量或人物会导致弃权，而不是凭部分信息输出答案。</p>'
    body+='<h2>先划分、再开发、最后审查</h2><p>固定种子61015，按时间、价格、总量、人物四类分层抽取32道官方训练题，每类4道开发、4道审查。排除上一轮明确人工查看过的样例。16道审查题在提取器代码冻结前不打开。标注来自题目原文，不来自候选解答、参考答案或模型预测。</p><p>这是一份相对本轮开发独立的小型审查集，不是外部盲测：同一助手负责实现和标注，历史基础模型也训练过整个训练集。分层关键词只用于抽样，类别不是纯粹或穷尽的。127条标注及目标仍需将来人工复核，尤其含隐含常识与歧义的题目。</p>'
    body+=table(['指标','开发16题','独立审查16题'],[['正确提取 / 输出关系',f"{d['tp']}/{d['predicted_facts']}",f"{a['tp']}/{a['predicted_facts']}"],['关系精确率',f"{d['fact_precision']:.1%}",f"{a['fact_precision']:.1%}"],['关系召回率',f"{d['fact_recall']:.1%}",f"{a['fact_recall']:.1%}"],['漏掉的关系',d['fn'],a['fn']],['目标完整匹配 / 全部题',f"{d['target_correct']}/16",f"{a['target_correct']}/16"],['全部关系与目标均正确的题',d['complete_questions'],a['complete_questions']],['求解器可给结果的题',d['solver_coverage'],a['solver_coverage']]])
    body+='<p>门槛在看审查题之前固定：至少20条输出关系，精确率≥95%，目标完整匹配≥90%，至少25%的题有关系输出，关键绑定错误为0。审查未通过。不能把“1个输出目标正确率100%”说成目标识别可靠，因为其余15题没有识别目标。也不能把开发题27/27当作泛化证据。</p>'
    body+='<h2>暴露出的具体缺口</h2>'+table(['缺口','审查中的表现','需要什么能力'],[['人物与指代','Jordyn’s age、her sister Diane、同一家庭不同成员无法统一','人物别名、亲属和代词指向；保留关系方向'],['单价与总价','two cameras at $110 被当作两台合计；标注按单价解释','价格作用范围和省略 each 的语义；歧义时弃权'],['同物品多批次','餐厅三批 meals 数量和价格不能只按 meal 名称合并','批次实体和局部数量—价格配对'],['时间与单位','18个月后的年龄与当前月龄；几周内每周数量','关系两端时间、目标时间、单位转换'],['总量与新增量','买10kg又加5kg，后两周每周翻倍；已有100元与1000元目标','事件账本、阶段总量、余额与新增需求的区分']])
    body+='<p>审查时发现评分代码把时间2与2.0视为不同JSON字符串，已做数值等价修正，并保留修正前输出和冻结哈希修订记录。提取器、门槛和标注未据此改动。修正后的精确率为8/9；修正不改变“未通过”的结论。相机题存在省略each的解释空间，即便宽松接受该关系，覆盖与目标识别仍远低于门槛。</p>'
    body+='<h2>下一次如何继续而不污染审查</h2><p>现在这16道审查题已曝光，可以作为下一轮开发材料，但不能再宣称它们是未见审查。下一轮应扩展训练标注，补齐“人物别名/代词”“价格作用范围/批次”“事件时间与目标范围”，并在修改前封存新的审查题。当前证据不支持继续靠关键词正则叠加就能达到99%，也不支持以更大模型替代语义提取质量验证。</p>'
    body+='<h2>逐题标注与预测差异</h2><input placeholder="搜索题号或题目" oninput="document.querySelectorAll(\'.case\').forEach(e=>e.hidden=!e.textContent.toLowerCase().includes(this.value.toLowerCase()))">'
    for label,ev,goldname in [('开发',dev,'development_gold.json'),('独立审查',audit,'audit_gold.json')]:
        gold={r['index']:r for r in read(goldname)['rows']}
        for r in ev['results']:
            body+=f'<details class="case"><summary>{label} #{r["index"]} · {esc(r["category"])} · TP {r["tp"]} / FP {r["fp"]} / FN {r["fn"]}</summary><p>{esc(r["question"])}</p>'
            body+='<h3>原文人工标注</h3><pre>'+esc(json.dumps(gold[r['index']],indent=2,ensure_ascii=False))+'</pre><h3>提取结果与求解</h3><pre>'+esc(json.dumps(r,indent=2,ensure_ascii=False))+'</pre></details>'
    dest=OUT/'report';dest.mkdir(exist_ok=True);(dest/'实验报告.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>MP2 第六轮关系绑定</title><style>body{font:16px/1.75 system-ui;max-width:1120px;margin:35px auto;padding:0 24px;color:#163042;background:#f5f8fb}h1,h2{color:#14596b}table{border-collapse:collapse;background:#fff;width:100%;margin:16px 0}td,th{border:1px solid #cbd9e2;padding:9px;text-align:left}details{background:white;padding:15px;margin:12px 0;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}input{padding:12px;width:90%}summary{cursor:pointer}</style>'+body+'</html>',encoding='utf-8')
    print(json.dumps(decision))
if __name__=='__main__':main()
