from data_access import ROOT
import json,html,hashlib
from pathlib import Path
import numpy as np
from round5_experiment import OUT

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def esc(s):return html.escape(str(s))
def table(headers,rows):return '<table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
def main():
    comp=read('validation_comparison.json');quality=read('train_quality.json');errs=read('remaining_error_graphs.json');age=read('age_validation_comparison.json');rate=read('rate_validation_comparison.json')
    old_manifest=json.loads((ROOT/'experiments/round4/asset_manifest.json').read_text())
    for rel,info in old_manifest.items():assert hashlib.sha256((ROOT/'experiments/round4/selected_bundle'/rel).read_bytes()).hexdigest()==info['sha256']
    summary=dict(selected='round4/selected_bundle',correct=750,n=798,accuracy=750/798,hard_correct=307,hard_n=355,remaining=48,target_correct=791,additional_needed=41,test_access=False,previous_bundle_hashes_verified=True,
        graph_trial=comp['records'][0]['metrics'],remaining_graph_diagnostics=dict(target_unit_known=sum(e['graph']['target']['unit'] is not None for e in errs),selected_root_found=sum(e['candidates'][e['selected']]['features']['root'] for e in errs),selected_target_dimension_known=sum(e['candidates'][e['selected']]['features']['target_dimension_known'] for e in errs),selected_ambiguous_bindings=sum(e['candidates'][e['selected']]['features']['ambiguous_bindings']>0 for e in errs)))
    (OUT/'selection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'research').glob('round5_*.py')};(OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    rows=[['保留基线',750,307,0,0]]+[[r['kind'],r['metrics']['correct'],r['metrics']['hard_correct'],r['metrics']['fixed'],r['metrics']['lost']] for r in comp['records']]
    body='<h1>第五轮：实体—数量—单位—时间约束图</h1><p>最佳仍为 <b>750/798（93.98%）</b>；冲突答案子集 307/355（86.48%）。本轮未读取测试集。达到 99% 至少需要 791 题正确，还差 41 题。</p>'
    body+='<h2>结果与选择</h2>'+table(['方案','验证正确 / 798','冲突子集 / 355','修复旧错','误伤旧对'],rows)
    body+='<p>完整图特征在训练内部留出集由 894/951 变为 895/951；验证集降至 749/798，因此不替换现有方案。其余消融在训练内部选择了零修正；表中的持平不证明这些约束无价值。新图提取耗时 %.3f 秒（仅提取，不是完整预测计时）。原模型未变，已核对所有资产 SHA256。</p>'%comp['feature_seconds']
    body+='<h2>训练集先行的质量审查</h2><p>保留每次数字出现的原文位置；同数值、不同语境不再合并为同一个图节点。不过旧计算链遇到多个可能来源时仍弃权绑定。实体字段目前只是名词/单位代理，所有者靠局部文字识别，不能当作完整实体解析。阶段词也不等于事件先后图。</p>'
    body+=table(['指标','结果'],[['参考解可恢复最终计算根',f"{quality['reference_root_rate']:.2%}"],['来源数值存在多义绑定的训练题',quality['reference_ambiguous_binding_questions']],['参考解目标维度一致 / 可比较',f"{quality['reference_target_dimension_match']}/{quality['reference_target_dimension_known']}"],['正确候选仍被报目标维度冲突 / 可比较',f"{quality['correct']['dimension_mismatch']}/{quality['correct']['dimension_known']}"],['随机12题人工开发审查：粗目标单位', '覆盖10题，覆盖部分10题正确；不是实体、所求范围或时间阶段的联合准确率']])
    body+='<p>结论：不能硬性否决“维度冲突”的候选。正确候选与错误候选的平均未使用数量分别为 %.3f、%.3f，简单“数有没有用完”几乎没有区分力；正确推理也会合法复用一个数量。</p>'%(quality['correct']['mean_unused'],quality['incorrect']['mean_unused'])
    body+='<h2>两个可解释的严格分支</h2><p>单位率链：枚举最多7个明确数量的乘除组合，要求精确单位约掉、每个数量使用一次且答案唯一。训练覆盖4题，仅2题正确：遗漏 daily→weekly 的隐含7，以及 thrice 所属人物倍率。训练内部选择禁用；验证仅覆盖2题，未启用改答。</p>'
    body+='<p>年龄阶段图：用 age(person,t)=age(person,0)+t，联立明确人物、年龄事实、倍数与年龄差。先在训练上修复小数分句和“明年”偏移，覆盖12题、数值12/12与参考一致。此为开发样本结果，且一题未指定未来事件被当成共同时间原点，不能把 current_ages 字段一概解释为真实当前年龄。验证覆盖4题、4/4数值一致，但训练内部没有改答收益，因此门控仍禁用。该分支是在首轮图特征验证分析后追加，属于自适应开发，不能视为独立盲测。</p>'
    ageexample=read('age_train_audit.json')[0]['proof']
    body+=table(['方程类型','人物系数','右侧常数','左时间','右时间'],[[e['kind'],e['coefficients'],e['rhs'],e.get('left_time'),e.get('right_time')] for e in ageexample['equations']])
    body+='<h2>改答审查与可靠性</h2>'+table(['验证题索引','变化','可支持的判断'],[[359,'92 → 97，修复','漏掉星期二且重复周末；图仍错误标出 Monday 为单位，不能把修复归因于正确理解一周。'],[176,'1020 → 870，误伤','50$ 后置货币符号、two moisturizers each 和人物倍数未完整绑定；漏乘数量2。'],[513,'8 → 12，误伤','把 Mary 的四年后年龄当作当前年龄；粗阶段词不足以阻止。']])
    body+='<p>这不是已证明的语义增益。没有对失败方案重新校准置信度，也没有因为一题修复就部署。内部筛选重用了旧的训练OOF分数，并非完全嵌套交叉验证；多轮验证集开发存在选择偏差。官方只评价最终数值，正确候选的推理也可能有错误，故“推理有错”不能自动等同于“答案选错”。</p>'
    body+='<h2>下一步的实际瓶颈</h2><ol><li>先建立训练内部独立的实体、指代、单位率分母、时间与目标范围标注集；当前粗图质量不足，继续调排序权重无法补足关系。</li><li>把有向动作建成收支/库存账本：区分初始量、已有总量、新增量、流出量及重复提及；不能仅靠每个数字出现次数判漏算或重复。</li><li>扩展阶段图时显式标注每个关系两端的时间；保留不确定性，按完整约束提取正确率和覆盖率共同筛选。</li></ol>'
    body+='<h2>保留方案的48道错题图</h2><p>下列参考解、标签及题号仅用于分析，推理代码不读取它们。</p><input id="q" placeholder="搜索题目或编号" oninput="document.querySelectorAll(\'.case\').forEach(e=>e.hidden=!e.textContent.toLowerCase().includes(this.value.toLowerCase()))">'
    for e in errs:
        g=e['graph'];body+=f'<details class="case"><summary>#{e["index"]} · {esc(e["question"][:150])}</summary><p>{esc(e["question"])}</p>'
        body+=table(['节点','数值','实体代理','单位','每单位','所有者代理','阶段代理'],[[n['id'],n['raw'],n['entity'],n['unit'],n['per'],n['owner'],n['stage']] for n in g['nodes']])
        body+='<p>目标：'+esc(g['target'])+'</p><pre>'+esc(json.dumps(e['candidates'],indent=2,ensure_ascii=False))+'</pre><p>参考解：'+esc(e['reference'])+'</p></details>'
    dest=OUT/'report';dest.mkdir(exist_ok=True)
    (dest/'实验报告.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>MP2 第五轮约束图</title><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#163042;background:#f6f8fb}h1,h2{color:#14596b}table{border-collapse:collapse;background:white;width:100%;margin:16px 0}td,th{border:1px solid #cbd9e2;padding:8px;text-align:left}details{background:white;padding:15px;margin:12px 0;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}input{width:90%;padding:12px}summary{cursor:pointer}</style>'+body+'</html>',encoding='utf-8')
    print(json.dumps(summary))
if __name__=='__main__':main()
