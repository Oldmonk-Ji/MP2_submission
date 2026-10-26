from data_access import ROOT
import json,html,hashlib
from collections import Counter
from round7_core import entities

OUT=ROOT/'experiments/round7'
SOURCES=[
 dict(name='python-crfsuite / CRFsuite',url='https://github.com/scrapinghub/python-crfsuite',code='https://github.com/scrapinghub/python-crfsuite/blob/master/pycrfsuite/_pycrfsuite.pyx',license='Python wrapper MIT; underlying CRFsuite BSD',use='已安装0.9.12并实际训练；Trainer/Tagger序列标注。只学习本地训练标注，没有下载权重。',limit='成熟的是优化/序列标注实现，不是现成的数学关系理解能力。'),
 dict(name='sklearn-crfsuite',url='https://github.com/TeamHG-Memex/sklearn-crfsuite',code='https://github.com/TeamHG-Memex/sklearn-crfsuite/tree/master/sklearn_crfsuite',license='MIT',use='可选scikit-learn接口；本轮直接使用底层python-crfsuite，未安装此包装层。',limit='同样需要任务标注；包装器不能补足训练数据。'),
 dict(name='CogComp/arithmetic',url='https://github.com/cogcomp/arithmetic',code='https://github.com/cogcomp/arithmetic/tree/master/src/main/java',license='MIT',use='参考数量schema、单位依赖图、节点/边分类和关系消解的分解方式。',limit='Java/Maven系统；未运行或复制其模型，未下载其数据。依赖与资源预算需另行评估。'),
 dict(name='FastCoref',url='https://github.com/shon-otmazgin/fastcoref',code='https://github.com/shon-otmazgin/fastcoref/tree/main/fastcoref',license='MIT',use='参考实体提及和候选先行词配对的接口设计，未安装。',limit='现成FCoref/LingMess使用预训练模型；本项目禁止额外预训练权重，不能直接使用。'),
 dict(name='MWPToolkit',url='https://github.com/LYH-YF/MWPToolkit',code='https://github.com/LYH-YF/MWPToolkit/tree/master/mwptoolkit',license='本轮未核验具体许可证；未复制代码',use='可参考方程表示、模型/数据/评估分离的架构。',limit='未安装、未运行其默认实验或外部数据。它是研究框架，不是已验证适合本题预算的即用方案。'),
 dict(name='SymPy',url='https://github.com/sympy/sympy',code='https://github.com/sympy/sympy/tree/master/sympy/solvers',license='本轮未单独复核许可证；未复制代码',use='后续复杂方程求解可考虑；当前线性方程已有NumPy实现。',limit='负责解方程，不负责把自然语言可靠地转成方程。')]
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def esc(s):return html.escape(str(s))
def table(h,rows):return '<table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def main():
    comparison=read('audit_comparison.json');train=read('training_annotations.json');audit=read('audit_annotations.json');selection=read('training_selection.json');systems=comparison['systems'];baseline=ROOT/'experiments/round4/selected_bundle'
    for rel,info in json.loads((ROOT/'experiments/round4/asset_manifest.json').read_text()).items():assert hashlib.sha256((baseline/rel).read_bytes()).hexdigest()==info['sha256']
    emitted={}
    for name in systems:
        bs=[b for r in read(f'audit_{name}.json')['details'] for b in r['bindings']];e=[b for b in bs if b['prediction']['role']!='ABSTAIN'];emitted[name]=dict(emitted=len(e),correct=sum(b['correct'] for b in e),joint_precision_when_emitted=sum(b['correct'] for b in e)/max(1,len(e)))
    diag=dict(owner_candidate_recall=sum(g['owner'] in entities(r['question']) for r in audit for g in r['labels'])/sum(len(r['labels']) for r in audit),training_stage_counts=dict(Counter(g['stage'] for r in train for g,m in zip(r['labels'],r['mask']) if m)),emission=emitted)
    size=sum(p.stat().st_size for p in (OUT/'learned').iterdir() if p.is_file())
    decision=dict(extraction_gate_passed=comparison['extraction_gate_passed'],validation_read=False,test_access=False,selected='round4/selected_bundle',historical_validation_correct=750,historical_validation_n=798,additional_needed_for_99=41,old_assets_hash_verified=True,learned_assets_bytes=size,diagnostics=diag,reason='Learned mention binding does not meet preregistered quality gate; no deployment or validation tuning.')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2));(OUT/'github_research.json').write_text(json.dumps(dict(date='2026-10-14',sources=SOURCES,external_data_or_weights_used=False),indent=2,ensure_ascii=False),encoding='utf-8')
    body='<h1>第七轮：从标注学习关系绑定</h1><p>已实际训练CRF数量标签模型与实体/价格范围配对分类器；新封存的24道训练题在模型冻结后才打开并标注。<b>提取质量未过门槛，未读取验证集或测试集，没有替换原模型。</b>历史最佳仍为750/798（93.98%），距验证99%还差41题。</p>'
    body+='<h2>开源检索与可行性</h2><p>下列判断来自项目自己的仓库/源码说明。没有把“有GitHub代码”当作在本题上已验证有效；只有CRFsuite在本轮实际安装、训练、保存和重载。未下载外部数据或额外预训练权重。</p>'
    for source in SOURCES:body+='<p><a href="'+source['url']+'">'+esc(source['name'])+'</a> · <a href="'+source['code']+'">相关代码</a><br>'+esc(source['use'])+' '+esc(source['limit'])+' '+esc(source['license'])+'</p>'
    body+='<h2>标注与封存</h2><p>上一轮32题的手工语义事实通过唯一数值对齐，得到99处弱监督绑定；数值冲突的28处不强行标为负例。新增24道训练题，人工判读106处数量绑定。训练共56题、205处有效监督；内部42题训练、14题调参，最后全56题重训。另封存24道新审查题，冻结代码和模型后人工标注106处；总计涉及80道官方训练题。</p><p>新标注记录数量角色、所属实体、粗时间阶段，以及单价对应哪一个数量提及。它是完整约束图的子任务，不含完整单位推导、所求量或每个关系两端的时间方程。因此不能与上一轮完整事实精确率直接横比。两种方法在本轮使用同一子任务标注比较。人物/价格标注来自题目原文；同一助手标注，未经外部人工裁决。关键词分层可能包含“old computer”等非年龄题，并非纯年龄专用集。</p>'
    body+='<h2>实际训练了什么</h2><ol><li>CRF为按原文顺序排列的数量提及预测角色和时间阶段；输入为局部词、位置、货币/百分比标记，人物词面遮蔽。</li><li>实体配对分类器比较每个数量与原文生成的候选实体，学习所属人物；亲属实体只是候选，需要学习选择。</li><li>价格配对分类器预测该单价对应哪个数量/批次，也允许没有显式数量。</li><li>比较局部/扩展上下文、联合标签/分开的角色时间标签等9种配置。分开时间标签的方案对含非当前阶段的训练序列重复3次加权；未增加新的数据。只按内部训练留出集选择。</li></ol>'
    body+='<p>内部最终选择 '+esc(selection['selected']['name'])+'。训练选择不能事后依据审查集改选。审查中上下文消融和最近人物基线更好，也完整保留结果，未据此更新冻结模型。</p>'
    rows=[]
    for name,m in systems.items():rows.append([name,f"{m['joint_binding_accuracy']:.1%}",f"{m['role_accuracy']:.1%}",f"{m['owner_accuracy']:.1%}",f"{m['price_scope_accuracy']:.1%}",f"{m['stage_accuracy']:.1%}",f"{m['noncurrent_stage_accuracy']:.1%}",f"{m['complete_questions']}/24"])
    body+=table(['方案','联合绑定','角色','实体','价格范围','阶段总体','非当前阶段','完整提及绑定题'],rows)
    body+='<p>联合绑定主模型34/106，旧规则7/106。但旧规则只输出13处、其中7处全对（53.8%）；学习模型输出全部106处、其中34处全对（32.1%）。覆盖提高不等于输出更可靠。主模型的显式价格数量配对为6/8，但所有价格提及仅6/13；非当前阶段3/31，整体阶段分数主要来自“当前/未指定”。</p>'
    body+='<h2>瓶颈与可靠性</h2><p>实体候选生成包含98/106个金标所属实体；缺失的speaker、tree/boy和某些宠物实体无法由后续排序补救。即使候选存在，学习式人物绑定仍低于最近人物基线（63.2% vs 70.8%）。这不支持宣称已经学会指代消解。</p>'
    body+='<p>205处训练监督中，当前/未指定阶段为168处，未来只有11处、过去3处。单纯用整体准确率选模型会掩盖时间阶段失败。手工新增标注不足以覆盖人物别名、跨句指代、批次价格、复杂时间端点；旧标注的数值自动对齐也是弱监督，不是人工复核后的全部金标。</p>'
    body+='<p>本轮检查门槛在审查之前固定：联合绑定、实体、价格范围及阶段均≥95%，至少12/24题所有提及绑定正确。实际仅2/24题，故不进入验证集。即便将来通过这个子任务，也还需核对完整目标/单位/时间方程才适合用来改答。下一轮若使用这24题开发，必须封存新的审查题，不能重复把它们当未见样本。</p>'
    body+='<h2>资源与复现</h2><p>python-crfsuite 0.9.12；只用CPU、四线程环境；本轮最终搜索训练约 %.2f 秒（不含标注和首次探索），主模型资产 %d 字节（%.3f MiB）。24道审查题提取约 %.3f 秒，仅代表这批提取，不是完整官方评估或30秒预算验证。原最佳资产哈希已核对。许可与AI协助需要在最终提交中披露。</p>'%(selection['seconds'],size,size/1048576,systems['learned']['seconds'])
    body+='<h2>逐题审查（分析专用）</h2><input placeholder="搜索题号或题目" oninput="document.querySelectorAll(\'.case\').forEach(e=>e.hidden=!e.textContent.toLowerCase().includes(this.value.toLowerCase()))">'
    for r in read('audit_learned.json')['details']:
        body+=f'<details class="case"><summary>#{r["index"]} · {esc(r["question"][:130])}</summary><p>{esc(r["question"])}</p>'+table(['提及','数值文字','金标','预测','全部正确'],[[b['mention'],b['raw'],b['gold'],b['prediction'],b['correct']] for b in r['bindings']])+'</details>'
    dest=OUT/'report';dest.mkdir(exist_ok=True);(dest/'实验报告.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>MP2 第七轮学习关系绑定</title><style>body{font:16px/1.75 system-ui;max-width:1240px;margin:32px auto;padding:0 24px;color:#17364a;background:#f5f8fb}h1,h2{color:#146879}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}td,th{border:1px solid #cad8e2;padding:8px;text-align:left;overflow-wrap:anywhere}details{background:white;padding:14px;margin:14px 0}summary{cursor:pointer}input{padding:12px;width:90%}</style>'+body+'</html>',encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'research').glob('round7_*.py')};(OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(decision))
if __name__=='__main__':main()
