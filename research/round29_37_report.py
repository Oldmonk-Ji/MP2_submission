"""Freeze completed semantic experiments and report full-population evidence."""
from data_access import ROOT
from round22_common import read,dump,sha
from pathlib import Path
import ast,datetime,json,sys

METHODS={29:'题干到完整算式',30:'上下文数量指针',31:'经执行检查的错误算式对照',32:'数量作用与两两运算关系',
         33:'人物身份标准化的关系模型',34:'局部数量/目标词稀疏对齐',35:'关系辅助监督与官方表示联合训练',36:'题干与候选全文循环编码对齐',37:'训练与推理一致的全部候选算式评分'}


def closure(files):
    seen=set();todo=list(files)
    while todo:
        f=todo.pop().resolve()
        if f in seen or not f.exists():continue
        seen.add(f)
        for node in ast.walk(ast.parse(f.read_text(encoding='utf-8-sig'))):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module]
            for name in names:
                p=ROOT/'research'/(name.split('.')[0]+'.py')
                if p.exists() and p.resolve() not in seen:todo.append(p)
    return seen


def main():
    reports=[];best_number=25;best=read(ROOT/'experiments/round25/selection.json')['selected']
    for number in METHODS:
        out=ROOT/f'experiments/round{number}';selection=read(out/'selection.json');winner=selection['selected']
        assert all(r['n']==798 for r in selection['records'])
        noncontrol=[r for r in selection['records'] if r['name']!='control'];trial=max(noncontrol,key=lambda r:(r['correct'],-r['damaged'],-r.get('alpha',0)))
        report=dict(round=number,method=METHODS[number],selected=winner,best_noncontrol=trial,
            configurations=len(noncontrol),baseline_correct=752,validation_n=798,test_opened=False)
        reports.append(report)
        if winner['correct']>best['correct']:
            assert (out/'verification.json').exists(),f'round {number} must reproduce official validation first'
            best_number=number;best=winner
    training_checks=[]
    expected=read(ROOT/'experiments/round22/plan.json')['training_ids']
    for number in [29,30,31,32,33]:
        for path in (ROOT/f'experiments/round{number}/full').rglob('training_ids.json'):
            actual=read(path);assert actual==expected;training_checks.append(str(path.relative_to(ROOT)))
    for number in [34,35,36]:
        path=ROOT/f'experiments/round{number}/training_ids.json';assert read(path)==expected;training_checks.append(str(path.relative_to(ROOT)))
    costs=[]
    for number in METHODS:
        for p in (ROOT/f'experiments/round{number}').rglob('complete.json'):
            d=read(p)
            costs.append(dict(path=str(p.relative_to(ROOT)),seconds=d.get('seconds'),n=d.get('n')))
    for number in [35,36]:
        for p in (ROOT/f'experiments/round{number}').glob('*_history.json'):
            h=read(p);costs.append(dict(path=str(p.relative_to(ROOT)),seconds=h[-1]['seconds'],n=4743,epochs=h[-1]['epoch']))
    summary=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rounds=reports,
        recommended_round=best_number,validation_correct=best['correct'],validation_n=798,validation_accuracy=100*best['correct']/798,
        hard_correct=best['hard_correct'],hard_n=355,hard_accuracy=100*best['hard_correct']/355,
        test_opened=False,test_accuracy=None,goal_97_achieved=False,training_n=4743,validation_exclusions=[],
        all_final_model_training_ids_checked=training_checks,training_costs=costs,
        cost_caveat='Recorded stage wall times overlap on the same single RTX A2000 Laptop GPU; do not sum them as elapsed time. Preprocessing and validation selection are additional.',
        environment=read(ROOT/'experiments/semantic_environment_20261016.json'),
        validation_caveat='Official validation has been repeatedly reused for development; this is not an untouched generalization estimate.')
    dump(ROOT/'experiments/semantic_continuation_20261016.json',summary)
    lines=['语义路线继续优化记录：第29—37轮','日期：2026-10-16','',
        f'当前推荐：第{best_number}轮；完整官方验证{best["correct"]}/798（{100*best["correct"]/798:.6f}%）。',
        f'不同答案冲突子集：{best["hard_correct"]}/355（{100*best["hard_correct"]/355:.6f}%），仅作诊断。',
        '测试集未读取、未用于调参，暂无新测试成绩；最终测试超过97%的目标尚未完成。',
        '不能把验证正确率、冲突子集正确率或合成算式的自洽率当作测试成绩。','',
        '一、完整验证结果（相对第25轮752/798；修复/误伤按官方最终数值评分）']
    for r in reports:
        s=r['selected'];t=r['best_noncontrol']
        lines.append(f'第{r["round"]}轮 {r["method"]}：选中{s["name"]}，{s["correct"]}/798；非零新模块最佳{t["correct"]}/798，修复{t["repaired"]}、误伤{t["damaged"]}；比较{r["configurations"]}项。')
    lines+=['','二、数据与监督边界',
        '每个最终模型使用全部4743道约定训练题，仍只排除已登记12题。没有删验证难题、拒答或测试调参。',
        '第34轮读取全部4743题；候选全部正确的组没有正负成对排序约束，不额外制造负标签。',
        '第29—33轮先做同题/规范化题干模板隔离的3795/948训练内诊断，随后全量重训。诊断模型不用于最终提交。',
        '算式抽取保留重复数值的不同出现位置；存在多种来源绑定时标记不确定。缺失或不确定的关系不自动判为错误。',
        '2488道训练题通过较严格的抽取筛选；clean只是抽取启发式，不等于经独立人工证明语义正确。',
        '构造7463条绑定、方向、运算符或遗漏变体，逐条执行并核对其结果不等于原训练答案。同源变体与原题同折。',
        '这些派生输入是算式，不是新增自然语言题，不给改写输入沿用旧官方embedding。',
        '第35轮为端到端联合训练，不用训练集拟合后的评分再另行训练堆叠融合层；辅助监督开关作匹配对照。',
        '第36轮只使用官方原始题干、候选全文和原始固定表示，词表与标准化仅由训练集拟合；没有外部题库、额外预训练权重或推理API。',
        '第36轮是有/无新增文本通道对照，不是严格去题目消融：两个版本都保留官方固定候选表示。','',
        '第37轮纠正候选集合不一致：训练候选排序损失覆盖全部四候选，而第29—33轮推理过滤了无法可靠抽取算式的候选评分。',
        '训练内诊断中，第30/31轮不加解析掩码的裸分选择分别747/948和761/948；这不是主模型融合成绩，也不能与加掩码诊断直接混比。',
        '第37轮的未解析答案字面量可能携带数值先验或抽取质量信号，即便提分也不能独自证明语义理解。',
        '三、为什么不能把小修补称为解决了语义理解',
        '第25轮46道验证错题中，严格算式抽取仅9道有至少两种不同的可比较算式；放宽抽取后为29道。',
        '已完成的第29/30/31/32/34轮所有配置合计只曾修复其中9道；这是事后覆盖诊断，不是可部署的按题选择器。',
        '这说明完整算式抽取和小型局部关系模型仍有覆盖率与判断质量双重瓶颈。不能继续只遍历融合权重就宣称接近97%。',
        '第36轮因此直接对齐题干和候选文本。训练题干无截断，候选词元保留99.568%；验证候选词元保留99.515%，少量长候选取首尾。',
        '完整算式模型的结构精确率使用teacher forcing，不是自由生成正确率。所有验证结果都是复用开发集证据。','',
        '四、复现入口',
        'Python：D:/anaconda3/envs/mp1/python.exe；PYTHONPATH需包含本项目.venv/Lib/site-packages。',
        '第29轮：round29_prepare.py；round29_train.py diagnostic/full reference_only/reference_and_candidate_ranking；round29_select.py 29。',
        '第30轮：round30_train.py，相同两个阶段和变体；round29_select.py 30。',
        '第31轮：round31_prepare.py；round31_train.py diagnostic/full；round29_select.py 31。',
        '第32/33轮：round32_train.py或round33_train.py diagnostic/full；round32_select.py 32或33。',
        '第34—36轮：分别运行round34_run.py、round35_run.py、round36_run.py。准备脚本要求输出目录不存在，不能覆盖冻结结果。',
        '第37轮：等第29—33轮全部全量模型完成后，运行round37_run.py。',
        '训练监督检查：research/round29_35_audit.py。新增胜出模型验证：research/round29_35_verify.py <轮号>。',
        '上述核验器只调用官方validation预测/评分；测试仍留给方案冻结后的最终评价。','',
        '机器可读总表：experiments/semantic_continuation_20261016.json。',
        '监督检查：experiments/semantic_supervision_audit_20261016.json。',
        '每轮selection.json记录全部配置，每轮completion_manifest.json记录代码与实验文件SHA256。',
        '本轮AI辅助包括方案设计、编码、算式派生、诊断与报告；提交时需如实披露。']
    (ROOT/'语义路线继续优化_2026-10-16_第29至37轮.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    seeds=[p for p in (ROOT/'research').glob('round*.py') if any(p.name.startswith(f'round{n}_') for n in range(29,38))]
    seeds+=[Path(__file__),ROOT/'research/round29_35_verify.py',ROOT/'research/round29_35_audit.py']
    sources=closure(seeds)
    for number in METHODS:
        out=ROOT/f'experiments/round{number}';files=[p for p in out.rglob('*') if p.is_file() and p.name!='completion_manifest.json']
        files+=list(sources)
        for p in (ROOT/'experiments').glob(f'round{number}*.log'):files.append(p)
        dump(out/'completion_manifest.json',dict(completed=True,goal_97_achieved=False,test_opened=False,
            hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
