from round22_common import ROOT,read,dump,sha
if __name__=='__main__':
    assert read(ROOT/'experiments/round199/feature_checks.json')['test_opened']==False
    rows=read(ROOT/'experiments/round199/validation_records.json')
    for r in rows:r['held']=read(ROOT/f"experiments/round199/{r['arm']}_held.json")['correct']
    best=max(rows,key=lambda x:x['correct']);report=dict(results=rows,fit_changes=read(ROOT/'experiments/round193/fit_changes.json')['counts'],fit_reference_coverage=1658,old_fit_reference_coverage=1623,current_recommendation='round104' if best['correct']<=759 else 'requires_frozen_feature_validation',test_opened=False,goal_active=True,target_test_correct=402)
    dump(ROOT/'experiments/context_binding_round191_199_summary.json',report)
    lines=['对象、单位与时间状态绑定核验（第191至199轮，2026-10-26）','仅从训练拟合部分制定规则并冻结后验证。测试集未打开。','同值训练拟合题944道；审查明确单位、对象和时间线索。只有存在明确兼容解释时，排除明确冲突；未知绑定保留竞争解释。','训练拟合参考等价正确程序覆盖1623→1658/2084。该指标不是人工语义正确率。','匹配消融（训练留出/1582，官方验证/798）：']
    lines.extend(f"{r['arm']}: {r['held']}/1582，{r['correct']}/798，资产{r['asset_bytes']}字节。" for r in rows)
    lines+=['当前104对照1477/1582、759/798。官方验证严格超过759且通过冻结检查后才进入测试。','程序缺失不直接判错；参考程序不一致没有作为负标签。所有分折训练保留父题隔离。','导出权重、全部验证新增特征、新输入重算、小批量预测、分批不变性及空关系输入检查通过。','本阶段全部训练进程结束。86%目标尚未宣布达成；此前冻结测试391/467=83.7259%不是本阶段新成绩。']
    (ROOT/'对象单位时间绑定核验_2026-10-26_第191至199轮.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(ROOT/'experiments/next_after_round199.json',dict(**report,next_start='If no VAL improvement: audit TRAIN-fit actual candidate equation spans. Current annotations are line-level; multiple same-valued operands on one line collapse context and retain ambiguity. Design occurrence-specific operand spans and result-state scopes using fit examples; do not strengthen hard pruning from held errors. Freeze before held. If VAL improves: run standard frozen feature_validation with-test first.',resource_policy='Laptop33s; official reference30s,four threads,16MiB unchanged.'))
    print(report)
