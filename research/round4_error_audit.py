"""Human/AI-reviewed validation diagnostics, kept entirely outside inference."""
from data_access import ROOT
import json
from collections import Counter
from round4_quantities import trace

OUT=ROOT/'experiments/round4'
GROUPS={
    '实体与关系方向':[17,44,97,137,161,171,175,325,412,444,521,550,613,633],
    '遗漏或重复计算范围':[77,133,206,210,275,328,359,362,371,454,469,473,497,586,617,655,726,788],
    '回答了不同的所求量':[48,330,331,481,583,679,728],
    '单位时段或百分比基数':[209,224,413,739],
    '局部数值计算':[52,65,522,719],
    '题目措辞与参考解不一致':[32,661],
    '官方解析标记与展示答案不一致':[733],
}
NOTES={
17:'把已知的第三次 92 分重复代入两个成绩，随后把单次未知成绩当作前两次平均。',
32:'题目写 1/4 more，按通常理解为原数的 1.25 倍；参考只用了 1/4 倍。保持官方标签 9，标为措辞争议。',
44:'46 对应成对数量，不能从总人数 94 中只减去 46；应计算男女数量差。',
48:'把总收入/单笔差额混作利润，并再次乘以股数；应包含剩余持仓价值并扣除原始成本。',
52:'0.6/9 被算成 2/15，错误中间值继续传播到 8 分钟。',
65:'3+5+7+9+11 被写成 36。',
77:'算出一名保镖的周薪后，漏乘两名保镖。',
97:'把卖球员收到的钱也从余额中扣除，现金流方向反了。',
133:'用每加仑覆盖面积 400 代替需要涂的总面积 1600。',
137:'凭空把第二项百分比的基数换成 950，而不是 300。',
161:'剩余目标长度、捐赠量和当前长度之间的加减关系错误。',
171:'混淆现有青蛙、存活蝌蚪和容量，甚至把未存活的数量称作变成蝌蚪的青蛙。',
175:'把姐姐是 Lexie 的两倍误读成 Lexie 是姐姐的两倍。',
206:'重复计票价，并在最后计算中忽略已存的钱。',
209:'把每周工作日夜晚当成 2 晚，把一周当成 4 晚。',
210:'算出剩余 180 秒后漏掉领先对手的 30 秒。',
224:'明确声称一小时有三个半小时。',
275:'已经算出增加班级后的全部学生，又把原来的学生加了一次。',
325:'混淆父亲、子女、孙辈之间的归属，最后 1 没有方程支持；正确候选的推理也不完全可靠。',
328:'Manny、24 名同学、老师被合计为 25 人，漏计一人。',
330:'将销售收入 980 作为利润，未扣毛线成本 672。',
331:'算出中间兄弟的年龄 32 后当成最小年龄；未利用连续整数差 1。',
359:'最终汇总漏掉周二，还把周末每天 5 错写成每天 10。',
362:'题目 125 已是收礼后的总额，再加一遍礼金导致重复计算。',
371:'只把增加额乘 12，原有月费没有乘 12。',
412:'把前三行误读为第 1 行和第 3 行，共两行。',
413:'把 8 盎司当作 1 磅，且汇总中遗漏一项水果。',
444:'把回家距离当作未知的去市场距离，错误建立 50+2x=140。',
454:'三次 more 往返应加上第一次，只算了新增三次。',
469:'计算一盘饼干后漏乘四个烤盘。',
473:'已算出的上身 12 平方英尺没有加入总材料面积。',
481:'回答第二次降价后的价格 456，题目问总降幅 44。',
497:'父母与三个兄弟姐妹被算成四人，且漏了主人本人。',
521:'将 Samir 现在的 15 岁错放到十年前，又多加了十年。',
522:'5+15+8+11+10+5 被算成 63 而非 54。',
550:'把幼虫时间为蛹时间三倍的方向颠倒。',
583:'回答全部篮子的剩余苹果 52，题目问每篮 13。',
586:'总出席人数只计男生 70，漏了已经出席的女生 140。',
613:'将两种文具之间的五倍加十二关系方向颠倒。',
617:'回答第二阶段 6 周，未加第一阶段 4 周。',
633:'分摊自己的半账单时用个人餐价 16，应该用总账单 30。',
655:'总价中只加入第一批一幅画的价格，漏掉另两幅。',
661:'题目写 at 20% of，通常应付 120；参考按 20% off 算 480。所选 720 两种解释下都错，但不能称 120 的推理必错。',
679:'回答对手一周总里程 7，没有扣掉自己已经跑的 6。',
719:'5+20+15+10 被算成 40。',
726:'每个半段都使用全部十季，重复计了总季数。',
728:'只计算 Gretel 的加薪，没有把 Hansel 的加薪纳入比较。',
733:'候选展示的最后 boxed 答案为 250，但更早的最后一个 #### 标题含 3；官方优先规则解析为 3。',
739:'题目明确按原始本金计算利息，候选使用复利，且 5600+392 又算成 6000。',
788:'只答新孵化的 45 只，漏掉原来的 20 只。',
}


def main():
    if not (OUT/'error_audit_raw.json').exists():
        from data_access import CODE,load,read_jsonl
        from features import metadata
        import numpy as np
        bank,_,y=load('validation');meta=metadata(bank)
        scores=np.load(ROOT/'experiments/round3/stack_scores.npz')['validation']
        chosen=np.where(meta['valid'],scores,-1e30).argmax(1)
        refs=read_jsonl(CODE/'data/gsm8k_validation.jsonl');raw=[]
        for i in np.flatnonzero(y[np.arange(len(y)),chosen]==0):
            good=int(np.flatnonzero(y[i])[0]);bad=int(chosen[i])
            raw.append(dict(index=int(i),question=bank[i]['question'],answers=meta['answers'][i],selected=bad,
                correct_candidates=np.flatnonzero(y[i]).tolist(),selected_text=bank[i]['candidates'][bad]['text'],
                one_correct_text=bank[i]['candidates'][good]['text'],reference=refs[i]['answer']))
        (OUT/'error_audit_raw.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=json.loads((OUT/'error_audit_raw.json').read_text(encoding='utf-8'))
    category={i:k for k,ids in GROUPS.items() for i in ids}
    assert set(category)==set(r['index'] for r in rows)==set(NOTES)
    for r in rows:
        r['primary_category']=category[r['index']];r['review_note']=NOTES[r['index']]
        t=trace(r['question'],r['selected_text'])
        r['selected_trace_diagnostics']={k:t[k] for k in ('root_found','steps','bad','collisions','unexplained','boxed_parse_disagreement')}
    result=dict(method='Manual/AI diagnostic review of all 50 errors from frozen round3; primary category is interpretive, multiple causes can coexist.',
        not_for_training_or_inference=True,official_labels_unchanged=True,count=len(rows),
        category_counts=dict(Counter(r['primary_category'] for r in rows)),rows=rows)
    (OUT/'error_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['category_counts'],ensure_ascii=True))


if __name__=='__main__':main()
