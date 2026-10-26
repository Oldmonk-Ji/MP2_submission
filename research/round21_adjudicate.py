"""Question-quality adjudication, written before subgroup scores are computed.

These are AI judgments for a post-hoc sensitivity analysis, not new gold labels.
Only clear defects and material ambiguities enter quality masks. Debatable cases
stay in that view. No model prediction or correctness is read here.
"""
from round21_audit_quality import OUT, ROOT, read, dump, sha, stable
import ast
import datetime
import math
import operator

# Independently checked quantities from question text. For ambiguous cases this
# is the official interpretation, which is NOT asserted to be uniquely correct.
CALCULATIONS = '''
3:20*(20+25)
9:160-12*9
10:(115-40)/15
13:8*3/4*60
16:2*12+16+10
28:21*4-10-5-3
29:(67+5)/4*3-5
32:14*5+14*3
34:25*2+6
38:4*3*75-75
39:18+2*4-1-24
42:3*2+6+8
46:1.50+4.35+12.65-6.50
48:1440/(12*12*10)
50:6*60/90*10
56:16+3*2+6*1
60:60-60/3-60/4
63:5+2+2*2+20/2-3
65:8*5+4*5*1.2
69:3+11-10
72:(40-2*4-4*6)/4
75:1500+5*800+5*800
76:11+6+4
77:80*(1-.4)+5
78:900*(1-3/4)*(1-5/9)-80
83:107-23*2
115:6*4/12+1
218:2*24-(2+4)*7
225:30*.45/.50
234:(16+16-1-7)/4
244:240*.5/24*12
252:3*(15+12)*2*5
313:2+6+2+4-16/2
341:4640/(40*20+(52-40)*20*1.5)
343:9450/(15*6)*19*5
374:12*2-6-8-(8-2)
426:(11+14+32/2)*6-99
452:(8+6*2+2*12)/2
524:100+(74+2*2+42-100)*.9
540:2*(225+125)-3-10
592:(3*.5+2)*8
618:20000*1.1
639:(30-20)/2-1
647:12+12*3+6+5
649:4+2*4+2*(4+2*4)
658:10*4-4*1
683:9*2+2*9*1.5
694:500*(1-.4)
734:(3*11-4)*2
756:1.5*20+1.5*2*20*50
786:100/20*4
788:400+(400-70)+2*(400-70)
807:2*4*(7-2)*52
811:100000*(1-.2)*(1-.3)
825:(103-27-(27-7))/2
912:(5*100+200+300)*3
918:40*3+40+60
921:(4+2)*3-2*2
1027:100-16*6
1046:(7+4+4+7)*2
1055:12006/2*(1-1/3)-1000-1000/5
1058:2*(5+10)
1059:(100-7*8)-8*5
1147:(5000-2000)*.3
1172:(12+3*12+46+250)*.25/2
1282:32/4+32*3/8+32/2+4
1299:10+10*3+10*3*5
1317:6*3/6
1385:200*10*20-2000
1393:21122*2+3922
1432:2000*(1-2/5)*(1-2/5)*(1-.4)
1442:400-5*20-3*12-4*30-5
1445:63-60
1461:(200*12+40*14+60*17)*8
1505:12+20+10+2*10
1507:60/(1+2)
1512:200+200*3/4+(200+200*3/4)/2
1602:200*3/4*2
1608:20*(1-.3)+4
1616:4*10*2*5
1661:6*10-4*10
1665:14+2*14-(6+2*6+(6+2*6)/3)
1707:(15-15/3+15/3*2)*3
1726:(30+20)*2
1754:(2+8)/2
1795:(5+2*5)*10
1838:1.3*60-30
1854:(96/8+2)*(8+2)
1888:18*16*7
1954:3/(3/4)
1995:4560/38/60
2018:(10+10-2)*(30-8)
2030:20+20/2+3/6*60
2059:5*2+5*5
2097:60+60*.8
2146:4+20+2*7
2167:50+50/2+(50/2+40)
2168:2*60-15-30-30
2174:950*.94-920*.95
2184:(4+7+4+9)*(2+2)
2268:20*(25+4*10+10*5+15*4)
2317:12+24+(24-6)+(12+9)
2334:4*5+3*2+10*2*2
2342:6*4/2-6
2351:6*200*3/400
2368:(12.5+2*12.5+(2*12.5-8.5))*60
2388:8*2+18+18*9
2392:(2+1)*5*4/30
2403:3*5/5
2413:49/(4+3)*4*3
2415:72-3*(12+7)
2429:(1/3+1/4)/(1/12)
2453:100-(4+2*4)*2-.5*10
2513:((100+20)/2+10-30)/2
2594:18-8-8
2620:(30/2*2+30/2)/5*(4+4)
2642:66-15*4
2665:5*20+15*(20-2)
2682:90/3/3+(90-90/3)/2
2697:150-7*10-4*15
2708:(20*1.5+10*2+20*1)*5
2727:(30/3-5)/5
2752:2*100+2*100+2.5*100
2779:60+4*60+(60+4*60-30)
2857:26+15+24-4-6-2*6
2912:2*4+3*(4/2)
3010:16+8+4+2
3045:1500*7*150
3051:5*(20*.8+12)
3060:(7*2+7+3)*1.5
3063:5*8*.5
3078:(100-23)/7
3142:354-63-2*63-(2*63+10)
3197:(2*1.30-2.50)*10
3243:2*12*6*7
3249:(20+40+10+30+50)/10
3279:14/2*12
3325:42-42/3-2
3415:40/2+10
3435:250*.8*1.4
3476:50-2*3-2*10
3516:(9*8-40)/8
3656:320*(1-1/4)*(1-1/6)*(1-.3)
3673:4+100*(4-.5)
3679:40*(1-2/5)*2
3681:200-200*(1-.7)*2
3744:50*8*14*5
3807:300*(90-30)/(300-100)
3824:100000/(150*10-100000*.01)
3830:5*3+5-12
3922:493-125-223
3951:2/10*12*100
3961:20+20/2+20
4045:1936+2*(1936-1932)
4052:16*.5
4171:4*3+2*3
4174:50*1.1+3*5
4239:25+25*2+25*2/2
4242:150*1.5*2+150/2
4296:80000*(1+.2+.05)
4316:25*2/5+(25-25*2/5)*(1-1/3)
4331:4*10*(1-.4)*3
4368:(2+2*2)*100*.95
4382:(100+800+(800-200))/3+25
4414:240/10/2
4423:5*(30+20)
4446:12*2+20
4468:(81-2*8)/5
4540:5*12/4
4578:38*2+75*3
4593:6+6/2+(2*(6/2)-2)+(2*(2*(6/2)-2)+6)
4632:30*.5+28*.75+20*.8+13
4633:6/2*(2+1+3)
4644:2**12
4657:22+(4*22-3)
'''

# Explicit text evidence; candidates alone never establish a question defect.
SPECIAL = {
34: ('defect', '关系方向倒置／整数约束冲突',
     '原文将集市票数 F 与棒球票数 B 关联为 F=2B+6。F=25 得 B=9.5，既不支持参考答案 56，也违反整数票数。参考解改成了 B=2F+6。',
     ['(25-6)/2', '25*2+6']),
56: ('ambiguous', '单价与整批价格未区分',
     'three binders for $2 / six notebooks for $1 没有 each。按整批价格总计 19；按每件价格总计 28。四个候选全部自行加了 each，不能证明原文已给出。',
     ['16+2+1', '16+3*2+6']),
1055: ('uncertain_keep', '储蓄与投资是否重叠（保留）',
       '参考解把存款 1000 与投资 200 当成两笔支出，可得 2802。若投资来自已有储蓄则为 3002；原文未写 from savings，暂沿常见分项支出读法保留，不作为确定缺陷剔除。',
       ['4002-1000-200', '4002-1000']),
1442: ('ambiguous', '裤子价格作用范围缺 each',
       '裙子和夹克明确 each，裤子只写 the pants cost $12。三条合计 12 时余款 163，每条 12 时余款 139；原文没有明确排除前一种读法。',
       ['400-100-12-120-5', '400-100-3*12-120-5']),
1445: ('defect', '所问差值方向与参考答案矛盾',
       'Lyssa 答对 60，Precious 答对 63。问 Lyssa 比 Precious 多多少，有向差是 -3（即少 3），参考却称 Lyssa 多 3。保留原标签，仅在敏感性子集排除。',
       ['75*.8-(75-12)', '(75-12)-75*.8']),
1854: ('ambiguous', '扩建发生在每条边还是每个尺寸',
       '原房间 8×12。若每条边向外延伸 2，两维各增加 4，面积 192；参考把长宽分别增加 2，面积 140。each side 的几何范围没有澄清。',
       ['(8+4)*(12+4)', '(8+2)*(12+2)']),
2174: ('defect', '缺少币种换算条件',
       '折后是 893 美元与 874 欧元。直接相减得到 19 需要未给出的 1:1 汇率。按 1 或 1.1 美元/欧元，价格差就不同，因此不能唯一确定题目所问的价差。',
       ['893-874*1', '893-874*1.1']),
2351: ('ambiguous', '取回动作的结束阶段未说明',
       '到土豆处单程为 6×200×3/400=9 分钟；fetch 通常还包括带回，往返为 18。参考只算单程。题设球场长度虽不常见，但应接受给定 200 码，不以常识改数。',
       ['6*200*3/400', '2*6*200*3/400']),
2665: ('uncertain_keep', '折扣阈值用词有瑕疵（保留）',
       'over $20 的措辞不规范；但 first 5 与 each additional 的对照仍强烈指向前五本 20、之后 18，总计 370。暂无另一套同样完整的定价条件，暂保留并标疑。',
       ['5*20+15*18']),
2912: ('ambiguous', '每座山次数与整组次数',
       '高山明确每座 4 次。小山 half as often 未补 each：按每座一半是 8+3×2=14；按整组小山次数为整组高山一半是 8+4=12。这是作用范围歧义，不是仅因模型易错。',
       ['2*4+3*(4/2)', '2*4+(2*4)/2']),
3063: ('uncertain_keep', '缩小比例的量纲（保留）',
       '若缩小的是体积，40×0.5=20；若三维长度均减半，体积为 5。原题问流体体积，沿常见题目直接对所问量应用比例，暂保留；不把科幻设定本身当缺陷。',
       ['5*8*.5', '5*8*.5**3']),
4242: ('uncertain_keep', '负重是否包含体重（保留）',
       '按搬运的外部负重，双手 450 加背心 75 为 525；含自身体重则为 675。语境倾向搬运负重，暂保留，不因较弱的替代读法剔除。',
       ['150*1.5*2+150/2', '150*1.5*2+150/2+150']),
4296: ('ambiguous', '房价与含佣金总收入混用',
       '按 sold it for a 20% profit，房屋成交价为 96000；再收到按原价计算的佣金 4000，总收入为 100000。问题问房子卖多少钱，参考将总收入作为房价；未明确佣金是否已包含在售价／利润中。',
       ['80000*1.2', '80000*1.2+80000*.05']),
4331: ('defect', '实体状态词写反',
       '题干说其余变成 caterpillars，而参考改成 butterflies 并出售。40% 没变成蝴蝶与其余仍为毛虫的表述不能支持明确的 24 只蝴蝶；72 依赖修正题干物种词。',
       ['4*10*.6*3']),
4382: ('defect', '自指约束无解',
       '第三人比第三人少 200，形成 x3=x3-200，无解。参考擅自把第二个 third 改成 second，才得到 525。',
       ['(100+800+(800-200))/3+25']),
4578: ('uncertain_keep', '计数对象与计数轮次（保留）',
       '数每个东西算一次时为 38×2+75×3=301；整轮算一次则为 5。但题末明确 every time ... counts something counts as once，较支持逐对象读法，暂保留。',
       ['38*2+75*3', '2+3']),
4644: ('ambiguous', '十二月新增与截至十二月累计',
       '每月存款依次倍增，十二月这一笔是 4096，全年截至十二月累计为 8190。by December 与 have to spare and save 混用累计和当月可存额，参考只取当月额。',
       ['2**12', '2*(2**12-1)']),
}

NOTES = {
3: '按两人合计劳动时间；不擅自假定同时开始或改问墙钟耗时。',
218: '按题干两组人各合计 2、4 个鸡蛋，48-42=6；没有 each，不把每组量重复乘人数。',
234: '共吃 24 片，四人各 6；唯一只吃意大利香肠的人吃 6，其余三人各吃两种口味各 3，全部剩余约束一致。',
252: '20 年只标阶段，不是每年产量乘数。',
426: '沿题目常见表述，把半片草地的除草量当该区杂草的一半。',
756: '50 times more expensive 按常见倍数价格 50 倍；不擅自加成 51 倍。',
807: '沿任务常见每年 52 周换算；不因非闰年非整周而剔除。',
1027: '100 美元最多买 16 本 6 美元漫画，余 4。',
1058: 'shoes 按一双两只的通常语境解释。',
1059: '余额 44，最多取 8 张五元钞，留下 4；不能把取款前余额当所求量。',
1282: '按每人 32 颗牙；人类牙齿的常见语境与后续 his teeth 一致。',
1661: '已还 60 是总借款的一半，当前欠 60，未来四个月后欠 20。',
1954: 'Jon/John 按同一人的显然拼写变体；未据此剔除。',
2097: '虽总行程超过自然日时长，代数关系和目标明确；不将夸张背景直接当标签错误。',
2413: 'Marta/Martha 按同一人的拼写变体处理。',
2453: 'already bought 之后又要求 buy twice as many，按新增 8 瓶，总数 12；不把已有与新增合并错。',
3051: '主句订单给出总数 5 个背包，按通常分配给孙辈的读法。',
3435: '250 磅是装满沙的容量基准；体积装 80%、密度 1.4 倍。',
3476: '每个孩子随身带 3 个苹果，共 6；two teachers 并未说明每人 3 个。',
3516: '按每包用完才开下一包的通常操作，发出 32 个等于 4 包。',
3673: 'after that 表示后续每只统一减 0.5，并非逐只递增折扣。',
4174: '沿题目叙述顺序：一次事故先涨 10%，再加三张罚单的固定费用。',
4423: '油耗虽高，但题设每公里 5 升明确，不能以现实常识重写题目。',
4446: '按通常 twice as old 读法，12×2+20；不因口语 2 times older 一律剔除。',
4540: '五件连衣裙共 60 小时，每周 4 小时需 15 周；明确难题保留。',
}


def arithmetic(expr):
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
    def walk(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](walk(n.left), walk(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -walk(n.operand)
        raise ValueError('Only constant arithmetic permitted')
    return float(walk(ast.parse(expr, mode='eval').body))


def main():
    assert not (OUT / 'frozen.json').exists()
    packet = read(OUT / 'review_packet_no_predictions.json')
    calcs = {int(i): expr for i, expr in (line.split(':', 1) for line in CALCULATIONS.strip().splitlines())}
    assert set(calcs) == {r['train_index'] for r in packet}
    rows = []
    for r in packet:
        i = r['train_index']
        decision, category, reason, alternatives = SPECIAL.get(i, ('keep', '未见足够的排除依据', NOTES.get(i, '按题干关系计算，与参考最终数值一致；候选中有错误属于正常任务，不据此删除。'), []))
        value = arithmetic(calcs[i])
        assert math.isclose(value, float(r['reference_answer']), abs_tol=1e-7), (i, value, r['reference_answer'])
        rows.append(dict(**r, decision=decision, category_zh=category, reason_zh=reason,
            reference_interpretation_calculation=calcs[i], reference_interpretation_value=value,
            independent_interpretations=[dict(expression=e, value=arithmetic(e)) for e in alternatives],
            full_reference_and_four_candidates_inspected=i in SPECIAL,
            reviewer='Codex AI; not external gold; no model prediction/correctness displayed during review'))
    dump(OUT / 'adjudication.json', rows)
    p = read(OUT / 'plan.json')
    ranking = read(OUT / 'uncertainty_ranking.json')
    cap = p['maximum_removed']
    defect = sorted((r['train_index'] for r in rows if r['decision'] == 'defect'), key=stable)
    ambiguous = sorted((r['train_index'] for r in rows if r['decision'] == 'ambiguous'), key=stable)
    quality = (defect + ambiguous)[:cap]
    uncertain = [r['train_index'] for r in ranking]
    combined = (quality + [i for i in uncertain if i not in quality])[:cap]
    masks = dict(full=[], defects_only=defect[:cap], defects_and_ambiguities=quality,
                 uncertainty_05=uncertain[:math.floor(.05 * p['n'])], uncertainty_10=uncertain[:cap],
                 quality_then_uncertainty_10=combined)
    detail = {}
    for name, removed in masks.items():
        assert len(removed) == len(set(removed)) <= cap
        assert set(removed) <= set(p['audit_ids'])
        detail[name] = dict(removed_ids=sorted(removed), kept_ids=sorted(set(p['audit_ids']) - set(removed)),
            removed_n=len(removed), kept_n=p['n'] - len(removed), coverage=(p['n'] - len(removed)) / p['n'])
    dump(OUT / 'masks.json', detail)
    paths = [OUT / n for n in ('plan.json', 'uncertainty_ranking.json', 'review_packet_no_predictions.json', 'adjudication.json', 'masks.json')]
    paths += [ROOT / 'research/round21_audit_quality.py', Path(__file__)]
    frozen = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        frozen_before_subgroup_score_computation=True, original_full_audit_already_seen=True,
        reviewed=len(rows), full_detail_reviewed=len(SPECIAL),
        categories={k: sum(r['decision'] == k for r in rows) for k in ('defect', 'ambiguous', 'uncertain_keep', 'keep')},
        hashes={str(f.relative_to(ROOT)): sha(f) for f in paths}, source_hashes=p['input_hashes'])
    dump(OUT / 'frozen.json', frozen)
    print({k: v for k, v in frozen.items() if k not in ('hashes', 'source_hashes')})
    print({k: (v['removed_n'], v['kept_n']) for k, v in detail.items()})


if __name__ == '__main__':
    from pathlib import Path
    main()
