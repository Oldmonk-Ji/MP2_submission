"""Record the assistant's explicit FIT-only semantic review, with provenance."""
from round20_common import ROOT,OUT,plan,read,dump,sha
import datetime

ISSUES={
217:('spurious_positive_reasoning','唯一数值正确候选把累计存款说成7枚硬币，直接输出7；没有累计1至7岁，也没有把硬币换成美元。官方数字7正确，候选推理不支持它。'),
3421:('spurious_positive_reasoning','唯一数值正确候选称周末有5天，用24/10=2.4后声称向上取整为2；终值2碰巧匹配，推理有明确错误。'),
3879:('positive_stage_contradiction','唯一数值正确候选先把V4定义为第4天总量40000，最后又加最初4000；总量/增量绑定在同一候选中互相冲突。题干ten times more本身也不严谨。'),
1341:('reference_question_conflict','题干是6个四年级班，参考解却按4个班计算2000；若额外假定每周匀速，6班应为6*(280/2*3)+320=2840。第三周产量本身未明示，不能擅自改标签。'),
2102:('spurious_positive_reasoning','唯一数值正确候选把124-58+24写成100而非90，随后漏加10名机组；两处错误抵消而得到官方终值67。'),
3696:('spurious_positive_reasoning','三个数值正确候选均写7+12+15-2=30，左式实际为32；三人共同看的2部各重复两次，应减4。终值正确但算术和去重逻辑均错。'),
3963:('quantity_scope_ambiguity','题干2 copies for 9 people没有明确per person，唯一正例又把2 contracts*9 people写成18 people，混淆份数与人数。作为作用范围不明确且正例单位有问题的题暂剔除，未宣称官方数值标签错误。'),
2259:('spurious_positive_binding','三个正例均把Alice的3600颗橡果转给Bob，用未给出的Bob数量求单价，再乘9；正确路径应先求Alice总价9*6000再除Alice的3600。终值15相同但人物-数量绑定错误。'),
362:('reference_unit_scope_conflict','题干先给20双红袜，却问袜子总只数；参考解把双数20当只数。按一致只数为40+20+120=180；官方90对应双数而非明确所问只数。保留原标签，实验中剔除此题。'),
4544:('category_overlap_ambiguity','non-fiction与autobiographies在通常含义上重叠；给出的9本非虚构却有10本自传。参考解将类别互斥处理，题干没有明确说明这是书店分区。'),
668:('missing_attribute_constraints','问既不格纹也不紫色，但只给格纹衬衣与紫色裤子数量，未给紫色衬衣或格纹裤子信息。参考解隐含另外两类为零。'),
487:('set_overlap_ambiguity','虫蛀与碰伤的苹果是否重叠未给；参考解直接扣17和26，隐含两类互斥。其余苹果数量严格说依赖重叠数。'),
256:('entity_wording_defect','题干在fruits/plants/seeds/trees之间切换，称每棵树有20株植物，种下种子后直接问种了多少树。24对应可推的种子数量，实体和生长阶段表述有缺陷。'),
}

# These FIT questions and all numeric-positive candidate texts were inspected.
# Correct final labels with a sound candidate are protected from OOF rejection.
SOUND={
2502:'(10+8+20)*20=760，20小时是尚需工作量；正例完整正确。',
947:'2*(2+3*5)=34，正例完整计入两栋楼。',
3131:'18+7*(3-5)=4，正例保留数量减少方向。',
4663:'70-21-(21-6)=34，正例完整推导。',
3598:'420-84-21-5=310，正例算术正确。',
969:'1+2+2+2+3=10，正例覆盖三批消费。',
2555:'原堆40，每人拿10，剩20；正例最后省略减法，但已足以确定剩量，没有必要仅因三模型错选而删除。',
3706:'全部交付的是fabric，不是yarn；(20+40+10)*2=140，两正例均正确。',
2341:'4周*5工作日*4蛋糕*8=640，正例遵守工作日。',
621:'(3*3+2*6)*4=84；正例首句有Tuesday赘述，但逐日表和总数一致，保留。',
3984:'18/2+21+56=86，正例正确处理每两勺。',
4538:'题干明确including the down payment，600*60+10000=46000；保留。',
584:'shorts/pants的pair通常指一件衣物；28+30=58，正例正确。',
908:'12.5*40*(1-.2)*(1-.15)=340，正例正确绑定税后基数。',
3540:'(20+40+8+3)*5=355，正例同时处理单套价与5套。',
1180:'3*4*20=240，仅问幼鱼，不计母鱼；两正例正确。',
70:'J=30，M+5=2*(J+5)，M=65；正例时间阶段正确。',
490:'200+200+500=900，短正例正确计入折返。',
3286:'本队84，对方117，合计201；参考解文字6有笔误但计算用8，正例完整正确。',
1189:'先装满3盆9粒，第四盆1粒；正例符合题目通常分装约定。',
1950:'按题干到12日包含当日读书的约定，8*12/(2/3)=144，正例正确。',
453:'加权均价(8*12+4*18)/12=14，三个正例均正确。',
58:'1000+1500+1500=4000，至少第一个正例关系完整正确。',
2069:'155250/1000*100=15525，正例正确。',
3133:'8*3+10*6+30=114，正例正确区分每只与鸡群共享水箱。',
1018:'(30000-10000)*(1-.4)=12000，正例资助基数正确。',
1678:'(8*3+6)/2+5=20，两正例正确。',
3111:'正例有显式自我纠错，最终用(94-54)/5+54/6=17；不把已修正中间错误当坏正例。',
427:'按剩余可支配现金约定，(25-8)/2-5*.5=6，两正例正确。',
1414:'已跑17英里，剩3英里，按6英里/小时需30分钟；正例正确，参考解6/3笔误未用于训练。',
882:'120/3/8=5，正例正确。',
1623:'(14+10)*3/4*4=72，正例正确。',
2430:'5*3*4*8=480，正例包装层级正确。',
1718:'2*(3+9)+3*11=57，正例正确绑定颜色与人物。',
3591:'2*50*(1-.4)=60，正例正确。',
1806:'按题目整年年龄约定2021-(37+3+2)=1979，正例正确；不额外引入未提供的生日精度。',
4079:'3/(6+12+3)*100按题目取整14，正例正确。',
446:'10*1.5*1.6=24，速度虽不现实但数学关系明确。',
3803:'28/2*10*2=280，至少第二个正例清楚且完整。',
1105:'30/(2*2+1)=6，正例容量与所求房间数正确。',
4093:'明确已有12根，新增72根，减已找到24根剩48；两个正例正确。',
1321:'(25+10)*4*3=420，正例正确。',
54:'至少第二个正例正确描述两批人下水，62-8=54；不因另一正例叙述错误而删整题。',
1415:'题干只提供一个小时工资，(8+2)*5*13.5=675；不擅自引入额外加班倍率。',
}
UNCERTAIN={
4521:'唯一正例把63%称为unsold但又正确计算保留销售收入；是文字极性错误还是可忽略笔误无法确定。仍只作不确定样本。',
3644:'唯一正例写two of the fish spat out而不是两块石头，但10-5+2正确；保留不确定性，不直接宣布数值标签错误。',
4268:'唯一正例把every other day combined解释为第七天，随后计算却用前六天的600；文字与方程冲突程度待复核。',
}


def main():
    root=OUT/'quality';assert not (root/'ranking.json').exists()
    packet=read(root/'uncertainty_review_packet.json');fit=set(plan()['partitions']['fit'])
    assert set(r['train_index'] for r in packet)==set(ISSUES)|set(SOUND)|set(UNCERTAIN)
    out=[]
    for r in packet:
        i=r['train_index']
        if i in ISSUES:kind,why=ISSUES[i];decision='exclude'
        elif i in SOUND:kind='reviewed_useful_hard_question';why=SOUND[i];decision='keep'
        else:kind='unresolved_positive_trace_wording';why=UNCERTAIN[i];decision='uncertain'
        out.append(dict(train_index=i,decision=decision,category=kind,reason_zh=why,
            source='assistant semantic review of top-60 FIT OOF-risk questions and all their numeric-positive candidate texts',
            human_gold=False,official_label_unchanged=True))
    for n in (1,2):
        source=ROOT/f'experiments/human_review/review_pack0{n}_adjudication.json'
        for r in read(source)['rows']:
            i=r['train_index']
            if i not in fit or i in {v['train_index'] for v in out}:continue
            decision='keep' if r['comparison']=='agreement' else 'exclude'
            out.append(dict(train_index=i,decision=decision,category='prior_human_ambiguity_review' if decision=='exclude' else 'prior_human_agreement',
                reason_zh=r.get('adjudication_reason',r.get('adjudication','')),
                source=str(source.relative_to(ROOT)),source_sha=sha(source),human_response_preserved=True,
                exclusion_is_experimental_not_label_correction=True,official_label_unchanged=True))
    dump(root/'adjudication.json',out)
    dump(OUT/'amendment_01.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reason='OOF flags include demonstrably useful hard cases; protect reviewed sound questions from uncertainty-only deletion.',
        change='A keep decision overrides uncertainty eligibility. Explicitly evidenced defects/ambiguities rank first; unresolved cases remain uncertainty-only.',
        timing='Before any filtered-arm fitting and before development comparisons are examined; full-data control may be training concurrently.',
        source='Top-60 FIT OOF-risk packet and existing human reviews intersected with current FIT only.',
        inspected=len(out),excluded=sum(r['decision']=='exclude' for r in out),protected=sum(r['decision']=='keep' for r in out),uncertain=sum(r['decision']=='uncertain' for r in out),
        limitation='Assistant review is not independent human gold; positive trace defects do not invalidate official final-number labels.'))
    print({k:sum(r['decision']==k for r in out) for k in ('exclude','keep','uncertain')})


if __name__=='__main__':main()
