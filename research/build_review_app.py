"""Build an answer-free offline review pack from OPEN TRAIN development only."""
from data_access import ROOT,CODE,read_jsonl,parse_answer
import json,hashlib
from html import escape
from fractions import Fraction

BASE=ROOT/'experiments/human_review'
APP=ROOT/'review_app'
SPECS={
1203:dict(title='“三倍年轻”怎样理解？',category='倍数关系',zh='Mason比Sydney年轻，原文用了“three times younger than”。Sydney又比Mason的父亲年轻六岁。Mason现在20岁，题目问父亲的年龄。',prompt='只判断第一句：Mason与Sydney的年龄应当是什么关系？',options=[('A','Mason的年龄是Sydney的三分之一。'),('B','Sydney比Mason大三个Mason的年龄，即Sydney年龄是Mason的四倍。')]),
1922:dict(title='已有的4本，要不要再加去年的6本？',category='总量与新增量',zh='Sara的姐姐衣柜里有4本小笔记本。去年夏天，她又订购了6本，随后丢了2本。题目问她现在有多少本。',prompt='开头“has 4 small notebooks”中的4本，应当放在什么时间点？',options=[('A','订购6本、丢失2本之前的起始数量，需要继续计入这两次变化。'),('B','她现在已经拥有的数量，去年的订购和丢失不应再次计入。')]),
2344:dict(title='两台相机的110美元属于哪种价格？',category='单价作用范围',zh='Selene购买两台即时成像相机，原文价格表述是“two instant cameras at $110”；还购买三个数码相框，每个120美元。全部商品享受5%折扣，题目问总付款。',prompt='只判断相机的110美元：它是每台价格，还是两台合计价格？',options=[('A','两台相机合计110美元。'),('B','每台相机110美元。')]),
3711:dict(title='后两周的“每周两倍”，以什么为基准？',category='时间与数量范围',zh='Sofia和母亲买了10公斤橙子；父亲又让她们多买5公斤，因为邻居也需要。之后她们估计，在接下来的两周，每周要买“twice as many oranges”。题目问三周总共购买多少公斤。',prompt='你怎样理解后两周每周的购买量？',options=[('A','后两周每周都是第一周实际购入量（10＋5公斤）的两倍。'),('B','后两周每周都是最初提到的10公斤的两倍。'),('C','每周都比上一周翻倍：第二周按第一周的两倍，第三周再按第二周的两倍。')]),
1201:dict(title='留作一周午餐，究竟留几根？',category='隐含条件',zh='Greg买了一包20根谷物棒，准备在一周的午餐时吃。他给这周的每一天各留了一根，用剩余中的3根向朋友Pete换了一瓶汽水，再把其余的平均分给两个妹妹。',prompt='原文“one for each day of the week”应该理解为预留几根？',options=[('A','7根，按一周七天计算。'),('B','5根，按工作日或上学日的五次午餐计算。')]),
3790:dict(title='“three times more”是三倍还是四倍？',category='倍数关系',zh='屋里有Puffy、Scruffy、Buffy和Juniper四只猫。原文说Puffy有“three times more whiskers than Juniper”，但只有Scruffy的一半。Buffy的胡须数是另外三只猫的平均数，Juniper有12根胡须。',prompt='只判断Puffy与Juniper的关系，不用计算Buffy的答案。',options=[('A','Puffy的胡须总数是Juniper的三倍。'),('B','Puffy比Juniper多三倍，胡须总数是Juniper的四倍。')]),
1585:dict(title='造型费是否包含基础修剪？',category='价格批次',zh='Neil每修剪一株黄杨收5美元；把它修成特殊形状收费15美元。顾客要修剪整排30株黄杨，另外要求其中4株修成球形。',prompt='你怎样理解4株球形黄杨的收费范围？',options=[('A','30株都收基础修剪费；其中4株再额外各收15美元造型费。'),('B','4株的每株15美元已包含基础修剪；另外26株按每株5美元收费。')]),
}
CLEAR={1967:('2000','(90-40)*2*20'),370:('5','18-26/2'),3215:('10','(8-3)*2'),4512:('51','(10-18/12)/2*12'),867:('22','6*2+10'),930:('19','(10-3)+(10+2)'),2683:('6','25-(25-2-4)'),4276:('12','60*(1-0.4)/3'),95:('25','50-(3*1+4*3+2*5)'),1790:('10','(340-40)/(6*5)'),443:('30','5*(2*2+0.5+3*0.5)'),4212:('14','(26+18+(26+18)/2)*7/11/3'),2266:('210','10*8+5*10+20*4'),2367:('4','(124-10*8)/11'),4046:('97','450*0.2+30*0.1+80*0.05'),4142:('2','(14+10)*0.5/6'),140:('200','(4*0.05+2.5-7*0.1)*100'),4553:('19','(77-2*6)/5+6'),2592:('32','12+(12+8)'),1378:('75','(1000-100)/12'),1895:('3','3*3*(1-2/3)'),1920:('828','4*1863/9'),4023:('6','27*(1-1/3)-8-8/2'),1095:('11','60-30-4*4-3'),2285:('113','3*48-24-7'),4562:('84','8*1.5*7'),3861:('48','4*6*2'),3545:('20','(500-200-150)/7.5'),2930:('35','(28*2+28*0.75+28)/3'),412:('21','7+7*2'),4021:('10','80-50-20'),4412:('16','(6+2)*2'),822:('482','12*12*3+50'),3874:('63','(1.5*10+2*3)*3'),2807:('1588','221*3+185*5'),4458:('66','50*0.9+5.5*2+10'),1343:('480','30*2*8'),4117:('6400','40*4*50*0.8'),146:('150','(200+100)/2'),1197:('1035','(40+50)*1.15*10'),2062:('63','120*0.5*1.05'),967:('27','32-5'),1693:('2220','(8+4)*80+(7+4/2)*140'),610:('11','(200-5*5-10*5-3*5)/10')}
def main():
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');plan=json.loads((ROOT/'experiments/round7/plan.json').read_text());allowed=set(plan['existing_train_questions'])|{i for g in plan['groups'].values() for i in g['new_train']};sealed={i for g in plan['groups'].values() for i in g['sealed_audit']}
    human=json.loads((BASE/'batch01.json').read_text(encoding='utf-8'));reviewed={r['train_index'] for r in human['rows']}
    assert allowed==set(CLEAR)|set(SPECS)|reviewed
    assert not set(SPECS)&sealed
    dispositions=[]
    for i,(answer,formula) in CLEAR.items():
        assert Fraction(parse_answer(refs[i]['answer']))==Fraction(answer),i
        dispositions.append(dict(train_index=i,status='assistant_assessed_clear',independent_answer=answer,calculation=formula,reference_agrees=True,note='Assistant interpretation and independent calculation agree with train reference. This is not an independent human annotation or proof that every semantic binding is correct.'))
    dispositions += [dict(train_index=i,status='human_reviewed',record='batch01.json') for i in sorted(reviewed)]
    dispositions += [dict(train_index=i,status='needs_human_interpretation') for i in SPECS]
    summary=dict(training_total=4755,screened_development=56,assistant_clear=44,previously_reviewed=5,pending=len(SPECS),outside_this_screen=4755-56,scope='Only the 56 open round7 development questions were screened. Other train questions are unassessed, not marked clear. No validation/test or round7 audit questions in this pack.')
    (BASE/'triage01_private.json').write_text(json.dumps(dict(summary=summary,dispositions=dispositions),indent=2,ensure_ascii=False),encoding='utf-8')
    questions=[]
    for i,s in SPECS.items():
        q=refs[i]['question'];questions.append(dict(id=f'train-{i}-interpretation-v1',train_index=i,question_sha256=hashlib.sha256(q.encode()).hexdigest(),original=q,translation=s['zh'],title=s['title'],category=s['category'],prompt=s['prompt'],options=[dict(id=k,text=v) for k,v in s['options']]+[dict(id='U',text='题意不足以唯一判断，或我不确定。'),dict(id='O',text='其他解释（可在下方补充）。')]))
    pack=dict(schema='mp2-review-pack/v1',id='mp2-open-train-review-01',version=1,split='train',title='关系复核 · 第一份待确认清单',summary=summary,questions=questions)
    APP.mkdir(exist_ok=True);(APP/'questions.json').write_text(json.dumps(pack,indent=2,ensure_ascii=False),encoding='utf-8')
    template=(ROOT/'research/review_app_template.html').read_text(encoding='utf-8');embedded=json.dumps(pack,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    sections=[]
    for n,q in enumerate(questions,1):
        opts=''.join('<p>'+escape(o['id']+'．'+o['text'])+'</p>' for o in q['options'])
        sections.append('<section class="panel" style="margin:16px 0"><h2>'+escape(f"{n}. {q['title']}")+'</h2><p>'+escape(q['original'])+'</p><p>'+escape(q['translation'])+'</p><strong>'+escape(q['prompt'])+'</strong>'+opts+'</section>')
    fallback='<div id="static-fallback" style="max-width:1000px;margin:24px auto;padding:20px"><h2>完整题目清单（7题）</h2><p>如果只看到这份清单，当前预览未能运行交互程序。仍可直接阅读，按“1:A 2:B …”回复聊天。要自动保存和切题，请用浏览器打开本文件。</p>'+''.join(sections)+'</div>'
    html=template.replace('__PACK_JSON__',embedded).replace('<div class="workspace">',fallback+'<div class="workspace">')
    html=html.replace('validatePack(pack);load();render();',"validatePack(pack);load();render();document.getElementById('static-fallback').hidden=true;")
    (APP/'复核工作台.html').write_text(html,encoding='utf-8')
    (BASE/'review_pack01_manifest.json').write_text(json.dumps(dict(pack_id=pack['id'],version=1,items={q['id']:dict(train_index=q['train_index'],question_sha256=q['question_sha256'],option_ids=[o['id'] for o in q['options']]) for q in questions}),indent=2))
    print(json.dumps(summary))
if __name__=='__main__':main()
