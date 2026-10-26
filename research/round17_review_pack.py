"""Publish only FIT uncertainties; preserve pack04 and its progress key."""
from data_access import ROOT,CODE,read_jsonl
from round17_labels import OUT
from html import escape
import json,hashlib
APP=ROOT/'review_app';HUMAN=ROOT/'experiments/human_review'
SPECS=[
dict(train_index=3059,id='train-3059-specialist-scope-v1',title='两位专家的时薪按谁计？',category='单价与服务对象',
     translation='Dakota住院3天。病床每天900美元，医院按“250美元/小时”安排两位专家分别看她15分钟，救护车费用1800美元。',
     prompt='原文“$250/hour for two specialists to look at her 15 minutes each”中的时薪250美元，应怎样理解？',
     options=[dict(id='A',text='每位专家分别按每小时250美元计费。'),dict(id='B',text='两位专家作为一组，合计按每小时250美元计费。')]),
dict(train_index=3886,id='train-3886-profit-base-v1',title='25%的利润以什么为基数？',category='倍数关系的参照量',
     translation='Niko买了9双袜子，每双2美元。他希望其中4双获得25%的利润，另外5双每双赚0.2美元，题目问总利润。',
     prompt='原文“gain 25% profit”中的25%，应以什么金额为基数？',
     options=[dict(id='A',text='这4双袜子的购入成本。'),dict(id='B',text='这4双袜子的卖出收入。'),dict(id='C',text='文字没有明确基数，存在不同解释。')]),
dict(train_index=3886,id='train-3886-candidate-role-v1',title='候选里的“2 each”表示什么？',category='候选数量角色复核',
     translation='原题说Niko买了9双袜子，每双2美元。下面只复核其中一份候选的措辞，不判断整份解答最终是否正确。',
     prompt='候选原句：“Niko has 9 pairs of socks, 2 each for a total cost of $2.”（Niko有9双袜子，2 each，总成本为2美元。）其中第一个没有美元符号的“2 each”表达什么？',
     options=[dict(id='A',text='每双的单价为2美元。'),dict(id='B',text='每双包含2只袜子，表示数量。'),dict(id='C',text='这句话不够清楚，不能唯一确定。')]),
]

def main():
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl');plan=json.loads((OUT/'plan.json').read_text(encoding='utf-8'));qs=[]
    for s in SPECS:
        assert plan['groups'][str(s['train_index'])]['split']=='fit'
        raw=bank[s['train_index']]['question'];q=dict(s,original=raw,question_sha256=hashlib.sha256(raw.encode()).hexdigest())
        q['options']=q['options']+[dict(id='U',text='我不确定，暂不判断。'),dict(id='O',text='其他解释（可在下方补充）。')];qs.append(q)
    pack=dict(schema='mp2-review-pack/v1',id='mp2-open-train-review-05',version=1,split='train',title='关系复核 · 第五批（3题）',
              summary=dict(screened_development=8,pending=3,scope='FIT-only source/claim interpretation; no audit samples or predictions'),questions=qs)
    template=(ROOT/'research/review_app_template.html').read_text(encoding='utf-8')
    fallback='<div id="static-fallback" style="max-width:1000px;margin:24px auto;padding:20px"><h2>第五批：3道训练侧复核题</h2>'+''.join('<section class="panel"><h2>'+escape(str(i+1)+'. '+q['title'])+'</h2><p>'+escape(q['original'])+'</p><p>'+escape(q['translation'])+'</p><p>'+escape(q['prompt'])+'</p>'+''.join('<p>'+escape(o['id']+'．'+o['text'])+'</p>' for o in q['options'])+'</section>' for i,q in enumerate(qs))+'</div>'
    html=template.replace('__PACK_JSON__',json.dumps(pack,ensure_ascii=False).replace('<','\\u003c')).replace('<div class="workspace">',fallback+'<div class="workspace">').replace('validatePack(pack);load();render();',"validatePack(pack);load();render();document.getElementById('static-fallback').hidden=true;")
    html=html.replace('关系复核工作台</h1>','关系复核工作台 · 第五批</h1>').replace('每题只核对一个有争议的关系。','本批只复核训练侧的对象、价格范围和数量角色。')
    html=html.replace('<div class="notice">','<p><a href="复核工作台_第四批.html">继续第四批（原进度保留）</a> · <a href="复核工作台_第三批.html">第三批</a></p><div class="notice">')
    current=json.loads((APP/'questions.json').read_text(encoding='utf-8'))
    assert current['id'] in ('mp2-open-train-review-04',pack['id'])
    for name,archive in [('复核工作台.html','复核工作台_第四批.html'),('questions.json','questions_第四批.json')]:
        if not (APP/archive).exists():
            assert current['id']=='mp2-open-train-review-04';(APP/archive).write_bytes((APP/name).read_bytes())
    (APP/'复核工作台.html').write_text(html,encoding='utf-8');(APP/'questions.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=dict(pack_id=pack['id'],version=1,items={q['id']:dict(train_index=q['train_index'],question_sha256=q['question_sha256'],option_ids=[o['id'] for o in q['options']]) for q in qs})
    (HUMAN/'review_pack05_manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'fit_uncertainties.json').write_text(json.dumps(dict(items=[dict(id=s['id'],train_index=s['train_index']) for s in SPECS],
        handling='No automatic adjudication. Specialist group identity and profit base are unresolved qualifiers; candidate 2-each role is masked. Human responses may affect only a later training version.',
        audit_in_review=False),indent=2))
    (APP/'使用说明.txt').write_text('第五批3题：专家时薪范围、利润基数、候选中的数量角色。均来自第17轮拟合侧，不含封存审查题。\n中文解释与选项不含预测或预设答案。作答后导出JSON发回聊天。\n顶部可继续第四批及第三批，原题包ID与浏览器进度保留。\n',encoding='utf-8')
    print('Published pack05: 3 FIT review items; pack04 archived with its original ID.')

if __name__=='__main__':main()
