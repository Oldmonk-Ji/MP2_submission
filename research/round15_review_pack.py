"""Blind review: open natural ambiguities plus clearly marked derived examples."""
from data_access import ROOT,CODE,read_jsonl
import json,hashlib,random
from html import escape
OUT=ROOT/'experiments/round15';APP=ROOT/'review_app';HUMAN=ROOT/'experiments/human_review'
SPECS={
4124:dict(title='同一题中 he 和 she 的指代',category='真实训练题 · 人物指代',translation='Viggo买了一件80美元的衬衫，用20美元和10美元纸币付款。题目前面称Viggo为“he”，中间说“if she gave the clerk…”，最后又问“did he give”。',prompt='中间的“she”应怎样处理？',options=[dict(id='A',text='视为指代Viggo，代词性别变化不引入新人物。'),dict(id='B',text='可能指另一位未明确介绍的人，不能确定付款关系。')]),
1627:dict(title='25美元是一盆还是一批？',category='真实训练题 · 单价作用范围',translation='供应商卖了20株兰花，每株50美元；又卖了15盆铜钱草，原文接着写“for $25”，这一处没有再次写each。之后支付两名工人每人40美元，并花150美元买新花盆。',prompt='“15 pieces potted Chinese money plant for $25”里的25美元，应作用于什么范围？',options=[dict(id='A',text='15盆铜钱草合计25美元。'),dict(id='B',text='每盆铜钱草25美元。')])}


def main():
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl');plan=json.loads((OUT/'plan.json').read_text());questions=[];private={}
    for i,s in SPECS.items():
        assert plan['groups'][str(i)]['split']=='fit';q=bank[i]['question']
        questions.append(dict(id=f'train-{i}-interpretation-v1',train_index=i,question_sha256=hashlib.sha256(q.encode()).hexdigest(),original=q,**s))
    gen=[json.loads(s) for s in (OUT/'derived_training_quarantine.jsonl').read_text(encoding='utf-8').splitlines()]
    select=[r for r in gen if r['id'].endswith('-00') or (r['family']!='person_binding' and r['id'].endswith('-01'))]
    for n,r in enumerate(select):
        order=list(range(len(r['review']['options'])));random.Random(15150+n).shuffle(order)
        opts=[dict(id=chr(65+j),text=r['review']['options'][k]) for j,k in enumerate(order)]
        qid='derived-'+r['id'];old=ord(r['review']['program_implied_choice'])-65
        private[qid]=dict(derived_id=r['id'],program_implied_choice=chr(65+order.index(old)),annotation_status='pending human review, not auto-gold')
        title={'person_binding':'朋友与姓名的对应','price_batch':'同样的价格，作用范围是否变化？','time_target':'所求量是未来总数还是新增量？'}[r['family']]
        questions.append(dict(id=qid,train_index=r['parent_train_index'],question_sha256=hashlib.sha256(r['question'].encode()).hexdigest(),original=r['question'],translation=r['translation'],title=title,category='训练派生题 · '+{'person_binding':'人物绑定','price_batch':'价格批次','time_target':'时间与目标'}[r['family']],prompt=r['review']['prompt'],options=opts))
    for q in questions:q['options'] += [dict(id='U',text='题意不足以唯一判断，或我不确定。'),dict(id='O',text='其他解释（可在下方补充）。')]
    pack=dict(schema='mp2-review-pack/v1',id='mp2-open-train-review-04',version=1,split='train',title='关系复核 · 第四批（7题）',summary=dict(screened_development=15,assistant_clear=13,pending=7,real_uncertainties=2,derived_review=5),questions=questions)
    template=(ROOT/'research/review_app_template.html').read_text(encoding='utf-8')
    blocks=['<section class="panel"><h2>'+escape(str(i+1)+'. '+q['title'])+'</h2><p>'+escape(q['category'])+'</p><p>'+escape(q['original'])+'</p><p>'+escape(q['translation'])+'</p><strong>'+escape(q['prompt'])+'</strong>'+''.join('<p>'+escape(o['id']+'．'+o['text'])+'</p>' for o in q['options'])+'</section>' for i,q in enumerate(questions)]
    fallback='<div id="static-fallback" style="max-width:1000px;margin:24px auto;padding:20px"><h2>第四批：2道原题歧义＋5道训练派生题</h2>'+''.join(blocks)+'</div>'
    html=template.replace('__PACK_JSON__',json.dumps(pack,ensure_ascii=False).replace('<','\\u003c')).replace('<div class="workspace">',fallback+'<div class="workspace">').replace('validatePack(pack);load();render();',"validatePack(pack);load();render();document.getElementById('static-fallback').hidden=true;")
    html=html.replace('关系复核工作台</h1>','关系复核工作台 · 第四批</h1>').replace('每题只核对一个有争议的关系。','本批含2道真实训练题歧义和5道训练派生题，只核对文字表达。')
    html=html.replace('训练题 #${q.train_index} · 不计分复核',"${q.id.startsWith('derived-')?'训练派生题 · 来源训练题':'真实训练题'} #${q.train_index} · 不计分复核")
    html=html.replace('<div class="notice">','<p><a href="复核工作台_第三批.html">继续第三批（原进度保留）</a></p><div class="notice">')
    for current,archive in [('复核工作台.html','复核工作台_第三批.html'),('questions.json','questions_第三批.json')]:
        if not (APP/archive).exists():(APP/archive).write_bytes((APP/current).read_bytes())
    (APP/'复核工作台.html').write_text(html,encoding='utf-8');(APP/'questions.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=dict(pack_id=pack['id'],version=1,items={q['id']:dict(train_index=q['train_index'],question_sha256=q['question_sha256'],option_ids=[o['id'] for o in q['options']],source_type='derived_train' if q['id'].startswith('derived-') else 'official_train') for q in questions})
    (HUMAN/'review_pack04_manifest.json').write_text(json.dumps(manifest,indent=2));(OUT/'derived_review_private.json').write_text(json.dumps(private,indent=2))
    (APP/'使用说明.txt').write_text('第四批7题：前2题为真实训练题中的歧义，后5题为明确标记的训练派生题。\n第三批未完成可点击页面顶部归档入口继续，原浏览器进度保留。\n作答后导出JSON发回聊天。不确定可选U，其他解释可选O。没有参考答案、模型预测或预选项。\n生成题不参与真实候选评估，也尚未进入正式训练。\n',encoding='utf-8')
    print('Published 7 questions; previous pack and progress namespace preserved.')


if __name__=='__main__':main()
