from data_access import ROOT,CODE,read_jsonl
import json,hashlib
from html import escape

def main():
    app=ROOT/'review_app';base=ROOT/'experiments/human_review'
    q=read_jsonl(CODE/'data/gsm8k_train.jsonl')[3077]['question']
    item=dict(id='train-3077-deadline-v1',train_index=3077,question_sha256=hashlib.sha256(q.encode()).hexdigest(),original=q,
        title='优惠截止日是否包含付款当天？',category='事件时间与优惠条件',
        translation='一家网络服务商规定：顾客在每月25日之前付款，可以享受5%的折扣。每月网络费用为50美元。某顾客连续4个月都在每月25日付款，题目问他总共支付了多少钱。',
        prompt='只根据原文，判断“before the 25th”与“paid every 25th”之间的关系。',
        options=[dict(id='A',text='付款日期满足折扣条件，4个月都应享受优惠。'),dict(id='B',text='付款日期不满足折扣条件，4个月都按原价支付。'),dict(id='U',text='题意不足以唯一判断，或我不确定。'),dict(id='O',text='其他解释（可在下方补充）。')])
    pack=dict(schema='mp2-review-pack/v1',id='mp2-open-train-review-03',version=1,split='train',title='事件关系复核 · 第三批（1题）',summary=dict(screened_development=20,assistant_clear=19,previously_reviewed=2,pending=1),questions=[item])
    template=(ROOT/'research/review_app_template.html').read_text(encoding='utf-8')
    fallback='<div id="static-fallback" style="max-width:1000px;margin:24px auto;padding:20px"><h2>第三批完整题目（1题）</h2><p>如交互未加载，可在聊天直接回复 1:A/B/U/O。</p><p>'+escape(q)+'</p><p>'+escape(item['translation'])+'</p><p>'+escape(item['prompt'])+'</p>'+''.join('<p>'+escape(o['id']+'．'+o['text'])+'</p>' for o in item['options'])+'</div>'
    html=template.replace('__PACK_JSON__',json.dumps(pack,ensure_ascii=False).replace('<','\\u003c')).replace('<div class="workspace">',fallback+'<div class="workspace">').replace('validatePack(pack);load();render();',"validatePack(pack);load();render();document.getElementById('static-fallback').hidden=true;")
    html=html.replace('关系复核工作台</h1>','事件关系复核工作台 · 第三批</h1>')
    current=json.loads((app/'questions.json').read_text(encoding='utf-8'))
    if current['id']=='mp2-open-train-review-02':
        for source,dest in [('questions.json','questions_第二批.json'),('复核工作台.html','复核工作台_第二批.html')]:
            if not (app/dest).exists():(app/dest).write_bytes((app/source).read_bytes())
    (app/'questions.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2),encoding='utf-8')
    (app/'复核工作台.html').write_text(html,encoding='utf-8')
    manifest=dict(pack_id=pack['id'],version=1,items={item['id']:dict(train_index=3077,question_sha256=item['question_sha256'],option_ids=[o['id'] for o in item['options']])})
    (base/'review_pack03_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    assert '__PACK_JSON__' not in html and 'static-fallback' in html
    print('Published pack03: 1 question; previous packs preserved.')
if __name__=='__main__':main()
