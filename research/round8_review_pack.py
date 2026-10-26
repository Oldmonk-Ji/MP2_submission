"""Publish only open training uncertainties, never audit examples."""
from data_access import ROOT,CODE,read_jsonl
import json,hashlib
from html import escape
OUT=ROOT/'experiments/round8';APP=ROOT/'review_app';BASE=ROOT/'experiments/human_review'
SPECS={
2549:dict(title='这里的收入要不要扣除支出？',category='所求量与费用范围',translation='Jesiah先生的奶牛每天产200加仑奶。牧场维护与饲料支出每月3000美元，每加仑奶卖3.55美元。题目要求计算他在六月的“total income”。',prompt='只判断所求量：这里的“total income”应该包括哪些计算？',options=[dict(id='A',text='计算六月卖奶收到的总金额，不扣除维护和饲料支出。'),dict(id='B',text='计算六月卖奶收到的金额，再扣除维护和饲料支出。')]),
4310:dict(title='开头的一箱玩具是否需要另算？',category='批次归属与重复计数',translation='Joel先收集了一箱旧玩具准备捐赠。随后从朋友处收集18只毛绒玩具、42个人偶、2套棋盘游戏和13个拼图。他的姐姐也给了他一些旧玩具，Joel又从自己的衣柜拿出姐姐所捐数量两倍的玩具。总共捐了108件，题目问其中多少件原本属于Joel。',prompt='开头“一箱旧玩具”和后来“从自己衣柜拿出的玩具”，应当怎样计数？',options=[dict(id='A',text='视为同一批个人玩具的概述与后续说明，不把开头的一箱再算一次。'),dict(id='B',text='视为不同批次：开头的一箱应在后来从衣柜拿出的玩具之外另外计入。')])}

def main():
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');plan=json.loads((OUT/'plan.json').read_text());train={i for g in plan['groups'].values() for i in g['train']}
    assert set(SPECS)<=train
    questions=[]
    for i,s in SPECS.items():
        q=refs[i]['question'];questions.append(dict(id=f'train-{i}-interpretation-v1',train_index=i,question_sha256=hashlib.sha256(q.encode()).hexdigest(),original=q,**{k:v for k,v in s.items() if k!='options'},options=s['options']+[dict(id='U',text='题意不足以唯一判断，或我不确定。'),dict(id='O',text='其他解释（可在下方补充）。')]))
    pack=dict(schema='mp2-review-pack/v1',id='mp2-open-train-review-02',version=1,split='train',title='关系复核 · 第二批（2题）',summary=dict(screened_development=24,assistant_clear=22,previously_reviewed=0,pending=2),questions=questions)
    template=(ROOT/'research/review_app_template.html').read_text(encoding='utf-8')
    sections=[]
    for n,q in enumerate(questions,1):
        sections.append('<section class="panel" style="margin:16px 0"><h2>'+escape(f"{n}. {q['title']}")+'</h2><p>'+escape(q['original'])+'</p><p>'+escape(q['translation'])+'</p><strong>'+escape(q['prompt'])+'</strong>'+''.join('<p>'+escape(o['id']+'．'+o['text'])+'</p>' for o in q['options'])+'</section>')
    fallback='<div id="static-fallback" style="max-width:1000px;margin:24px auto;padding:20px"><h2>第二批：完整题目清单（2题）</h2><p>若交互未加载，可直接按 1:A 2:B 回复聊天。第一批7题已收录，无需重做。</p>'+''.join(sections)+'</div>'
    html=template.replace('__PACK_JSON__',json.dumps(pack,ensure_ascii=False).replace('<','\\u003c')).replace('<div class="workspace">',fallback+'<div class="workspace">').replace('validatePack(pack);load();render();',"validatePack(pack);load();render();document.getElementById('static-fallback').hidden=true;")
    html=html.replace('关系复核工作台</h1>','关系复核工作台 · 第二批</h1>').replace('每题只核对一个有争议的关系。','第一批7题已收录，本批新增2题。每题只核对一个有争议的关系。')
    archive=APP/'复核工作台_第一批.html'
    if not archive.exists():archive.write_bytes((APP/'复核工作台.html').read_bytes())
    oldpack=APP/'questions_第一批.json'
    if not oldpack.exists():oldpack.write_bytes((APP/'questions.json').read_bytes())
    (APP/'复核工作台.html').write_text(html,encoding='utf-8');(APP/'questions.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=dict(pack_id=pack['id'],version=1,items={q['id']:dict(train_index=q['train_index'],question_sha256=q['question_sha256'],option_ids=[o['id'] for o in q['options']]) for q in questions})
    (BASE/'review_pack02_manifest.json').write_text(json.dumps(manifest,indent=2));print('Published 2 open-train questions; batch01 preserved.')
if __name__=='__main__':main()
