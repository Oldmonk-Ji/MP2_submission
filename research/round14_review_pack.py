"""Publish two open-TRAIN ambiguities; no audit questions or predictions in UI."""
from data_access import ROOT, CODE, read_jsonl
import json
import hashlib
from html import escape

OUT=ROOT/'experiments/round14';APP=ROOT/'review_app';HUMAN=ROOT/'experiments/human_review'
SPECS={
3073:dict(title='平均年龄对应哪个时间？',category='当前与未来的目标绑定',translation='八年后，Bernard先生的年龄将是Luke现在年龄的3倍。Luke现在20岁。问题接着问：比“他们的平均年龄”少10岁是多少？原文最后没有再写明平均年龄的时间。',prompt='这里“their average age”应对应哪一组年龄？',options=[dict(id='A',text='Bernard和Luke现在的年龄。'),dict(id='B',text='Bernard和Luke八年后的年龄。'),dict(id='C',text='Bernard八年后的年龄与Luke现在的年龄。')]),
3471:dict(title='“今年将满12岁”如何确定起点？',category='生日与时间基点',translation='Bush和Matt是兄弟，Bush比Matt小3岁。原文说Bush“今年将满12岁”，接着问Matt从现在起10年后的年龄。没有提供今天的日期或Bush的生日。',prompt='仅根据题目，怎样理解计算年龄的时间起点？',options=[dict(id='A',text='把今年达到12岁的年龄作为“现在”的计算基点。'),dict(id='B',text='起点应是发问当天；还需知道今年生日是否已过，才能确定确切年龄。')])}


def main():
    rows=read_jsonl(CODE/'data/gsm8k_train.jsonl');plan=json.loads((OUT/'plan.json').read_text())
    questions=[]
    for i,spec in SPECS.items():
        assert plan['groups'][str(i)]['split'] in ('fit','development')
        q=rows[i]['question']
        questions.append(dict(id=f'train-{i}-interpretation-v1',train_index=i,question_sha256=hashlib.sha256(q.encode()).hexdigest(),original=q,
                              **{k:v for k,v in spec.items() if k!='options'},
                              options=spec['options']+[dict(id='U',text='题意不足以唯一判断，或我不确定。'),dict(id='O',text='其他解释（可在下方补充）。')]))
    pack=dict(schema='mp2-review-pack/v1',id='mp2-open-train-review-03',version=1,split='train',title='关系复核 · 第三批（2题）',
              summary=dict(screened_development=15,assistant_clear=6,previously_reviewed=1,ambiguity_masked=6,pending=2),questions=questions)
    template=(ROOT/'research/review_app_template.html').read_text(encoding='utf-8')
    sections=[]
    for n,q in enumerate(questions,1):
        sections.append('<section class="panel" style="margin:16px 0"><h2>'+escape(f"{n}. {q['title']}")+'</h2><p>'+escape(q['original'])+'</p><p>'+escape(q['translation'])+'</p><strong>'+escape(q['prompt'])+'</strong>'+''.join('<p>'+escape(o['id']+'．'+o['text'])+'</p>' for o in q['options'])+'</section>')
    fallback='<div id="static-fallback" style="max-width:1000px;margin:24px auto;padding:20px"><h2>第三批：完整题目清单（2题）</h2><p>若交互未加载，可按 1:A 2:B 的格式回复。前两批已收录，无需重做。</p>'+''.join(sections)+'</div>'
    html=template.replace('__PACK_JSON__',json.dumps(pack,ensure_ascii=False).replace('<','\\u003c')).replace('<div class="workspace">',fallback+'<div class="workspace">').replace('validatePack(pack);load();render();',"validatePack(pack);load();render();document.getElementById('static-fallback').hidden=true;")
    html=html.replace('关系复核工作台</h1>','关系复核工作台 · 第三批</h1>').replace('每题只核对一个有争议的关系。','前两批已收录，本批新增2题。每题只核对一个有争议的关系。')
    for old,new in [('复核工作台.html','复核工作台_第二批.html'),('questions.json','questions_第二批.json')]:
        dest=APP/new
        if not dest.exists():dest.write_bytes((APP/old).read_bytes())
    (APP/'复核工作台.html').write_text(html,encoding='utf-8')
    (APP/'questions.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=dict(pack_id=pack['id'],version=1,items={q['id']:dict(train_index=q['train_index'],question_sha256=q['question_sha256'],option_ids=[o['id'] for o in q['options']]) for q in questions})
    (HUMAN/'review_pack03_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    notes=dict(purpose='Research adjudication notes, NOT new teacher labels for fitting.',
               clear_cases={113:'Both stated ages share a past event; gap 12, Marianne 30 when Bella 18.',1240:'The future clauses refer to one common event; Anne = 2*6*8 = 96.',2773:'Brian current 32, Christian current 64, Christian in eight years 72.',2776:'All stated ages and target use the same past frame; Greg 16.',2580:'Barbara current 8; Mike gaining eight years implies Barbara 16.',2890:'Both ratio endpoints five years ahead; Ron 48 then, Maurice 12 then, 7 now.'},
               ambiguity_masked=[30,2222,3246,3346,4440,4290],already_human_reviewed=[1203],pending=list(SPECS),
               note='Keep ambiguous times older/younger relations masked. No reference or model prediction is exposed in the review page; none of these notes enters this frozen model.')
    (OUT/'open_train_triage.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    (APP/'使用说明.txt').write_text('当前为第三批，共2题。前两批页面已归档，答案文件和浏览器保存记录不变。\n打开 http://127.0.0.1:8767/复核工作台.html ，如仍显示第二批，请刷新。\n作答后导出 JSON 发给我。无唯一解释可选 U，补充解释可选 O。\n题目均来自训练侧开放分区，不含本轮审查题、模型预测或标准答案。\n',encoding='utf-8')
    print('Published pack03: 2 open-TRAIN questions. Previous packs preserved; audit untouched by review publication.')


if __name__=='__main__':main()
