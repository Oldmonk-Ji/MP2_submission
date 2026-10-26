"""TRAIN-derived hard-type problems with executable programs and four choices.

Numerical consistency is checked. Linguistic fidelity still requires review;
these examples are quarantined from the frozen natural-candidate experiment.
"""
from data_access import ROOT, CODE, read_jsonl
from fractions import Fraction as F
import json,random,hashlib
OUT=ROOT/'experiments/round15'


def build():
    rows=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl');plan=json.loads((OUT/'plan.json').read_text());rng=random.Random(15015);out=[]
    for typ,parent in [('person_binding',3993),('price_batch',2553),('time_target',1691)]:
        assert plan['groups'][str(parent)]['split']=='fit'
        for j in range(16):
            if typ=='person_binding':
                ga=12+j%4;la=15+j%3;gh=40+j%5;lh=8+j%6
                q=f"Greta earns ${ga} per hour and her friend Lisa earns ${la} per hour. Greta worked {gh} hours. Her friend worked {lh} hours. How much did Lisa earn?"
                zh=f'Greta每小时赚{ga}美元，她的朋友Lisa每小时赚{la}美元。Greta工作了{gh}小时，她的朋友工作了{lh}小时。题目问Lisa赚了多少。'
                truth=F(la*lh);values=[truth,F(ga*lh),F(la*gh),F(ga*gh)]
                descriptions=[f"Lisa is Greta's friend. Lisa worked {lh} hours at ${la} per hour.",f"Lisa worked {lh} hours at Greta's rate of ${ga} per hour.",f"Lisa worked Greta's {gh} hours at ${la} per hour.",f"Lisa earned Greta's {gh} hours times ${ga} per hour."]
                program=dict(owner='Lisa',rate=la,hours=lh,operation='rate * hours',distractor_rate=ga,distractor_hours=gh)
                prompt='第二句之后的“Her friend”指谁？';opts=['Lisa','Greta','原文没有说明的第三人'];label='A'
            elif typ=='price_batch':
                a=5+j%3;b=4+j%4;pa=5+j%5;pb=2+j%3;batch=j%2==0
                detail=f'The {b} cartons of orange juice together cost ${pb}.' if batch else f'Each carton of orange juice costs ${pb}.'
                q=f'Wyatt bought {a} loaves of bread and {b} cartons of orange juice. Each loaf of bread costs ${pa}. {detail} How much did he spend in total?'
                zh=f'Wyatt买了{a}条面包和{b}盒橙汁，每条面包{pa}美元。'+(f'{b}盒橙汁合计{pb}美元。' if batch else f'每盒橙汁{pb}美元。')+'他一共花了多少钱？'
                juice=pb if batch else b*pb;truth=F(a*pa+juice)
                values=[truth,F(a*pa+(b*pb if batch else pb)),F(pa+juice),F(a*pb+b*pa)]
                descriptions=[f'Bread costs {a} * {pa}; orange juice costs {juice} in total.',f'Bread costs {a} * {pa}; orange juice costs {b*pb if batch else pb} in total.',f'The bread costs only {pa}; orange juice costs {juice}.',f'Use ${pb} for each loaf and ${pa} for each carton.']
                program=dict(bread_count=a,bread_unit_price=pa,juice_count=b,juice_price=pb,juice_scope='whole_batch' if batch else 'per_carton',operation='bread_count*bread_unit_price + juice_price*(1 if whole_batch else juice_count)')
                prompt=f'原文中橙汁的${pb}作用于什么范围？';opts=['这一批橙汁的合计价格','每盒橙汁的单价','每条面包的单价'];label='A' if batch else 'B'
            else:
                past=200+j*10;drop=40+j%5;factor=5+j%2;extra=60+j%4;future_target=j%2==0
                now=past-drop;future=factor*now+extra;target='the total number of books he will have in five years' if future_target else 'the increase in his number of books from now to five years from now'
                q=f'Evan had {past} books two years ago. He owns {drop} fewer books now. In five years, he will have {extra} more than {factor} times the number he owns now. What is {target}?'
                zh=f'Evan两年前有{past}本书，现在比那时少{drop}本。五年后，他的书会比现在的{factor}倍多{extra}本。'+('求五年后的书本总数。' if future_target else '求从现在到五年后增加的书本数。')
                truth=F(future if future_target else future-now)
                values=[truth,F(future-now if future_target else future),F(factor*past+extra),F(now+extra)]
                descriptions=[f'Now he has {past} - {drop} = {now}. Five years from now he has {factor} * {now} + {extra} = {future}. Use the requested '+('total.' if future_target else 'increase, subtracting the current count.'),f'Use the '+('increase instead of total.' if future_target else 'future total instead of increase.'),f'Use his past count {past} as the base for the future multiplier.',f'Add only {extra} to the current count, omitting the multiplier.']
                program=dict(past=past,decrease=drop,current=now,future_factor=factor,future_extra=extra,future=future,target='future_total' if future_target else 'increase',operation='future_factor*(past-decrease)+future_extra - (0 if future_total else past-decrease)')
                prompt='最后一句要求的是哪一个量？';opts=['五年后的总数','从现在到五年后新增的数量','两年前的总数'];label='A' if future_target else 'B'
            order=list(range(4));rng.shuffle(order)
            candidates=[dict(index=k,text=descriptions[t]+'\n#### '+str(values[t]),answer=str(values[t]),correct=values[t]==truth,error_family=['intended','binding_or_scope_swap','wrong_quantity','mixed_relation'][t]) for k,t in enumerate(order)]
            assert any(c['correct'] for c in candidates)
            out.append(dict(id=f'r15-derived-{typ}-{j:02d}',parent_train_index=parent,parent_partition='fit',parent_question_sha256=hashlib.sha256(rows[parent]['question'].encode()).hexdigest(),family=typ,question=q,translation=zh,program=program,answer=str(truth),candidates=candidates,review=dict(prompt=prompt,options=opts,program_implied_choice=label),
                            provenance='deterministic parameters and independently authored wording derived from the stated official TRAIN parent; no external dataset/model',
                            supervision_status='program checked; natural-language-to-program fidelity pending blind review; excluded from frozen natural audit and models'))
    (OUT/'derived_training_quarantine.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf-8')
    result=dict(problems=len(out),candidates=sum(len(r['candidates']) for r in out),parents=3,variants_per_family=16,unique_question_texts=len({r['question'] for r in out}),synthetic_evaluation_accuracy=None,used_in_frozen_training=False,used_in_natural_audit=False,validation_access=False,test_access=False)
    (OUT/'generation_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return out


if __name__=='__main__':build()
