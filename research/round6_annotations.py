"""Manual semantic annotations from TRAIN question text, not model outputs.

Do not import this module from any extractor or selector. The audit annotation
file is created only after the extractor source is frozen.
"""
import json
from data_access import ROOT,CODE,read_jsonl

OUT=ROOT/'experiments/round6'
def f(kind,owner,item,value,unit='count',time='now',related=None):
    return dict(kind=kind,owner=owner,item=item,value=value,unit=unit,time=time,related=related)
def t(kind,owner,item,unit='count',time='final'):
    return dict(kind=kind,owner=owner,item=item,unit=unit,time=time)

DEV={
1967:([f('age','doug','age',40,'year',0),f('age_sum','betty+doug','age',90,'year',0),f('price_age_ratio','betty','nut_pack',2,'dollar/year',0),f('count','betty','nut_pack',20)],t('expense','betty','nut_pack','dollar')),
370:([f('age','kimiko','age',26,'year',0),f('age_ratio','kayla','age',.5,'ratio',0,'kimiko'),f('threshold','kayla','age',18,'year',0)],t('time_to_threshold','kayla','age','year')),
3215:([f('age','ali','age',8,'year',0),f('age_difference','ali','age',3,'year',0,'yusaf'),f('age_ratio','umar','age',2,'ratio',0,'yusaf')],t('age','umar','age','year',0)),
1203:([f('age','mason','age',20,'year',0),f('age_ratio','mason','age',1/3,'ratio',0,'sydney'),f('age_difference','sydney','age',-6,'year',0,"mason's father")],t('age',"mason's father",'age','year',0)),
4276:([f('count','james','shirt',3),f('total_price','james','shirt',60,'dollar','before_discount'),f('discount','james','shirt',.4,'ratio')],t('unit_price','james','shirt','dollar/shirt','after_discount')),
95:([f('count','jimmy','pen',3),f('unit_price','jimmy','pen',1,'dollar/pen'),f('count','jimmy','notebook',4),f('unit_price','jimmy','notebook',3,'dollar/notebook'),f('count','jimmy','folder',2),f('unit_price','jimmy','folder',5,'dollar/folder'),f('payment','jimmy','money',50,'dollar')],t('change','jimmy','money','dollar')),
1790:([f('earn_rate','alfonso','work',6,'dollar/day'),f('target_cost','alfonso','helmet',340,'dollar'),f('balance','alfonso','money',40,'dollar','initial'),f('work_rate','alfonso','work',5,'day/week')],t('duration','alfonso','work','week')),
443:([f('earn_rate','jackson','chore',5,'dollar/hour'),f('duration','jackson','vacuum',2,'hour'),f('repetition','jackson','vacuum',2),f('duration','jackson','dish',.5,'hour'),f('duration_ratio','jackson','bathroom',3,'ratio','now','dish')],t('earnings','jackson','chore','dollar')),
4046:([f('total_price','father','lumber',450,'dollar','initial'),f('total_price','father','nail',30,'dollar','initial'),f('total_price','father','fabric',80,'dollar','initial'),f('increase','father','lumber',.2,'ratio'),f('increase','father','nail',.1,'ratio'),f('increase','father','fabric',.05,'ratio')],t('increase_amount','father','project','dollar')),
1922:([f('balance',"sara's sister",'notebook',4,'count','initial'),f('change',"sara's sister",'notebook',6),f('change',"sara's sister",'notebook',-2)],t('balance',"sara's sister",'notebook')),
4142:([f('count','cyrus','bite',14),f('change','cyrus','bite',10),f('count','family_excluding_cyrus','person',6),f('total_ratio','family_excluding_cyrus','bite',.5,'ratio','now','cyrus')],t('per_person','family_excluding_cyrus','bite','count/person')),
140:([f('balance','pete','money',2.5,'dollar','initial'),f('balance','raymond','money',2.5,'dollar','initial'),f('expense','pete','money',4,'nickel'),f('balance','raymond','money',7,'dime','final')],t('expense','pete+raymond','money','cent')),
1920:([f('count','developer','land',4,'acre'),f('unit_price','developer','land',1863,'dollar/acre'),f('partition','developer','land',9,'lot')],t('unit_price','developer','land','dollar/lot')),
4023:([f('balance','mother','sweet',27,'count','initial'),f('retained_fraction','mother','sweet',1/3,'ratio'),f('count','mother','child',3),f('received','eldest','sweet',8),f('received_ratio','youngest','sweet',.5,'ratio','now','eldest')],t('received','second_child','sweet')),
1095:([f('balance','abraham','money',60,'dollar','initial'),f('count','abraham','shower_gel',4),f('unit_price','abraham','shower_gel',4,'dollar/shower_gel'),f('count','abraham','toothpaste',1),f('total_price','abraham','toothpaste',3,'dollar'),f('balance','abraham','money',30,'dollar','final')],t('expense','abraham','laundry_detergent','dollar')),
2285:([f('count','zara','cow',24),f('count','zara','sheep',7),f('partition','zara','animal',3,'group'),f('group_size','zara','animal',48,'count/group')],t('count','zara','goat','count','now')),
}
def save_annotations(data,name):
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');plan=json.loads((OUT/'plan.json').read_text());rows=[]
    split='development' if name=='development_gold.json' else 'audit'
    expected={i for cat,v in plan['split'].items() for i in v[split]};assert set(data)==expected
    for category,v in plan['split'].items():
        for i in v[split]:
            facts,target=data[i];rows.append(dict(index=i,category=category,question=refs[i]['question'],facts=facts,target=target))
    (OUT/name).write_text(json.dumps(dict(annotation_source='manual question-text annotation; no candidate or reference answers',annotator='same assistant; not external adjudication',rows=rows),indent=2,ensure_ascii=False),encoding='utf-8')
if __name__=='__main__':save_annotations(DEV,'development_gold.json')
