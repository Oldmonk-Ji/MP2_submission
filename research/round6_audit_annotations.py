"""Manually authored AFTER source freeze, BEFORE viewing audit predictions.

Original questions only. This is a single-assistant annotation, not independent
human adjudication. All audit errors are retained; no extractor tuning follows.
"""
from round6_annotations import f,t,save_annotations

AUDIT={
4512:([f('age_ratio','isabella','age',2,'ratio',0,'antonio'),f('age','isabella','age',10,'year',1.5)],t('age','antonio','age','month',0)),
867:([f('age_difference','mehki','age',10,'year',0,'jordyn'),f('age_ratio','jordyn','age',2,'ratio',0,'zrinka'),f('age','zrinka','age',6,'year',0)],t('age','mehki','age','year',0)),
930:([f('age_difference','juliet','age',3,'year',0,'maggie'),f('age_difference','juliet','age',-2,'year',0,'ralph'),f('age','juliet','age',10,'year',0)],t('age_sum','maggie+ralph','age','year',0)),
2683:([f('age','denise','age',25,'year',2),f('age_difference','diane','age',-4,'year',0,'denise'),f('threshold','diane','age',25,'year','future')],t('time_to_threshold','diane','age','year')),
4212:([f('count','victor','shrimp',26),f('count_difference','austin','shrimp',-8,'count','now','victor'),f('total_ratio','brian','shrimp',.5,'ratio','now','austin+victor'),f('unit_price','austin+brian+victor','shrimp',7/11,'dollar/shrimp'),f('partition','austin+brian+victor','earnings',3,'person')],t('per_person','austin+brian+victor','earnings','dollar/person')),
2266:([f('count','restaurant','meal_batch_1',10),f('unit_price','restaurant','meal_batch_1',8,'dollar/meal_batch_1'),f('count','restaurant','meal_batch_2',5),f('unit_price','restaurant','meal_batch_2',10,'dollar/meal_batch_2'),f('count','restaurant','meal_batch_3',20),f('unit_price','restaurant','meal_batch_3',4,'dollar/meal_batch_3')],t('revenue','restaurant','meal','dollar')),
2344:([f('count','selene','instant_camera',2),f('unit_price','selene','instant_camera',110,'dollar/instant_camera'),f('count','selene','digital_photo_frame',3),f('unit_price','selene','digital_photo_frame',120,'dollar/digital_photo_frame'),f('discount','selene','all_purchases',.05,'ratio')],t('expense','selene','all_purchases','dollar','after_discount')),
2367:([f('count','adult','ticket',10),f('count','child','ticket',11),f('unit_price','adult','ticket',8,'dollar/ticket'),f('total_price','adult+child','ticket',124,'dollar')],t('unit_price','child','ticket','dollar/ticket')),
4553:([f('count','boyden_family','child',3),f('count','boyden_family','adult',2),f('count','boyden_family','ticket',5),f('unit_price_difference','adult','ticket',6,'dollar/ticket','now','child'),f('total_price','boyden_family','ticket',77,'dollar')],t('unit_price','adult','ticket','dollar/ticket')),
2592:([f('distance_difference','harvey','run',8,'mile','now','sam'),f('distance','sam','run',12,'mile')],t('distance_sum','harvey+sam','run','mile')),
1378:([f('target_cost','alfred','shopping',1000,'dollar'),f('duration','alfred','saving',12,'month'),f('balance','alfred','money',100,'dollar','initial')],t('saving_rate','alfred','saving','dollar/month')),
1895:([f('group_size',"theresa's sons",'meatball',3,'count/plate','initial'),f('count',"theresa's sons",'plate',3),f('consumed_fraction',"theresa's sons",'meatball',2/3,'ratio')],t('balance',"theresa's sons",'meatball')),
3711:([f('count','sofia_family','orange',10,'kg','week1'),f('change','sofia_family','orange',5,'kg','week1'),f('weekly_ratio','sofia_family','orange',2,'ratio','weeks2_3','week1_total'),f('duration','sofia_family','purchase',2,'week','weeks2_3'),f('duration','sofia_family','purchase',3,'week','all')],t('total_purchased','sofia_family','orange','kg','weeks1_3')),
1201:([f('balance','greg','granola_bar',20,'count','initial'),f('daily_reserved','greg','granola_bar',1,'count/day'),f('duration','greg','reservation',7,'day'),f('change','greg','granola_bar',-3),f('count',"greg's sisters",'person',2)],t('received_per_person',"greg's sisters",'granola_bar','count/person')),
4562:([f('work_rate','brendan','grass',8,'yard/day','initial'),f('increase','brendan','grass',.5,'ratio'),f('duration','brendan','work',7,'day')],t('distance','brendan','grass','yard')),
3861:([f('count','jamal','half_dozen_pack',4),f('group_size','jamal','crayon',6,'count/half_dozen_pack'),f('unit_price','jamal','crayon',2,'dollar/crayon')],t('expense','jamal','crayon','dollar')),
}
if __name__=='__main__':save_annotations(AUDIT,'audit_gold.json')
