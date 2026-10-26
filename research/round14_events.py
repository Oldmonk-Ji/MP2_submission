"""Question-only event supervision. Never import validation or test data."""
from data_access import ROOT, CODE, read_jsonl
import json, hashlib

OUT=ROOT/'experiments/round14'
# Each record: quoted evidence, relation type, semantic arguments.
# Symbolic expressions preserve unknown quantities; they are NOT numeric answer labels.
SPECS={
1370: [
 ('5 years ago, Joan had 3 times as much experience as Bill','comparison',dict(subject='Joan',object='Bill',measure='experience_years',factor=3,event_time='not_applicable',state_time='now-5',equation='J_past=3*B_past')),
 ('Now she has twice as much experience as Bill','comparison',dict(subject='Joan',object='Bill',alias='she=Joan',measure='experience_years',factor=2,event_time='not_applicable',state_time='now',equation='J_now=2*B_now')),
 ('How many years of experience does Bill have now?','target',dict(owner='Bill',measure='experience_years',state_time='now',expression='B_now'))],
1476:[
 ('buys handmade rugs at $40 each','unit_price',dict(actor='Mrs. Oaklyn',batch='rugs',direction='purchase',amount=40,unit='USD/rug')),
 ('sells them at $60 each','unit_price',dict(actor='Mrs. Oaklyn',batch='rugs',direction='sale',amount=60,unit='USD/rug',alias='them=rugs')),
 ('she bought 20 rugs','purchase',dict(source='unknown_seller',recipient='Mrs. Oaklyn',batch='rugs',quantity=20,unit='rug',event_time='purchase')),
 ('the profit she will make from selling the rugs','target',dict(owner='Mrs. Oaklyn',measure='net_profit',state_time='after_sale',expression='20*(60-40)'))],
3804:[
 ('500 marbles that Cindy had','state',dict(owner='Cindy',batch='marbles',quantity=500,unit='marble',state_time='before_gifts')),
 ('she gave her four friends 80 marbles each','transfer',dict(source='Cindy',recipient='four_friends',batch='gift_subset',parent_batch='marbles',quantity='4*80',unit='marble',event_time='gift',state_time='not_applicable')),
 ("four times the number of marbles she has remaining",'target',dict(owner='Cindy',measure='scaled_remaining_count',state_time='after_gifts',expression='4*(500-4*80)'))],
3419:[
 ('normally costs $3 is now sold at $2','price_change',dict(batch='milk',old_price=3,new_price=2,unit='USD/gallon',old_state_time='normal',state_time='sale')),
 ('A box of cereal was sold at a discount of $1','discount',dict(batch='cereal',amount=1,unit='USD/box')),
 ('3 gallons of whole milk and 5 boxes of cereal','purchase',dict(source='unknown_seller',recipient='buyer',batches=['milk','cereal'],quantities=[3,5],units=['gallon','box'],event_time='purchase')),
 ('How much will you save via discounts','target',dict(owner='buyer',measure='savings',state_time='purchase',expression='3*(3-2)+5*1'))],
1237:[
 ('$45','unit_price',dict(batch='school_textbooks',amount=45,unit='USD/textbook')),
 ('bookshops outside the school cost 20% less','relative_price',dict(batch='outside_textbooks',reference_batch='school_textbooks',factor=.8)) ,
 ('how much can Peter save','target',dict(owner='Peter',measure='savings',state_time='purchase',expression='3*45*0.2')),
 ('buy 3 math textbooks','purchase',dict(recipient='Peter',source='outside_bookshops',batch='outside_textbooks',quantity=3,unit='textbook',event_time='planned_purchase'))],
1987:[
 ('24 pieces of rare action figures','state',dict(owner='Angela',batch='figures',quantity=24,unit='figure',state_time='initial')),
 ('She sold off a quarter of them at the pawnshop','sale',dict(source='Angela',recipient='pawnshop',batch='sold',parent_batch='figures',quantity='24/4',unit='figure',event_time='first')),
 ('gave one-third of the remainder to her daughter','transfer',dict(source='Angela',recipient='Angela_daughter',batch='gift',parent_batch='figures',quantity='(24-24/4)/3',unit='figure',event_time='after_sale',scope_state_time='after_sale')),
 ('How many does she have left?','target',dict(owner='Angela',measure='remaining_count',state_time='after_gift',expression='(24-24/4)*(1-1/3)'))],
4089:[
 ('100 stickers','state',dict(owner='Clara',batch='stickers',quantity=100,unit='sticker',state_time='initial')),
 ('She gives 10 stickers to a boy she likes','transfer',dict(source='Clara',recipient='boy',batch='boy_gift',parent_batch='stickers',quantity=10,unit='sticker',event_time='first_gift')),
 ('She gives half of the stickers which she has left to her best friends','transfer',dict(source='Clara',recipient='best_friends',batch='friends_gift',parent_batch='stickers',quantity='(100-10)/2',unit='sticker',event_time='second_gift',scope_state_time='after_first_gift')),
 ('How many stickers does Clara have left?','target',dict(owner='Clara',measure='remaining_count',state_time='after_second_gift',expression='(100-10)/2'))],
716:[
 ('3 shirts that cost $15 each','unit_price',dict(batch='cheap_shirts',quantity=3,amount=15,unit='USD/shirt',parent_batch='all_shirts')),
 ('The remaining shirts are $20 each','unit_price',dict(batch='other_shirts',quantity='5-3',amount=20,unit='USD/shirt',parent_batch='all_shirts')),
 ('cost, in dollars, of all 5 shirts','target',dict(owner='buyer',measure='total_cost',state_time='purchase',expression='3*15+(5-3)*20'))],
4327:[
 ('Suki bought 6.5 bags of coffee beans that each weighed 22 kilograms','purchase',dict(source='unknown_seller',recipient='Suki',batch='suki_beans',quantity=6.5,unit='bag',mass_per_bag=22,event_time='purchase')),
 ('Jimmy bought 4.5 bags of coffee beans that each weighed 18 kilograms','purchase',dict(source='unknown_seller',recipient='Jimmy',batch='jimmy_beans',quantity=4.5,unit='bag',mass_per_bag=18,event_time='purchase')),
 ('combined their coffee beans and then repackaged them into 8-kilogram containers','merge_repackage',dict(input_batches=['suki_beans','jimmy_beans'],batch='combined_beans',capacity=8,unit='kg/container',event_time='after_purchase')),
 ('How many containers did they use?','target',dict(owner='Suki_and_Jimmy',measure='container_count',state_time='after_repackage',expression='ceil((6.5*22+4.5*18)/8)'))],
2099:[
 ('grape jelly twice as much as strawberry jelly','comparison',dict(subject='grape',object='strawberry',measure='sold_jars',factor=2,state_time='today',equation='grape=2*strawberry')),
 ('raspberry jelly twice as much as plum jelly','comparison',dict(subject='raspberry',object='plum',measure='sold_jars',factor=2,state_time='today',equation='raspberry=2*plum')),
 ('raspberry jelly sells a third as much as the grape jelly','comparison',dict(subject='raspberry',object='grape',measure='sold_jars',factor='1/3',state_time='today',equation='raspberry=grape/3')),
 ('6 jars of plum jelly today','state',dict(owner='store',batch='plum',quantity=6,unit='sold_jar',state_time='today')),
 ('how many jars of strawberry jelly did they sell?','target',dict(owner='store',measure='strawberry_sold_jars',state_time='today',expression='6*2*3/2'))],
2361:[
 ('Governor Sandoval gave twice as many commencement addresses as Governor Hawkins','comparison',dict(subject='Sandoval',object='Hawkins',measure='address_count',factor=2,state_time='comparison_period',equation='Sandoval=2*Hawkins')),
 ('Governor Sloan gave ten more commencement addresses than Governor Sandoval in the same amount of time','comparison',dict(subject='Sloan',object='Sandoval',measure='address_count',difference=10,state_time='comparison_period',equation='Sloan=Sandoval+10')),
 ('Governor Sandoval gave 12 commencement addresses','state',dict(owner='Sandoval',quantity=12,unit='address',state_time='comparison_period')),
 ('how many commencement addresses did the three of them give altogether?','target',dict(owner='three_governors',measure='total_address_count',state_time='comparison_period',expression='12+12/2+12+10'))],
3077:[
 ('5% discount if the customer pays before the 25th day of the month','conditional_discount',dict(batch='internet_service',discount=.05,condition='payment_day<25',event_time='payment',state_time='monthly_bill')),
 ('monthly internet rate is $50','unit_price',dict(batch='internet_service',amount=50,unit='USD/month')),
 ('he paid every 25th of the month','payment',dict(source='customer',recipient='provider',batch='internet_service',event_time='day_25',state_time='monthly_bill')),
 ('how much did the customer pay in all for 4 months','target',dict(owner='customer',measure='total_payment',state_time='four_months',expression='4*50',status='pending_review',alternatives=['strict_before:4*50','inclusive_deadline:4*50*(1-0.05)']))],
438:[
 ('5 packs of tuna for $2 each','unit_price',dict(batch='tuna',quantity=5,amount=2,unit='USD/pack')),
 ('4 bottles of water for $1.5 each','unit_price',dict(batch='water',quantity=4,amount=1.5,unit='USD/bottle')),
 ('she paid $56 for her shopping','payment',dict(source='Barbara',recipient='supermarket',batch='whole_basket',quantity=56,unit='USD',event_time='checkout')),
 ('How much did Barbara spend on different than the mentioned goods?','target',dict(owner='Barbara',measure='other_goods_cost',state_time='checkout',expression='56-5*2-4*1.5'))],
733:[
 ('500 lychees','state',dict(owner='Mr. Shaefer',batch='lychees',quantity=500,unit='lychee',state_time='initial')),
 ('He sold half of them','sale',dict(source='Mr. Shaefer',recipient='market_buyers',batch='sold',parent_batch='lychees',quantity='500/2',unit='lychee',event_time='market')),
 ('they ate 3/5 of them','consumption',dict(actor='household',batch='home_remainder',parent_batch='lychees',quantity='(500-500/2)*3/5',unit='lychee',event_time='after_market',scope_state_time='after_sale')),
 ('number of lychees remaining','target',dict(owner='household',measure='remaining_count',state_time='after_eating',expression='(500-500/2)*(1-3/5)'))],
2236:[
 ('50 rows of purple beads with 20 beads per row','batch_quantity',dict(batch='purple',quantity='50*20',unit='bead')),
 ('40 rows of blue beads with 18 beads per row','batch_quantity',dict(batch='blue',quantity='40*18',unit='bead')),
 ('80 gold beads','batch_quantity',dict(batch='gold',quantity=80,unit='bead')),
 ('$1 per 10 beads','unit_price',dict(batches=['purple','blue','gold'],amount=1,denominator=10,unit='USD/10_beads')),
 ('how much do all the beads she bought cost?','target',dict(owner='Carly',measure='total_cost',state_time='purchase',expression='(50*20+40*18+80)/10'))],
1511:[
 ('180 seashells','state',dict(owner='Ali',batch='shells',quantity=180,unit='shell',state_time='initial')),
 ('gave away 40 seashells to his friends','transfer',dict(source='Ali',recipient='friends',batch='friends_gift',parent_batch='shells',quantity=40,unit='shell',event_time='first')),
 ('gave 30 seashells to his brothers','transfer',dict(source='Ali',recipient='brothers',batch='brothers_gift',parent_batch='shells',quantity=30,unit='shell',event_time='second')),
 ('sold half of the remaining seashells','sale',dict(source='Ali',recipient='unknown_buyers',batch='sold',parent_batch='shells',quantity='(180-40-30)/2',unit='shell',event_time='after_gifts',scope_state_time='after_gifts')),
 ('how many seashells did he have left?','target',dict(owner='Ali',measure='remaining_count',state_time='after_sale',expression='(180-40-30)/2'))],
2385:[
 ('2 fish every day for 30 days','production',dict(actor='Luke',batch='fish',quantity='2*30',unit='fish',event_time='summer_30_days')),
 ('each fish gives him 2 fillets','conversion',dict(input_batch='fish',batch='fillets',factor=2,unit='fillet/fish')),
 ('how many fish fillets does he have?','target',dict(owner='Luke',measure='fillet_count',state_time='after_processing',expression='2*30*2'))],
2984:[
 ('twice as many hats as scarves','comparison',dict(subject='hats',object='scarves',measure='item_count',factor=2,equation='hats=2*scarves',state_time='planned_purchase')),
 ('60% of her money on hats and the rest on scarves','budget_partition',dict(owner='Kiki',shares={'hats':.6,'scarves':.4},state_time='planned_purchase')),
 ('she currently has $90','state',dict(owner='Kiki',quantity=90,unit='USD',state_time='now')),
 ('they are sold at $2 each','unit_price',dict(batch='scarves',amount=2,unit='USD/scarf',alias='they=scarves')),
 ('how many scarves will she buy','target',dict(owner='Kiki',measure='scarf_count',state_time='planned_purchase',expression='90*(1-0.6)/2'))],
4333:[
 ('Angie bought three times as many pretzels at the mall as Shelly did','comparison',dict(subject='Angie',object='Shelly',measure='purchased_pretzels',factor=3,equation='Angie=3*Shelly',state_time='purchase')),
 ('Shelly bought half as many pretzels as Barry','comparison',dict(subject='Shelly',object='Barry',measure='purchased_pretzels',factor=.5,equation='Shelly=Barry/2',state_time='purchase')),
 ('Barry bought 12 pretzels','purchase',dict(source='unknown_seller',recipient='Barry',batch='barry_pretzels',quantity=12,unit='pretzel',event_time='purchase')),
 ('how many did Angie buy?','target',dict(owner='Angie',measure='purchased_pretzels',state_time='purchase',expression='12/2*3'))],
840:[
 ('14 eggplants in every seed packet','conversion',dict(input_batch='eggplant_seeds',batch='eggplants',factor=14,unit='plant/packet')),
 ('10 sunflowers in every seed packet','conversion',dict(input_batch='sunflower_seeds',batch='sunflowers',factor=10,unit='plant/packet')),
 ('4 seed packets of eggplants and 6 seed packets of sunflower','purchase',dict(source='unknown_seller',recipient='Shyne',batches=['eggplant_seeds','sunflower_seeds'],quantities=[4,6],units=['packet','packet'],event_time='purchase')),
 ('how many plants can Shyne grow in her backyard?','target',dict(owner='Shyne',measure='potential_plant_count',state_time='after_growth',expression='4*14+6*10'))]
}

# Migrate human-disputed targets without converting either interpretation into gold.
SPECS[2549]=[
 ('200 gallons of milk every day','production',dict(actor='Jesiah_cows',batch='milk',quantity=200,unit='gallon/day',event_time='daily')),
 ('total income','target',dict(owner='Jesiah',measure='income',state_time='June',status='ambiguous',alternatives=['gross_sales','net_after_maintenance_and_feed'],human_choice='gross_sales',reference_convention='net_after_maintenance_and_feed'))]
SPECS[4310]=[
 ('His sister gave him','transfer',dict(source='Joel_sister',recipient='Joel',batch='sister_gift',quantity='s',unit='toy',event_time='collection')),
 ('from his own closet','origin',dict(owner='Joel',batch='closet_toys',event_time='collection')),
 ('How many of the donated toys were Joel’s?','target',dict(owner='Joel',measure='originally_owned_donation_count',state_time='before_donation',status='ambiguous',alternatives=['opening_bin_is_summary','opening_bin_is_additional_batch'],human_choice='opening_bin_is_additional_batch',reference_convention='opening_bin_not_separately_counted'))]

def main():
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    plan=json.loads((OUT/'plan.json').read_text())
    assert set(SPECS)==set(plan['train'])|{2549,4310}
    assert not set(SPECS)&set(plan['sealed_audit'])
    records=[]
    for i, facts in SPECS.items():
        q=refs[i]['question']; atoms=[]
        for j,(quote,kind,args) in enumerate(facts):
            start=q.index(quote)
            status=args.get('status','explicit_relation_with_derived_expression' if 'expression' in args or 'equation' in args else 'explicit')
            atoms.append(dict(id=f'{i}:{j}',type=kind,evidence={'start':start,'end':start+len(quote),'text':quote},arguments=args,status=status,supervise=status not in ('pending_review','ambiguous')))
        records.append(dict(index=i,split='train',question=q,question_sha256=hashlib.sha256(q.encode()).hexdigest(),source='assistant_question_only_annotation',independent_human_gold=False,atoms=atoms))
    payload={'schema':'mp2-event-relations/v2','records':records}
    (OUT/'event_annotations.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    # Flat typed relations are training targets, not input features or deployed lookup.
    with (OUT/'supervision.jsonl').open('w',encoding='utf-8') as f:
        for r in records:
            for a in r['atoms']:
                if a['supervise']:f.write(json.dumps(dict(question=r['question'],train_index=r['index'],**a),ensure_ascii=False)+'\n')
    print(json.dumps({'questions':len(records),'atoms':sum(len(r['atoms']) for r in records),'supervised':sum(a['supervise'] for r in records for a in r['atoms'])}))

if __name__=='__main__':main()
