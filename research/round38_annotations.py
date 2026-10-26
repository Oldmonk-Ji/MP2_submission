"""Evidence-grounded root causes; assistant adjudication, not independent human gold."""
from data_access import ROOT
from round22_common import read,dump,sha
from features import safe_arithmetic,parse_answer
from fractions import Fraction
from collections import Counter

# ID, primary cause, secondary causes, exact question evidence, exact erroneous
# candidate evidence, correct typed relation, candidate relation, executable
# target expression, adjudication. No validation questions occur here.
SPECS=[
(3243,'rate_chain_missing_level',[],
 ['2 apartment buildings with 12 floors each','Each floor has 6 apartments','each apartment needs 7 doors in total'],
 ['**Number of floors**: 2'],
 'floors=buildings*(floors/building); apartments=floors*(apartments/floor); doors=apartments*(doors/apartment)',
 'building_count was assigned to floor_count; omitted floors/building factor','2*12*6*7','semantic_contradiction'),
(3343,'equality_scope',[],
 ['Lilibeth fills 6 baskets where each basket holds 50 strawberries','three of Lilibeth\'s friends pick the same amount as her'],
 ['each of the **3 friends** picks **50 strawberries**'],
 'each_friend_strawberries=Lilibeth_total_strawberries; group_total=(1+friends)*Lilibeth_total',
 'each_friend_strawberries=strawberries_per_basket','(1+3)*6*50','semantic_contradiction'),
(770,'target_aggregate_vs_unit',[],
 ['3/4 of his 400 tickets','Jensen and Finley, who share the tickets in a ratio of 4:11','How many tickets does Finley get?'],
 ['Thus, **Finley gets 20 tickets**.'],
 'Finley_tickets=transferred_tickets*Finley_parts/(Jensen_parts+Finley_parts)',
 'one_ratio_part assigned to Finley_share','400*3/4*11/(4+11)','semantic_contradiction'),
(3307,'inverse_rate_time',[],
 ['Tony can run a mile twice as fast as Tina','with a time of 6 minutes is one-third as fast a runner as Tom'],
 ["Tom's time = $ 3 \\times 6 = 18 $ minutes"],
 'equal_distance implies time inversely proportional to speed; Tom_time=Tina_time/3',
 'Tom_time=3*Tina_time','6/2+6+6/3','semantic_contradiction'),
(918,'recipient_batch_omission',['entity_binding'],
 ['Roselyn gives Mara three times as many books as she gives Rebecca','remains with 60 books','Rebecca received 40 books','how many books did Roselyn have before?'],
 ['After giving 120 books, she still has **60 books**','120 + 60'],
 'initial_books=Mara_transfer+Rebecca_transfer+final_books; Mara_transfer=3*Rebecca_transfer',
 'Rebecca_transfer omitted from conserved total','3*40+40+60','semantic_contradiction'),
(1629,'target_aggregate_vs_unit',[],
 ['buying 5 pizzas','a total of 20 children','divide each pizza into either 6, 8, or 10 slices','how should many slices should she choose to divide the pizzas into'],
 ['So only **40 slices** is divisible by 20.','\\boxed{40}'],
 'target=slices_per_pizza; total_slices=pizza_count*slices_per_pizza; integer(total_slices/children)',
 'total_slices returned instead of slices_per_pizza','40/5','semantic_contradiction'),
(23,'comparison_direction',['calendar_scope'],
 ['50% more expensive during the weekends','his last haircut on Monday','the day before'],
 ['by **subtracting 50% of the previous day\'s price**'],
 'day_before_Monday=Sunday; weekend_price=weekday_price*(1+0.5)',
 'weekend_price=weekday_price*(1-0.5)','18*(1+0.5)','semantic_contradiction'),
(1905,'fraction_base_scope',[],
 ['1/3 of the money on food and 1/5 of it on clothes','3/4 of the remaining money'],
 ['**She spent 1/5 of the remaining money on clothes**'],
 'food=initial/3; clothes=initial/5; saved=(initial-food-clothes)*3/4',
 'clothes=(initial-food)/5','(240-240/3-240/5)*3/4','semantic_contradiction'),
(124,'aggregate_member_omission',[],
 ['two of the stops are for orders of two pizzas and the rest are for single pizzas','minutes per stop'],
 ['\\frac{40}{8} = 5'],
 'all_stops=two_pizza_stops+single_pizza_stops; single_pizza_stops=(pizzas-2*two_pizza_stops)',
 'single_pizza_stop_count substituted for all_stop_count','40/(2+(12-2*2))','semantic_contradiction'),
(2739,'equality_scope',['target_owner'],
 ['five shopping bags that weigh the same amount as Gordon’s two shopping bags','One of Gordon’s shopping bags weighs three pounds and the other weighs seven','Trace’s shopping bags all weigh the same amount'],
 ['So, **one of Gordon’s bags weighs 5 pounds**','5 \\times 5 = 25'],
 'Trace_total_mass=Gordon_total_mass; Trace_mass_per_bag=Trace_total_mass/5',
 'group equality changed to per-bag equality; Gordon average returned as Trace item mass','(3+7)/5','semantic_contradiction'),
(3888,'compound_comparison',[],
 ['Frank will be three times as old as Ty is now','Ty is currently 4 years more than two times as old as Carla is now','Carla is currently 2 years older than Karen is now','Karen is currently 2 years old'],
 ['**Ty is currently 4 years older than Carla**, so Ty is $ 4 + 2 = 6 $ years old'],
 'Ty_now=4+2*Carla_now; Frank_future=3*Ty_now; future target already specified, do not add five again',
 'factor two and offset four collapsed into unrelated addition','3*(4+2*(2+2))','semantic_contradiction'),
(3259,'target_measure_type',[],
 ['$3 for small glasses and $5 for large ones','He has $50','buys 8 small ones and leaves with $1 in change','how many large ones did he buy?'],
 ['x = 50 - 25 = 25','\\boxed{25}'],
 'large_count=(initial_money-change-small_count*small_unit_price)/large_unit_price',
 'large_glasses_cost returned as large_glasses_count','(50-1-8*3)/5','semantic_contradiction'),
(670,'state_transition_order',[],
 ['a $50.00 balance','charged a 20% interest fee','puts $20.00 on his credit card the following month','hit with another 20% interest fee'],
 ['60.00 \\times 20\\% = 12.00','72.00 + 20.00'],
 'balance_1=initial*1.2; pre_second_interest=balance_1+purchase; final=pre_second_interest*1.2',
 'purchase added after second interest instead of before it','(50*1.2+20)*1.2','semantic_contradiction'),
(2003,'algebra_error',[],
 ['32 cherry sours','ratio of cherry sours to lemon sours is 4:5','they make up 25% of the sours in the bag'],
 ['72 + 0.25T = T \\Rightarrow 0.25T = 72'],
 'cherry+lemon+orange=total; orange=.25*total; .75*total=cherry+lemon',
 'correct original relation, incorrect rearrangement; do not label relation comprehension false','(32+32*5/4)/(1-.25)','arithmetic_or_algebra_only'),
(1172,'aggregate_member_omission',[],
 ['12 cans at home and three times as many at his grandparents\' house','His neighbor gave him 46','250 cans from the office','half of his entire amount collected'],
 ['total money = 296 × 0.25'],
 'total_cans=home+grandparents+neighbor+office; savings=total_cans*unit_price/2',
 'only neighbor+office used despite entire-scope target','(12+3*12+46+250)*.25/2','semantic_contradiction'),
(3696,'overlap_multiplicity',[],
 ['Dalton watched 7 movies, Hunter watched 12, and Alex watched 15','They all watched 2 movies together, but every other movie they watched was different'],
 ['Subtract the overlap: $34 - 2 = 32$'],
 'unique_total=sum(person_totals)-(person_count-1)*common_movies',
 'common movies removed once instead of twice','7+12+15-2*2','semantic_contradiction'),
(4611,'state_double_subtraction',['arithmetic_error'],
 ['32 mushrooms','9 mushrooms she could safely eat','Of the rest, she identified twice the amount she ate as poisonous'],
 ['23 - 18 - 9 = 6'],
 'unknown=initial-safe-poisonous; after_safe=initial-safe; unknown=after_safe-poisonous',
 'safe mushrooms subtracted again from after_safe; additionally arithmetic equality is false','32-9-2*9','semantic_and_arithmetic'),
(1202,'amount_rate_mixing',[],
 ['10% of his $150,000 a year salary','20% of the cost of a $450,000 house','How long will it take?'],
 ['15,000 + 90,000 = 105,000'],
 'years=downpayment_money/saving_money_per_year',
 'annual saving rate added to required accumulated money','(.2*450000)/(.1*150000)','semantic_contradiction'),
(2032,'target_component_vs_total',['arithmetic_error'],
 ['Four of them are warehouse workers who make $15/hour','other two are managers who make $20/hour','10% of her workers\' salaries','25 days a month and 8 hours a day','total for their wages and taxes for one month'],
 ['4 workers × 25 days × 8 hours/day = 80 hours','\\boxed{\\$200}'],
 'monthly_total=(4*15+2*20)*25*8*(1+.1)',
 'tax component alone returned instead of wages+taxes; earlier multiplication also wrong','(4*15+2*20)*25*8*1.1','semantic_and_arithmetic'),
(3635,'target_aggregate_vs_unit',[],
 ['37 dollars on notebooks','a total of 12 notebooks','3 red notebooks at 4 dollars each','2 green notebooks at 2 dollars each','How much does each blue notebook cost?'],
 ['37 - 16 = 21','Each blue notebook costs **$21**'],
 'blue_unit_price=(total_spending-red_spending-green_spending)/(total_count-red_count-green_count)',
 'blue_batch_spending relabeled as price per notebook','(37-3*4-2*2)/(12-3-2)','semantic_contradiction'),
(317,'event_inclusion_direction',[],
 ['60,000 fans','75% of the seats were sold','5,000 fans stayed home','How many attended'],
 ['Add the 5,000 who stayed home','45,000 + 5,000 = 50,000'],
 'attended=sold_tickets-nonattendees',
 'nonattendees added to attendees','60000*.75-5000','semantic_contradiction'),
(3448,'arithmetic_error',[],
 ['20% of her salary goes to paying tax, and 5% goes to insurance','a quarter of the money left after the deductions'],
 ['$2000 - $400 - $100 = $1700'],
 'after_deductions=salary-tax-insurance; after_utilities=after_deductions*.75',
 'scope and operation correct; numerical subtraction incorrect','(2000-2000*.2-2000*.05)*.75','arithmetic_or_algebra_only'),
(1164,'time_scope',['target_direction'],
 ['13 cases for a 3-day camp','each child consumes 3 bottles a day','how many more bottles of water does the camp organizer still need to buy?'],
 ['Total number of bottles consumed per day','312 - 189'],
 'needed=children*bottles_per_child_day*camp_days; additional=needed-purchased',
 'one-day demand used for three-day target and shortage direction reversed','(14+16+12+(14+16+12)/2)*3*3-13*24','semantic_contradiction'),
(2172,'state_endpoint',[],
 ['Jack came and took half of all the candies and 4 more candies','Paul came and took the remaining 7 sweets','at first'],
 ['We are told that Paul took 7 sweets, so the final number of sweets is 7'],
 'after_Jack=7; after_Paul=0; initial/2-4=7',
 'remaining-before-Paul amount assigned to after-Paul state','2*(7+4)','semantic_contradiction'),
]


def main():
    out=ROOT/'experiments/round38';lookup={r['train_index']:r for r in read(out/'discovery24.json')};records=[]
    assert set(lookup)=={s[0] for s in SPECS}
    for idx,cause,secondary,qquotes,cquotes,relation,wrong,expr,status in SPECS:
        row=lookup[idx];question=row['question'];candidate=row['candidates'][row['chosen']]
        def evidence(text,quotes):
            result=[]
            for quote in quotes:
                assert quote in text,(idx,quote)
                start=text.index(quote);result.append(dict(text=quote,start=start,end=start+len(quote)))
            return result
        answer=safe_arithmetic(expr);gold=Fraction(parse_answer(row['reference']));assert abs(float(answer)-float(gold))<1e-7,(idx,answer,gold)
        records.append(dict(train_index=idx,fold=0,source='assistant_source_grounded_adjudication',independent_human_gold=False,
            question_sha256=__import__('hashlib').sha256(question.encode()).hexdigest(),chosen_candidate=row['chosen'],
            root_cause=cause,secondary_causes=secondary,status=status,
            question_evidence=evidence(question,qquotes),candidate_evidence=evidence(candidate,cquotes),
            correct_relation=relation,candidate_claim=wrong,target_expression=expr,execution_matches_training_reference=True,
            semantic_negative_allowed=status!='arithmetic_or_algebra_only',
            annotation_boundary='The quoted claim/operation is annotated; not every sentence of the numerically wrong candidate is false.'))
    dump(out/'root_cause_annotations.json',dict(schema='mp2-grounded-root-cause/v3',records=records,test_opened=False))
    report=dict(n=24,primary_counts=dict(Counter(r['root_cause'] for r in records)),status_counts=dict(Counter(r['status'] for r in records)),
        exact_source_spans_checked=True,execution_check_n=24,
        caution='Execution agreement is a sanity check. Semantic adjudications are assistant-reviewed evidence labels, not independent human agreement.',test_opened=False)
    dump(out/'root_cause_summary.json',report);print(report)


if __name__=='__main__':main()
