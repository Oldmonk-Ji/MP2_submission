"""Source-reviewed TRAIN discovery annotations; never imported at inference.

Expressions were written from question text before loading reference answers.
Unknown cases remain in ordinary TRAIN; only process supervision is withheld.
"""
from data_access import ROOT, CODE, read_jsonl, parse_answer
from round22_common import read, dump, sha
from round44_programs import expression, value, quantities
import hashlib, json, datetime

OUT=ROOT/'experiments/round46'
# id: (source-occurrence expression, relation type, exact evidence, adjudication)
GOLD={
1633:('(n0*60+n1-n2*60-n3*60-n5-n6-n7*60-n8)/60','time_state','11 hours 20 minutes','All activities consume disjoint portions of the flight; two movies is not two additional hours.'),
2874:('n0*(n1+n3+n6)*n7','object_binding','paid each of them $10 per hour','Sum three workdays per worker, multiply by worker count and hourly pay; ordinals are not durations.'),
1613:('n0*n4+n1*n5+(n2-n3)*n6+n7','object_binding','a box of hair dye for $10 to dye every head','Dye cost is per dye job, tips are added once; haircut and perm revenues have separate quantities.'),
685:('n0*n1/100*n3/100','fraction_scope','Of the students who are accepted','Attendance fraction applies to accepted students, not all applicants; repeated 20000 is the same cohort.'),
1572:('n0*n1+n2-n0','time_state','When he is 50 years old','Both siblings age by the same elapsed time; their current age ratio does not stay constant.'),
2739:('(n3+n4)/n0','object_binding','five shopping bags that weigh the same amount','Equality concerns total bag weight; divide Gordon total by Trace bag count.'),
3565:('n2+n0+n3/n1','operation_direction','Marie bought six times as many packs','Allison glue is Marie glue plus eight; Allison paper is Marie paper divided by six.'),
2000:('n0*(1+n2)*(1+n2)','time_state','Every night, each vampire turns 5 more people','Each existing vampire remains and adds five; repeat for the two nights, below population cap.'),
498:('(n0-n1-n2*n1)/5','object_binding','95 cents in nickels','Subtract gifts in cents, then convert remaining cents to nickels; a nickel is five cents.'),
4094:('n3-(n0-n0*n2)','fraction_scope','3/5 of its carrying capacity','Occupied seats use bus capacity as fraction base; subtract remaining seats from waiting passengers.'),
446:('n0*(1+n1/100)*(1+n2/100)','time_state','60% faster than he ran on Thursday','Friday increase is based on Thursday, which already includes the Monday-relative increase.'),
4668:('(n0+n6)*n7+n3','time_state',"she didn't read the remaining pages",'Count ten complete chapters, twenty partial pages and two additional complete chapters, not eleven complete chapters.'),
460:('(n2-n1)/(1-n0/10)','fraction_scope','nine-tenths of the dust particles','Final dust minus added dust equals one tenth of the original; ten is the explicit denominator in nine-tenths.'),
160:('(n0*n2-n3)*n1','object_binding',"two courses of the last wall",'Subtract two courses once across all four walls, then multiply by bricks per course.'),
860:('n1+(n1+n2)+2*(n1+n2-n3)+n1*n4','time_state','on both Wednesday and Thursday than on Tuesday','Wednesday and Thursday are each Tuesday minus four; Friday half uses Monday as its base.'),
3562:('((n0-n1)*(1-n2)-n3)*(1-n4)','fraction_scope','Half of the rest are shale','Successive fractions use the remaining population after earlier categories; final remaining layers each have a distinct rock kind.'),
66:('n0*n1*n2*n3','object_binding','2 of each color crayon in each box','Colors times crayons/color/box times boxes/hour times hours.'),
3833:('(n2+n2*(1+n4/100))*n5*n6','object_binding','At the second hospital','Add daily patient counts at the two hospitals before multiplying by workdays and working weeks.'),
4286:('n0+n0*n1-n2','time_state','on Monday than Tuesday','Monday includes lunch and dinner; Tuesday includes one breakfast sandwich.'),
1321:('n0*(n1+n2)*n3','object_binding','each carriage could accommodate 10 more passengers','Extra capacity is per carriage; multiply complete per-carriage capacity by carriages and trains.'),
2752:('n0*n3+n1*n4*n3+n2*n5/100*n3','fraction_scope',"250% of a worker's salary",'Two construction salaries plus one double salary plus one 250%-of-base salary; 250% is not a 250% increase.'),
3892:('n2+(n0-n1*n2)/n3','time_state','before she gets tired','First segment distance is speed times duration; remaining distance uses the lower speed, then add segment times.'),
124:('n3/(n1+n0-n1*n2)','object_binding','two of the stops are for orders of two pizzas','Stops are two double orders plus remaining single orders; divide available time by stops, not pizzas.'),
2980:('n0/7-n2*n3','object_binding','every day of the week','Required daily pages minus planned daily pages; seven days per week.'),
2772:('n0+n0*n1-n2','fraction_scope','both silk and cashmere fabrics','Inclusion-exclusion: silk count plus cashmere count minus union; half is based on silk count.'),
3046:('n0*2+n1*4','object_binding','7 chickens and 5 sheep','Chickens have two legs and sheep four; species-specific rates, not total animals times one shared rate.'),
851:('n0*n3+n0*n1*n4+n0*n1*n2*n5','object_binding','three times as many giraffe enclosures as zebra enclosures','Zebra enclosures are per tiger enclosure; giraffe multiplier uses zebra enclosure count; sum species animal totals.'),
1598:('n0+n1+n2','time_state','how much money, in dollars, did she have at first','Initial money equals purchases plus remaining budget, not the remaining budget alone.'),
2189:('n0-n2-n3-n3*n4','object_binding','They put the border together first','Border, Trevor and Joe placements are separate contributions; Joe has three times Trevor placements.'),
1308:('1/n1','fraction_scope','1/10 of the episodes he bought','Each day watches a fixed fraction of the original purchase, not a shrinking remainder.'),
2278:('n0*(1-n2)*n1','fraction_scope','paints the rest of the rooms with purple','Complement fraction selects purple rooms; multiply by walls per room.'),
3131:('n0+(n2-n1)*n3','time_state','3 new goldfish every week','Weekly deaths and purchases are both repeated; final state equals initial plus seven net changes.'),
4513:('(n0*n1+n0*n2*n1*n3)*7','fraction_scope','each sheep drinking 1/4 times as much water as a cow','Sheep count and per-sheep water use have different bases; sum species daily water and convert to one week.'),
3888:('n1*(n2+n3*(n4+n5))','time_state','how old will Frank be in 5 years','The first relation already defines Frank future age; do not add or subtract five again.'),
4294:('(n2+n3)-(n1+n3)','object_binding','contained the color red than contained the color white','Red-white flowers belong to both totals and cancel; red-yellow versus yellow-white determines the difference.'),
3587:('n0*n1*n3','object_binding','removes 1 washing machine from each box','Target is removed machines, not initial or remaining inventory; one removal per box.'),
3286:('(n0*n1+n2*n3+n4)+(n0*n5*n6+n2*n7*n8+n4*n7)','fraction_scope','half the 3 pointers and free throws','Opponent doubles number of two-point baskets and halves both other scoring counts; sum both teams scores.'),
2934:('(n0+(n0*n2-n1)+(n0*n2-n1)*n3)*n4','fraction_scope','1/3 the number of food-throwing time-outs','Swearing fraction uses food-throwing total after its subtraction, then all time-outs have the same duration.'),
3000:('(n0*n2+n1*n3)*n4','object_binding','4 adult patients and 3 child patients every hour','Bind each patient type to its own fee, sum hourly receipts, multiply by eight hours.'),
947:('n0*(n3+(n1-1)*n4)','object_binding','the rest of the floors','First-floor capacity counted once per building; other floors count is stories minus one.'),
791:('5+n0*n1','object_binding',"number of letters in the player's name",'Sarah has five letters; add twice current age. Five is a text-derived count, not a supplied numeric occurrence.'),
4460:('(1+1+n0)*2+(n1+n2)*4','object_binding','Including Javier','People include Javier, wife and children with two legs; dogs and cat each four legs.'),
49:('n2/(1+n0+n0*n1)*n0*n1','operation_direction','5 times the price of the table','Chair=one unit, table=three chair units, couch=fifteen chair units; infer unit price from total.'),
3448:('n0*(1-n1/100-n2/100)*(1-n3)','fraction_scope','money left after the deductions','Tax and insurance both use salary base; utility fraction applies to the post-deduction balance.'),
1276:('n0*n1/n2','object_binding','withdraws $300 from each bank','Withdrawal is per bank; total withdrawn divided by note denomination.'),
4287:('n0*n2/n3+n0*n1*n2/n4','object_binding','Color paint comes in 1-gallon paint cans','Bedrooms and other rooms use different can capacities; both divisions are integral here.'),
113:('n0+n2-n1','time_state','when Bella turns 18','Age difference is invariant; Marianne advances by Bella elapsed age.'),
2813:('2*(n0+n1*n2)','object_binding','Jamie and Oliver','Two travelers; airfare once per person, hotel per person per day.'),
1912:('n3*n4/(n0*n1/100)','fraction_scope','25% of their tail feathers at one time','Safe yield per flamingo is percentage of its tail feathers; divide total boa feather demand by safe yield.'),
2255:('2*n1+n2+n3-n4-n5','time_state','Two of the chaperones were parents in that group','Initially two student groups plus parents and teachers; subtract departed students and departed parents once each.'),
2415:('n5-n1*(n3+n4)','object_binding','how many kitchens are there on the entire space station','Subtract all bedrooms and bathrooms from total rooms; no extra division by cylinders for the requested entire-station total.'),
3696:('n1+n2+n3-2*n4','object_binding','They all watched 2 movies together','Shared films counted three times in individual totals, so subtract two duplicate copies, not all three.'),
3079:('(n2/n0+2*n2/n1)/3*60','time_state','Sunday, Tuesday, and Thursday','One walking trip and two running trips; average trip durations, not speeds, then convert hours to minutes.'),
4029:('n0/(1+n2+n1*n2)','operation_direction','Snap spends twice as much as Crackle','Pop=one unit, Crackle=three, Snap=six; divide total by ten units.'),
}
UNKNOWN={
1629:'Discrete divisibility choice requires a predicate/selection operator; keep source relation but do not pretend a numeric literal is an executed solution.',
1694:'Question requires rounding to nearest integer percent; current four-operation executor cannot express final rounding.',
4093:'The antecedent of twigs it needs may refer to additional twigs versus total nest; withhold process label pending independent clarification.',
709:'Exact day count across unspecified three years and whole-card purchase semantics are implicit; do not certify a unique process from numeric agreement.',
4177:'Twice as much Tuna has an unstated comparison base (trout versus salmon); do not invent a certain antecedent.',
2610:'Three times more is linguistically ambiguous between threefold and fourfold; retain as ordinary training, no certain process label.',
}

def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'discovery_review_before_reference.json').exists()
    old={r['train_index'] for r in read(ROOT/'experiments/round44/reviewed_programs.json')}
    rows=[json.loads(s) for s in (ROOT/'experiments/round38/all_train_oof_errors.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=sorted([r for r in rows if r['fold']!=2 and r['train_index'] not in old],key=lambda r:hashlib.sha256(('round46:'+str(r['train_index'])).encode()).hexdigest())[:60]
    assert {r['train_index'] for r in rows}==set(GOLD)|set(UNKNOWN)
    out=[]
    for r in rows:
        i=r['train_index'];q=r['question'];qs=quantities(q)
        a=dict(train_index=i,fold=r['fold'],question=q,source='assistant source adjudication; not independent human gold',quantities=qs)
        if i in UNKNOWN:a.update(status='unknown_or_executor_unsupported',reason=UNKNOWN[i]);out.append(a);continue
        exp,kind,quote,note=GOLD[i];start=q.index(quote);p=expression(exp)
        a.update(status='source_reviewed',program=p,expression=exp,relation_type=kind,
                 evidence=dict(start=start,end=start+len(quote),text=quote),adjudication=note,
                 source_derived_answer=str(value(p,qs)))
        out.append(a)
    dump(OUT/'discovery_review_before_reference.json',out)
    dump(OUT/'discovery_review_freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         hash=sha(OUT/'discovery_review_before_reference.json'),code_hash=sha(__file__),
         note='Expressions source-reviewed before this script loads official references. Historical TRAIN and old error packet references have been available previously; not a blinded human study.'))
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');checks=[]
    for r in out:
        if r['status']!='source_reviewed':continue
        correct=r['source_derived_answer']==parse_answer(refs[r['train_index']]['answer'])
        checks.append(dict(train_index=r['train_index'],execution_agrees=correct,source_answer=r['source_derived_answer'],reference_answer=parse_answer(refs[r['train_index']]['answer'])))
    dump(OUT/'discovery_execution_checks.json',checks)
    print(dict(reviewed=len(out),source_programs=len(checks),unknown=len(UNKNOWN),execution_agrees=sum(r['execution_agrees'] for r in checks)),flush=True)
    print('DISAGREEMENTS',[r for r in checks if not r['execution_agrees']],flush=True)

if __name__=='__main__':prepare()
