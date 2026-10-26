"""Post-feature-freeze source review, diagnostic evaluation only; never train labels."""
from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump,sha
from round44_programs import expression,value,edits,valid
from round46_graph import graph,features,csr
import numpy as np,sys,datetime
from collections import Counter,defaultdict

OUT=ROOT/'experiments/round46'
GOLD={
469:('n2-n0-n1','time_state','She is in the process of placing more and more plates','Additional final stage excludes initial stack and already added 37.'),
4663:('n0-n0*n3/100-(n0*n3/100-n5)','fraction_scope','6 less than the first','First day uses total crickets; second day subtracts six from first-day amount, third consumes remainder.'),
1328:('(n1+n2)+(n4*(n1+n2)-n6)-n1-n4*(n1+n2)','time_state','twice as many pins as Richard in the first round','Second-round Patrick references first-round Richard; compare sums across both rounds.'),
3262:('n0*n6*n2+n0*(1+n5/100)*n6*n4','fraction_scope','120% more expensive','Second-half episode cost is 220% of first-half cost, not 120%; each half has eleven episodes.'),
3466:('n0*n2*5+n1','time_state','120 coffee cups in total over the weekend','Weekdays have hourly rate and daily opening time; weekend figure is already a total, not an hourly rate.'),
1380:('2+1+n0+n1','object_binding','Sarah entering the bedroom with 4 friends','Charlie and Susan plus Sarah plus four friends plus eight living-room occupants.'),
4018:('n0-n0*(1-n1)*n2/4','fraction_scope','three-fourths of those hens','Egg-layers are three fourths of hens, not all chickens; complement includes roosters and nonlaying hens.'),
686:('n0*n1/100+n0*(1-n1/100)*(1-n2/100)','fraction_scope','Among the remaining students','Red fraction applies to nonblue remainder; yellow is rest of that group, add blue separately.'),
1957:('n0+n1+n1*(1-n2/100)+(n0+n1*(1-n2/100))','time_state','as many as the sum of the kilometers from Monday and Wednesday','Wednesday uses Tuesday as reduction base; Thursday sums Monday and Wednesday.'),
2341:('n0*n1*n2*n3','object_binding','during the five weekdays','Production per weekday times five days per week times four weeks times price per cake.'),
2467:('n0-(n1+n2-n3)','object_binding','both vegan and kosher','Union subtracts intersection once; neither equals total minus union.'),
3435:('n0*n1/100*(1+n2/100)','fraction_scope','40% heavier than sand','Occupancy fraction and relative material weight are different multipliers; increase factor is 1.4.'),
528:('n2*n0/(n0+n1)','fraction_scope','ratio of books to pens','Book fraction of the total is seven of ten ratio parts, not seven thirds.'),
4056:('n0+n0*n1','object_binding','twice as many as Blaine','Add both fish counts; Keith multiplier references Blaine.'),
1151:('n0*n3/n1*n2/n4','operation_direction','75 chairs in 10 days','Worker requirement proportional to output and inverse time; here durations cancel.'),
710:('(n0-n1-n2-n2*n3-n4)*(1-n5)','fraction_scope','half of the remaining money','Utilities fraction uses car payment; retirement fraction uses balance after all expenses.'),
1214:('n0*(n1+n1/n2)','fraction_scope','a third of his weightlifting time','Additional warmup/cardio per visit is one third of lifting time; then multiply by visits.'),
167:('n0*(1-n2/100)*(1-n3/100)','time_state','dropped another 20% in value','Second depreciation applies to already depreciated value.'),
2097:('n1+n1*n0/100','fraction_scope','80% of the time','One route uses 80% of the other route duration; requested total includes both.'),
692:('n0*(1-n1)','fraction_scope','the rest were cash sales','Cash is complement of credit fraction of total sales.'),
3582:('n2*n0/n1','operation_direction','2 cups of turnips with 5 cups of potatoes','Maintain turnip-to-potato ratio, not turnip-to-total-mixture ratio.'),
4601:('n0+(n0*n3-n2)+(n0+(n0*n3-n2)+n6)','time_state','3 more pages than the sum of the first two nights','Night-two deduction is from twice night one; night three adds three to both previous nights.'),
415:('n0*(n2+n4)-n1*n0*(n0-1)/2','time_state','5 years apart, each','Three sibling ages form an arithmetic sequence; advance all three ages equally.'),
1982:('n0*n1+n2*n3+n4*n5-n6*n7-n8*n9-n10*n11','object_binding','How many total cookies does she keep','Convert each dozen quantity to cookies and subtract only donated batches.'),
}

def prepare():
    assert not (OUT/'held_source_review_before_reference.json').exists()
    for p,h in read(OUT/'held_review_protocol.json')['feature_and_proposal_hashes'].items():assert sha(ROOT/p)==h,p
    rows=[]
    for r in read(OUT/'held_source_review_packet.json'):
        i=r['train_index'];exp,kind,quote,note=GOLD[i];q=r['question'];start=q.index(quote);p=expression(exp)
        rows.append(dict(**r,program=p,expression=exp,relation_type=kind,evidence=dict(start=start,end=start+len(quote),text=quote),
            adjudication=note,source_answer=str(value(p,r['quantities'])),source='assistant source review, not independent human gold'))
    dump(OUT/'held_source_review_before_reference.json',rows)
    dump(OUT/'held_source_review_freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=sha(OUT/'held_source_review_before_reference.json'),code_sha256=sha(__file__)))
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');checks=[]
    for r in rows:checks.append(dict(train_index=r['train_index'],source_answer=r['source_answer'],reference_answer=parse_answer(refs[r['train_index']]['answer']),agrees=r['source_answer']==parse_answer(refs[r['train_index']]['answer'])))
    dump(OUT/'held_execution_checks.json',checks)
    print(dict(n=len(rows),execution_agrees=sum(r['agrees'] for r in checks),disagreements=[r for r in checks if not r['agrees']]),flush=True)

def evaluate():
    assert all(r['agrees'] for r in read(OUT/'held_execution_checks.json'))
    rows=read(OUT/'held_source_review_before_reference.json');pairs=[]
    for r in rows:
        qs=r['quantities'];counts=Counter();seen=set()
        for kind,p,d in edits(r['program'],qs):
            key=repr(p);a=valid(p,qs)
            if a is None or a==r['source_answer'] or key in seen or counts[kind]>=4:continue
            counts[kind]+=1;seen.add(key)
            pairs.append(dict(parent=r['train_index'],q=r['question'],qs=qs,positive=r['program'],negative=p,kind=kind))
    xp=csr([features(r['q'],r['qs'],r['positive']) for r in pairs]);xn=csr([features(r['q'],r['qs'],r['negative']) for r in pairs])
    records=[]
    for arm in ('sft','exact','grpo'):
        margins=[]
        for seed in (46046,46047):
            with np.load(OUT/'diagnostic'/f'{arm}_s{seed}'/'model.npz') as w:
                def score(x):return (np.maximum(np.maximum(x@w['w0']+w['b0'],0)@w['w1']+w['b1'],0)@w['w2']+w['b2']).reshape(-1)
                margins.append(score(xp)-score(xn))
        m=np.mean(margins,axis=0);byparent=defaultdict(list);bykind=defaultdict(list)
        for p,v in zip(pairs,m):byparent[p['parent']].append(bool(v>0));bykind[p['kind']].append(bool(v>0))
        rec=dict(arm=arm,pairs=len(pairs),correct=int((m>0).sum()),parents=len(byparent),parent_mean_accuracy=float(np.mean([np.mean(v) for v in byparent.values()])),
            by_kind={k:dict(n=len(v),correct=sum(v)) for k,v in bykind.items()},parent_accuracy={str(k):float(np.mean(v)) for k,v in byparent.items()})
        records.append(rec);print('NEW_HELD_RELATIONS',rec,flush=True)
    dump(OUT/'held_relation_results.json',dict(records=records,note='24 new source-reviewed parents after feature/proposal freeze; stratified errors/correct controls; no use in any training gradients. Not independent human annotation.',test_opened=False))

if __name__=='__main__':globals()[sys.argv[1]]()
