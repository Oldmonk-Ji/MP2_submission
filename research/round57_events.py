"""Source-reviewed TRAIN-fit event supervision. No TEST or held labels."""
from fractions import Fraction
import json
from round22_common import ROOT,load,dump,sha
import numpy as np

def execute(event,balances):
    out={k:Fraction(str(v)) for k,v in balances.items()}
    amount=Fraction(event['amount'])
    for owner,sign in ((event['giver'],-1),(event['receiver'],1)):
        if owner is not None:out[owner]=out.get(owner,Fraction(0))+sign*amount
    return out

def seeds():
    # Amount expressions are separately source reviewed, not inferred from answers.
    return [
      (99,'books',{'Mike':45,'Corey':90,'Lily':0},[
        ('Mike','Lily','10','10','Mike gave 10 books to Lily'),
        ('Corey','Lily','25','10+15','Corey gave Lily 15 more than Mike gave')],{'Mike':35,'Corey':65,'Lily':35},'Lily','after',35),
      (404,'popsicle sticks',{'Miss Davis':170,'groups':0},[
        ('Miss Davis','groups','150','15*10','Miss Davis gave 15 popsicle sticks to each of the 10 groups')],{'Miss Davis':20,'groups':150},'Miss Davis','after',20),
      (417,'pieces of gum',{'Elyse':100,'Rick':0,'Shane':0},[
        ('Elyse','Rick','50','Elyse_before/2','She gave half to her brother Rick'),
        ('Rick','Shane','25','Rick_after_event0/2','He gave half of his to his friend Shane'),
        ('Shane',None,'11','11','Shane chewed 11 pieces of gum')],{'Elyse':50,'Rick':25,'Shane':14},'Shane','after',14),
      (447,'bones',{'Lassie':50},[
        ('Lassie',None,'25','Lassie_before/2','Lassie eats half of her bones on Saturday'),
        (None,'Lassie','10','10','On Sunday she received 10 more bones')],{'Lassie':35},'Lassie','before',50),
      (498,'cents',{'Ray':95,'Peter':0,'Randi':0},[
        ('Ray','Peter','25','25','Ray gives 25 cents to Peter'),
        ('Ray','Randi','50','2*25','twice as many cents to Randi as he gave to Peter')],{'Ray':20,'Peter':25,'Randi':50},'Ray','after_then_divide_5_cents_per_nickel',4),
      (689,'candies',{'Billy':6,'Caleb':11,'Andy':9,'father':36},[
        ('father','Billy','8','8','He gave 8 candies to Billy'),
        ('father','Caleb','11','11','11 to Caleb'),
        ('father','Andy','17','36-8-11','the rest to Andy')],{'Billy':14,'Caleb':22,'Andy':26,'father':0},'Andy_minus_Caleb','after',4)
    ]

def main():
    out=ROOT/'experiments/round57';out.mkdir(exist_ok=False)
    bank,_,_,ids=load('train')
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:folds=z['folds'];assert np.array_equal(z['ids'],ids)
    rows={int(i):(r,int(f)) for i,r,f in zip(ids,bank,folds)};reviews=[];pairs=[]
    for idx,unit,start,events,end,target,time,answer in seeds():
        row,fold=rows[idx];assert fold!=2
        q=row['question'];state={k:Fraction(v) for k,v in start.items()};annot=[]
        for step,(giver,receiver,amount,expression,evidence) in enumerate(events):
            a=q.index(evidence);event=dict(giver=giver,receiver=receiver,entity=unit,amount=amount,amount_expression=expression,
              time_before=step,time_after=step+1,source_span=[a,a+len(evidence)],evidence=evidence)
            before=state;state=execute(event,before)
            event.update(state_before={k:str(v) for k,v in before.items()},state_after={k:str(v) for k,v in state.items()})
            annot.append(event)
            # Direction reversal has a deterministically different state, even if later cancellation could hide it.
            negative={**event,'giver':receiver,'receiver':giver}
            assert execute(negative,before)!=state
            negative['state_after']={k:str(v) for k,v in execute(negative,before).items()}
            pairs.append(dict(train_index=idx,fold=fold,event_index=step,kind='transfer_direction',positive=event,negative=negative,
              negative_state={k:str(v) for k,v in execute(negative,before).items()},supervision='source_reviewed_not_answer_only'))
        assert state=={k:Fraction(v) for k,v in end.items()},idx
        if idx==447:computed=(state['Lassie']-10)*2
        elif idx==498:computed=state['Ray']/5
        elif idx==689:computed=state['Andy']-state['Caleb']
        else:computed=state[target]
        assert computed==answer
        reviews.append(dict(train_index=idx,fold=fold,question=q,initial_state=start,events=annot,final_state=end,
          query_owner=target,query_time=time,source_derived_answer=str(computed),
          review_status='assistant_source_reviewed_not_independent_human_gold',
          limitations='Amounts resolved manually; no automatic event extractor or reference-derived process labels.'))
    temporal=[]
    for r in reviews:
        for e in r['events']:
            for owner in (e['giver'],e['receiver']):
                if owner is None:continue
                prior=e['state_before'].get(owner,'0');after=e['state_after'][owner]
                assert prior!=after
                temporal.append(dict(train_index=r['train_index'],fold=r['fold'],owner=owner,entity=e['entity'],
                  source_span=e['source_span'],time=e['time_after'],positive=after,negative=prior,
                  kind='after_state_not_before_state',event=e))
    dump(out/'source_reviewed_events.json',reviews);dump(out/'transition_preferences.json',pairs)
    dump(out/'temporal_preferences.json',temporal)
    dump(out/'checks.json',dict(parents=len(reviews),events=len(pairs),all_fit_only=True,evidence_spans_verified=True,
      exact_state_execution_verified=True,direction_negatives_verified=True,test_opened=False,held_opened_for_rule_design=False,
      before_state_inverse_case=447,held_metrics=None,training_run=False))
    dump(out/'freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research/round57_events.py',*out.glob('*.json')]}))
    print(dict(parents=len(reviews),events=len(pairs),all_checks_passed=True))

if __name__=='__main__':main()
