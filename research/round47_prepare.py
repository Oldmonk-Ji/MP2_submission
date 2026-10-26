"""Correct process-supervision preservation before definitive ablation.

The round46 pilot accidentally regenerated instead of retaining two existing
same-answer/different-binding pairs. Preserve the pilot and all old records;
use corrected inputs and fresh training in round47, before any validation.
"""
from data_access import ROOT
from round22_common import read,dump,sha
from round44_programs import tree,canonical,value
from round46_graph import features,graph,csr
from scipy import sparse
import numpy as np,shutil,datetime,copy
from collections import Counter

OLD=ROOT/'experiments/round46';OUT=ROOT/'experiments/round47'
ALIASES={
1633:(2,4,'Two hours of reading is a duration; two movies is a movie count.'),
2874:(1,7,'Ten work hours on the first day is a duration; ten dollars/hour is the pay rate.'),
3833:(4,2,'Twenty percent is the comparison increase, not twenty patients from the first hospital.'),
66:(0,3,'Four crayon colors is a category count, not the four-hour production interval.'),
2934:(0,4,'Five running time-outs is an event count, not five minutes per time-out.'),
947:(0,3,'Two buildings is the outer container count, not two apartments on the first floor.'),
2000:(0,3,'Two initial vampires is a population count, not the number of nights.'),
2255:(3,5,'Two teachers is an initial member count, not two parents leaving the group.'),
}

def main():
    OUT.mkdir();plan=read(OLD/'plan.json')
    for p,h in plan['source_hashes'].items():assert sha(ROOT/p)==h,p
    dump(OLD/'pilot_interruption.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reason='Source supervision audit found two existing equal-answer but semantically wrong bindings omitted by regeneration. Pilot stopped before full validation; partial checkpoints retained, not used for definitive comparison.',
        successor='round47',test_opened=False))
    for name in ('train_features.npz','train_targets.npz','reviewed_programs.json','preparation_result.json',
                 'held_source_review_before_reference.json','held_execution_checks.json','held_review_protocol.json','held_source_review_freeze.json'):
        shutil.copy2(OLD/name,OUT/name)
    pairs=read(OLD/'constraint_pairs.json')
    with np.load(OUT/'train_targets.npz') as z:foldmap=dict(zip(map(int,z['ids']),map(int,z['folds'])))
    def key(r):return (r['train_index'],repr(canonical(tree(r['positive']))),repr(canonical(tree(r['negative']))))
    seen={key(r) for r in pairs};restored=0
    for r in read(ROOT/'experiments/round44/constraint_pairs.json'):
        if key(r) in seen:continue
        r=dict(r,fold=foldmap[r['train_index']]);pairs.append(r);seen.add(key(r));restored+=1
    review={r['train_index']:r for r in read(OUT/'reviewed_programs.json')};aliaschecks=[]
    for i,(source,wrong,note) in ALIASES.items():
        r=review[i];p=tree(r['program']);qs=graph(r['question'])['quantities']
        assert qs[source]['value']==qs[wrong]['value'] and foldmap[i]!=2
        def replace(n):return ('N',wrong) if n[0]=='N' and n[1]==source else n if n[0] in ('N','L') else (n[0],replace(n[1]),replace(n[2]))
        neg=canonical(replace(p));assert neg!=canonical(p)
        assert value(p,qs)==value(neg,qs)
        changed=copy.deepcopy(qs);changed[wrong]['value']=str(float(changed[wrong]['value'])+1)
        assert value(p,changed)!=value(neg,changed)
        pairs.append(dict(train_index=i,fold=foldmap[i],question=r['question'],positive=p,negative=neg,action='object_binding',
            decision=dict(same_numeric_answer=True,correct_source=source,wrong_source=wrong),
            annotation=note+' Final-answer rewards remain identical; only reviewed process consistency distinguishes these programs.'))
        aliaschecks.append(dict(train_index=i,correct_source=dict(index=source,**qs[source]),wrong_source=dict(index=wrong,**qs[wrong]),
            adjudication=note,same_execution_value=True,counterfactual_value_separation=True))
    dump(OUT/'same_value_relation_review.json',aliaschecks);dump(OUT/'constraint_pairs.json',pairs)
    sparse.save_npz(OUT/'constraint_positive.npz',csr([features(r['question'],graph(r['question'])['quantities'],r['positive']) for r in pairs]))
    sparse.save_npz(OUT/'constraint_negative.npz',csr([features(r['question'],graph(r['question'])['quantities'],r['negative']) for r in pairs]))
    np.save(OUT/'constraint_parent_ids.npy',[r['train_index'] for r in pairs])
    sourcepaths=[ROOT/p for p in plan['source_hashes']]+[ROOT/'research'/f'round47_{n}.py' for n in ('prepare','run','evaluate')]
    plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sourcepaths},
        pilot_correction='Restore all existing relation supervision and add eight explicitly source-reviewed same-numeric-answer/wrong-object pairs. No architecture/proposal change after held source review.',
        held_source_review='24 additional parents used only for diagnostic evaluation, never process gradients, including final models.',
        no_validation_seen_for_new_method=True)
    dump(OUT/'plan.json',plan)
    result=read(OUT/'preparation_result.json');result.update(pairs=len(pairs),restored_existing_pairs=restored,new_same_value_pairs=len(ALIASES),
        actions=dict(Counter(r['action'] for r in pairs)),same_value_pairs=sum(value(r['positive'],graph(r['question'])['quantities'])==value(r['negative'],graph(r['question'])['quantities']) for r in pairs))
    dump(OUT/'preparation_result.json',result)
    dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='prepared_freeze.json'}))
    print(result,flush=True)

if __name__=='__main__':main()
