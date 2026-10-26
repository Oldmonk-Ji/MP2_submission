"""Gold denominators and comparisons refer to sampled lexical heads/predicates.

Explicitly separate raw head links, role agreement and reliable semantics.
Neither shared heads nor shared numbers authorize a selector override.
"""
from data_access import ROOT
from round17_model import OUT,Model,inventory,object_features
from itertools import combinations
import json,hashlib,time

def nearest(r,a):
    opts=inventory(r['question'])
    return min(opts,key=lambda o:(object_features(r,a,o).get('distance',5),o)) if opts else None

def main():
    frozen=json.loads((OUT/'frozen.json').read_text())
    for name,h in frozen['source_sha256'].items():assert hashlib.sha256((ROOT/'research'/name).read_bytes()).hexdigest()==h
    for name,k in [('local_model.joblib','model_sha256'),('fit_annotations.json','fit_sha256'),('audit_packet.json','audit_packet_sha256'),('plan.json','plan_sha256')]:
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==frozen[k]
    rows=json.loads((OUT/'audit_annotations.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((OUT/'audit_annotations.json').read_bytes()).hexdigest()==json.loads((OUT/'audit_label_seal.json').read_text())['sha256']
    model=Model(OUT/'local_model.joblib');start=time.perf_counter()
    preds={r['id']:[model.predict(r,a) for a in r['anchors']] for r in rows}
    roles={};objects={'correct':0,'n':0};invalid={'correct':0,'n':0};details=[]
    for r in rows:
        if r['kind']!='candidate':continue
        for i,(a,g,p) in enumerate(zip(r['anchors'],r['labels'],preds[r['id']])):
            invalid['n']+=1;invalid['correct']+=p['valid']==g['valid']
            if g['valid'] and g['role']:
                bucket=roles.setdefault(g['role'],dict(n=0,correct=0));bucket['n']+=1;bucket['correct']+=p['role']==g['role']
            if g['valid'] and g['object']:
                objects['n']+=1;objects['correct']+=p['object']==g['object']
            details.append(dict(record=r['id'],anchor=i,gold=g,prediction=p))
    groups={}
    for r in rows:groups.setdefault(r['train_index'],[]).append(r)
    counts={name:dict(gold=0,proposed=0,correct=0,role_correct_links=0,unequal_value_correct=0) for name in ('same_numeric_value','nearest_shared_object','learned_raw','learned_gated')}
    proposals=[];unassessable=0;pair_n=0
    for parent,recs in groups.items():
        cand=[r for r in recs if r['kind']=='candidate'];l,r=cand;pair_n+=1
        for i,(a,ga,pa) in enumerate(zip(l['anchors'],l['labels'],preds[l['id']])):
            for j,(b,gb,pb) in enumerate(zip(r['anchors'],r['labels'],preds[r['id']])):
                known=all(not g['valid'] or all(g[k] is not None for k in ('object','predicate','role')) for g in (ga,gb))
                gold=known and ga['valid'] and gb['valid'] and ga['predicate']!='other' and ga['object']==gb['object'] and ga['predicate']==gb['predicate']
                choices=dict(same_numeric_value=a['value']==b['value'],nearest_shared_object=nearest(l,a)==nearest(r,b),
                             learned_raw=pa['valid'] and pb['valid'] and pa['object']==pb['object'] and pa['predicate']==pb['predicate'] and pa['predicate']!='other',
                             learned_gated=pa['eligible'] and pb['eligible'] and pa['object']==pb['object'] and pa['predicate']==pb['predicate'])
                if not known:unassessable+=1
                for name,proposed in choices.items():
                    c=counts[name];c['gold']+=bool(gold)
                    # Unknown-gold proposals stay in the precision denominator,
                    # never become automatic true positives.
                    c['proposed']+=bool(proposed);c['correct']+=bool(proposed and gold)
                    c['role_correct_links']+=bool(proposed and gold and pa['role']==ga['role'] and pb['role']==gb['role'])
                    c['unequal_value_correct']+=bool(proposed and gold and a['value']!=b['value'])
                if choices['learned_raw']:
                    def claim(row,anchor,p):return dict(candidate_index=row['candidate_index'],object=p['object'],predicate=p['predicate'],role=p['role'],value=anchor['value'],start=anchor['context_start'],end=anchor['context_end'],text=anchor['context'])
                    proposals.append(dict(train_index=parent,left=claim(l,a,pa),right=claim(r,b,pb),
                        eligible=choices['learned_gated'],same_value=a['value']==b['value'],gold_head_predicate_match=bool(gold),
                        semantic_support='unverified',unresolved=['batch/time identity','multiplier RHS','final-target feasibility'],answer_override=False))
    for c in counts.values():
        c['precision']=c['correct']/c['proposed'] if c['proposed'] else None;c['recall']=c['correct']/c['gold'] if c['gold'] else None
    target_roles=('count','unit_price','multiplier');macro=sum(roles.get(k,{}).get('correct',0)/max(roles.get(k,{}).get('n',0),1) for k in target_roles)/3
    c=counts['learned_gated'];passed=c['gold']>=30 and all(roles.get(k,{}).get('n',0)>=8 for k in target_roles) and macro>=.8 and (c['precision'] or 0)>=.95 and (c['recall'] or 0)>=.8
    result=dict(candidate_pairs=pair_n,roles=roles,role_macro_accuracy=macro,objects=objects,validity=invalid,
                alignments=counts,unassessable_pairs=unassessable,diagnostic_seconds=time.perf_counter()-start,
                screening_gate_passed=passed,semantic_support_verified=False,validation_allowed=False,test_access=False,answer_overrides=0,
                caveat='Pair denominators are correlated anchors within eight parents; head+predicate identity is not full entity/event identity.')
    (OUT/'audit_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    (OUT/'audit_alignment_proposals.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
