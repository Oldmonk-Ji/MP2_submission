"""Report extraction separately from unverified candidate contrast proposals."""
from data_access import ROOT,parse_answer
from round16_rates import OUT,RateModel,evaluate
from round15_learn import Model
from itertools import combinations
import json,hashlib

def span(text,start,end):
    return dict(start=start,end=end,text=text[start:end])

def contrasts(rows,model):
    groups={};result=[];pairs=0;question_with_proposals=set()
    for r in rows:groups.setdefault(r['train_index'],[]).append(r)
    for index,records in groups.items():
        q=next(r for r in records if r['kind']=='question')
        cand=[r for r in records if r['kind']=='candidate']
        predictions={r['id']:[model.predict(r,a) for a in r['anchors']] for r in records}
        for left,right in combinations(cand,2):
            la,ra=parse_answer(left['text']),parse_answer(right['text'])
            if la is None or ra is None or la==ra:continue
            pairs+=1;emitted=0
            for i,a in enumerate(left['anchors']):
                lp=predictions[left['id']][i]
                for j,b in enumerate(right['anchors']):
                    rp=predictions[right['id']][j]
                    if a['value']!=b['value'] or lp['scope']==rp['scope']:continue
                    if lp['scope']=='-' or rp['scope']=='-':continue
                    # Repeated numeric values are not resolved by nearest name.
                    def multiplicity(r):return sum(z['value']==a['value'] for z in r['anchors'])
                    qa=[(k,z) for k,z in enumerate(q['anchors']) if z['value']==a['value']]
                    unresolved=len(qa)!=1 or multiplicity(left)!=1 or multiplicity(right)!=1
                    evidence=[span(q['text'],z['context_start'],z['context_end']) for _,z in qa]
                    qp=predictions[q['id']][qa[0][0]] if len(qa)==1 else None
                    def claim(r,z,p):return dict(value=z['value'],scope=p['scope'],quantity_span=span(r['text'],z['start'],z['end']),source_span=span(r['text'],z['context_start'],z['context_end']))
                    result.append(dict(question_id=f'train:{index}',candidate_pair=[left['candidate_index'],right['candidate_index']],difference_type='rate_scope',
                        question_evidence=evidence,left_claim=claim(left,a,lp),right_claim=claim(right,b,rp),
                        question_scope_proposal=qp['scope'] if qp else None,
                        unresolved_identity=unresolved,semantic_label='insufficient',
                        reason='Numeric match and learned pointers propose an alignment; identity and source support are NOT independently verified.',
                        answer_override=False,provenance='unmodified train texts; model proposals, not semantic gold'))
                    question_with_proposals.add(index);emitted+=1
                    if emitted>=2:break
                if emitted>=2:break
    return result,dict(answer_conflict_pairs=pairs,proposed_records=len(result),
                      proposed_pairs=len({(r['question_id'],tuple(r['candidate_pair'])) for r in result}),
                      proposed_questions=len(question_with_proposals),verified_support_records=0,
                      note='Proposal coverage is not verified alignment coverage; repeated values may be unresolved.')

def main():
    frozen=json.loads((OUT/'frozen.json').read_text())
    for n,h in frozen['sources'].items():assert hashlib.sha256((ROOT/'research'/n).read_bytes()).hexdigest()==h
    for n,a in frozen['assets'].items():assert hashlib.sha256((OUT/(n+'.joblib')).read_bytes()).hexdigest()==a['sha256']
    assert hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest()==frozen['plan_sha256']
    assert hashlib.sha256((ROOT/'experiments/round15/fit_annotations.json').read_bytes()).hexdigest()==frozen['fit_sha256']
    models={n:RateModel(OUT/(n+'.joblib')).predict for n in frozen['assets']}
    models['round15']=Model(ROOT/'experiments/round15/natural_candidates').predict
    all_results={}
    for split,path in [('development',ROOT/'experiments/round15/audit_annotations.json'),('fresh_audit',OUT/'audit_annotations.json')]:
        if not path.exists():continue
        rows=json.loads(path.read_text(encoding='utf-8'));metrics={}
        for name,predict in models.items():
            metrics[name],details=evaluate(rows,predict)
            (OUT/f'{split}_{name}.json').write_text(json.dumps(dict(metrics=metrics[name],details=details),indent=2),encoding='utf-8')
        records,coverage=contrasts(rows,RateModel(OUT/'local_pointer.joblib'))
        (OUT/f'{split}_contrasts.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        all_results[split]=dict(metrics=metrics,contrast_coverage=coverage)
    fresh=all_results.get('fresh_audit',{}).get('metrics',{}).get('local_pointer',{})
    passed=bool(fresh) and fresh['rate_mentions']>=20 and (fresh['exact_rate_precision'] or 0)>=.95 and (fresh['rate_recall'] or 0)>=.8
    (OUT/'result.json').write_text(json.dumps(dict(results=all_results,rate_extraction_gate_passed=passed,
        semantic_support_gate_passed=False,validation_allowed=False,answer_overrides=0,test_access=False),indent=2))
    print(json.dumps(all_results,indent=2))

if __name__=='__main__':main()
