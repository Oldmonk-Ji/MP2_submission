"""Semantic evaluation uses manually annotated TRAIN questions only."""
from data_access import ROOT
import argparse,json,hashlib
from collections import Counter
from round6_bindings import extract,solve_ledger,solve_time

OUT=ROOT/'experiments/round6'
def canonical(f):
    g=dict(f)
    if 'value' in g:g['value']=round(float(g['value']),7)
    if g.get('kind')=='age_ratio':g.setdefault('related_time',g['time'])
    # JSON distinguishes 2 from 2.0, but a numeric time coordinate does not.
    for key in ('time','related_time'):
        if isinstance(g.get(key),(int,float)):g[key]=round(float(g[key]),7)
    return json.dumps(g,sort_keys=True,ensure_ascii=True)
def run(phase):
    if phase=='audit':
        frozen=json.loads((OUT/'source_freeze.json').read_text())
        for name,h in frozen['sha256'].items():assert hashlib.sha256((ROOT/'research'/name).read_bytes()).hexdigest()==h
    rows=json.loads((OUT/f'{phase}_gold.json').read_text(encoding='utf-8'))['rows'];results=[]
    for r in rows:
        p=extract(r['question']);pf=[]
        for i,f in enumerate(p['facts']):
            f=dict(f)
            if f['kind']=='age_ratio':
                prov=next(a for a in p['provenance'] if a['fact_index']==i);f['related_time']=prov.get('right_time',f['time'])
            pf.append(f)
        gold=Counter(map(canonical,r['facts']));pred=Counter(map(canonical,pf));tp=sum((gold&pred).values());fp=sum((pred-gold).values());fn=sum((gold-pred).values())
        target_ok=p['target']==r['target'];sol=solve_ledger(p) or solve_time(p)
        results.append(dict(index=r['index'],category=r['category'],question=r['question'],tp=tp,fp=fp,fn=fn,target_correct=target_ok,target_emitted=p['target'] is not None,complete=fp==0 and fn==0 and target_ok,prediction=p,solution=sol,missing=[json.loads(s) for s in (gold-pred).elements()],wrong=[json.loads(s) for s in (pred-gold).elements()]))
    def metrics(rs):
        tp=sum(r['tp'] for r in rs);fp=sum(r['fp'] for r in rs);fn=sum(r['fn'] for r in rs);n=len(rs);targets=sum(r['target_emitted'] for r in rs);tc=sum(r['target_correct'] for r in rs)
        return dict(n=n,tp=tp,fp=fp,fn=fn,predicted_facts=tp+fp,fact_precision=tp/max(1,tp+fp),fact_recall=tp/max(1,tp+fn),target_emitted=targets,target_correct=tc,target_exact_all=tc/max(1,n),target_precision=tc/max(1,targets),question_coverage=sum(bool(r['prediction']['facts']) for r in rs)/max(1,n),complete_questions=sum(r['complete'] for r in rs),solver_coverage=sum(r['solution'] is not None for r in rs))
    overall=metrics(results);g=json.loads((OUT/'plan.json').read_text())['gates']
    passed=overall['predicted_facts']>=g['min_audited_predicted_facts'] and overall['fact_precision']>=g['min_fact_precision'] and overall['target_exact_all']>=g['min_target_exact'] and overall['question_coverage']>=g['min_question_coverage'] and overall['fp']==0
    report=dict(phase=phase,overall=overall,by_category={k:metrics([r for r in results if r['category']==k]) for k in ('time','price','total','owner')},extraction_gate_passed=passed,gate_note='False typed facts are conservatively treated as critical binding errors. Abstentions count as missed gold facts, not correct predictions.',results=results)
    (OUT/f'{phase}_evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    for r in results:print(json.dumps({k:r[k] for k in ('index','tp','fp','fn','target_correct','solution','wrong')},ensure_ascii=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','audit']);run(p.parse_args().phase)
