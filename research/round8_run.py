"""Frozen before fresh audit. All fitting and selection are train-only."""
from data_access import ROOT
import json,hashlib,copy,contextlib,time
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import round7_learn as learn
import round7_core as old
import round8_core as new
OUT=ROOT/'experiments/round8'

@contextlib.contextmanager
def backend(enhanced):
    names=('entities','qfeatures','owner_features','scope_features')
    previous={k:getattr(learn,k) for k in names}
    for k in names:setattr(learn,k,getattr(new if enhanced else old,k))
    try:yield
    finally:
        for k,v in previous.items():setattr(learn,k,v)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def train():
    rows=json.loads((OUT/'training_annotations.json').read_text(encoding='utf-8'))
    ids=np.random.default_rng(81015).permutation(len(rows));hold=set(ids[:20]);tr=[r for i,r in enumerate(rows) if i not in hold];ho=[r for i,r in enumerate(rows) if i in hold]
    (OUT/'internal_split.json').write_text(json.dumps({'fit':[r['index'] for r in tr],'tune':[r['index'] for r in ho]},indent=2))
    configs=[]
    for enhanced in (False,True):
        for c in (1,10):
            name=f'features{int(enhanced)}_C{c}'
            with backend(enhanced):
                learn.fit(tr,OUT/'internal'/name,c,True,True)
                metrics,_=learn.metric(ho,learn.Model(OUT/'internal'/name).predict)
            configs.append(dict(name=name,enhanced=enhanced,C=c,metrics=metrics))
            print(name,metrics['joint_binding_accuracy'],flush=True)
    chosen=max([c for c in configs if c['enhanced']],key=lambda c:c['metrics']['joint_binding_accuracy'])
    baseline=max([c for c in configs if not c['enhanced']],key=lambda c:c['metrics']['joint_binding_accuracy'])
    systems={}
    for name,config,mask in [('expanded_baseline',baseline,True),('relation_features',chosen,True),('without_dispute_mask',chosen,False)]:
        data=copy.deepcopy(rows)
        if not mask:
            for r in data:
                for i in r['disputed_mentions']:r['mask'][i]=r['original_mask'][i]
        with backend(config['enhanced']):learn.fit(data,OUT/name,config['C'],True,True)
        systems[name]=dict(enhanced=config['enhanced'],C=config['C'],dispute_mask=mask)
    a=json.loads((OUT/'ambiguity_training.json').read_text(encoding='utf-8'))
    model=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1,sublinear_tf=True),LogisticRegression(C=1,class_weight='balanced',max_iter=1000))
    model.fit([r['question'] for r in a],[r['ambiguity'] for r in a]);joblib.dump(model,OUT/'ambiguity.joblib')
    selection=dict(configs=configs,selected=chosen,systems=systems,ambiguity_threshold=.5,ambiguity_policy='Exploratory screening only, never automatically rewrites or discards a binding. Six positive train examples; evaluate fresh audit before deployment.')
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    sources=['round8_run.py','round8_core.py','round8_annotations.py','round7_learn.py','round7_core.py']
    frozen=dict(sources={f:digest(ROOT/'research'/f) for f in sources},assets={str(p.relative_to(OUT)):digest(p) for name in systems for p in (OUT/name).iterdir()},labels=digest(OUT/'training_annotations.json'),plan=digest(OUT/'plan.json'),ambiguity=digest(OUT/'ambiguity.joblib'))
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2));print('FROZEN',chosen['name'])

def audit():
    f=json.loads((OUT/'frozen.json').read_text());s=json.loads((OUT/'selection.json').read_text())
    for name,h in f['sources'].items():assert digest(ROOT/'research'/name)==h
    for name,h in f['assets'].items():assert digest(OUT/name)==h
    assert digest(OUT/'plan.json')==f['plan'] and digest(OUT/'training_annotations.json')==f['labels'] and digest(OUT/'ambiguity.joblib')==f['ambiguity']
    rows=json.loads((OUT/'audit_annotations.json').read_text(encoding='utf-8'))
    result={}
    for name,config in {**s['systems'],'round7_frozen':{'enhanced':False},'round6_rules':{'enhanced':False}}.items():
        with backend(config['enhanced']):
            predictor=learn.rule_predict if name=='round6_rules' else learn.Model(ROOT/'experiments/round7/learned' if name=='round7_frozen' else OUT/name).predict
            metric,detail=learn.metric(rows,predictor)
        result[name]=metric
        (OUT/f'audit_{name}.json').write_text(json.dumps(dict(metrics=metric,details=detail),ensure_ascii=False,indent=2),encoding='utf-8')
    probs=joblib.load(OUT/'ambiguity.joblib').predict_proba([r['question'] for r in rows])[:,1]
    true=np.array([r['ambiguity'] for r in rows]);pred=probs>=.5
    ambiguity=dict(positive_questions=int(sum(true)),clear_questions=int(sum(true==0)),recall=float(np.mean(pred[true==1])) if sum(true) else None,false_alarm_rate=float(np.mean(pred[true==0])) if sum(true==0) else None,details=[dict(index=r['index'],gold=r['ambiguity'],probability=float(p),flagged=bool(p>=.5)) for r,p in zip(rows,probs)])
    gate=json.loads((OUT/'plan.json').read_text())['gates'];m=result['relation_features']
    passed=all(m[k]>=gate[k] for k in ('joint_binding_accuracy','owner_accuracy','stage_accuracy','price_scope_accuracy')) and m['complete_questions']>=gate['min_complete_questions'] and ambiguity['recall'] is not None and ambiguity['recall']>=gate['min_ambiguity_recall'] and ambiguity['false_alarm_rate']<=gate['max_clear_false_alarm_rate']
    decision=dict(systems=result,ambiguity=ambiguity,extraction_gate_passed=bool(passed),validation_read=False,test_access=False,selected='round4/selected_bundle',limitation='12 same-assistant annotated fresh audit questions, not human-independent gold; reduced mention schema omits target and relation endpoints. No 99% claim.')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))

if __name__=='__main__':
    import sys
    train() if sys.argv[1]=='train' else audit()
