"""Train CRF labels and discriminative entity/scope links from local labels only."""
from data_access import ROOT
import argparse,json,time,hashlib,platform
from pathlib import Path
import numpy as np
import pycrfsuite
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from round7_core import quantities,entities,qfeatures,owner_features,scope_features

OUT=ROOT/'experiments/round7'
def supervised(rows):
    for r in rows:
        for i,m in enumerate(r['mask']):
            if m:yield r,i,r['labels'][i]
def fit(rows,path,c,context=True,factorized=False):
    path.mkdir(parents=True,exist_ok=True);trainer=pycrfsuite.Trainer(verbose=False)
    for r in rows:
        indices=[i for i,m in enumerate(r['mask']) if m]
        if not indices:continue
        trainer.append([qfeatures(r['question'],r['quantities'],i,context) for i in indices],[r['labels'][i]['role']+'|'+r['labels'][i]['stage'] for i in indices])
    trainer.set_params({'c1':.05,'c2':1/c,'max_iterations':150,'feature.possible_transitions':True});trainer.train(str(path/'labels.crfsuite'))
    if factorized:
        for field in ('role','stage'):
            tr=pycrfsuite.Trainer(verbose=False)
            for r in rows:
                ids=[i for i,m in enumerate(r['mask']) if m]
                if not ids:continue
                repeats=3 if field=='stage' and any(r['labels'][i]['stage']!='N' for i in ids) else 1
                for _ in range(repeats):tr.append([qfeatures(r['question'],r['quantities'],i,context) for i in ids],[r['labels'][i][field] for i in ids])
            tr.set_params({'c1':.05,'c2':1/c,'max_iterations':150,'feature.possible_transitions':True});tr.train(str(path/f'{field}.crfsuite'))
    diagnostics={}
    for kind in ('owner','scope'):
        xx=[];yy=[];ww=[];skipped=0
        for r,i,label in supervised(rows):
            q=r['question'];qs=r['quantities']
            if kind=='scope' and label['role']!='P':continue
            choices=list(entities(q)) if kind=='owner' else [-1]+[j for j in range(len(qs)) if j!=i]
            truth=label[kind]
            if truth not in choices:skipped+=1;continue
            for option in choices:
                xx.append(owner_features(q,qs,i,option,context) if kind=='owner' else scope_features(q,qs,i,option,context));yy.append(int(option==truth));ww.append(1/len(choices))
        v=DictVectorizer();x=v.fit_transform(xx);model=LogisticRegression(C=c,max_iter=800,class_weight='balanced',solver='lbfgs');model.fit(x,yy,sample_weight=ww);joblib.dump((v,model),path/f'{kind}.joblib')
        diagnostics[kind]=dict(pairs=len(yy),unrepresentable_gold_mentions=skipped,features=x.shape[1])
    (path/'config.json').write_text(json.dumps(dict(C=c,context=context,factorized=factorized,fit_diagnostics=diagnostics),indent=2));return diagnostics
class Model:
    def __init__(self,path):
        self.path=Path(path);self.cfg=json.loads((self.path/'config.json').read_text());self.tag=pycrfsuite.Tagger();self.tag.open(str(self.path/'labels.crfsuite'));self.link={k:joblib.load(self.path/f'{k}.joblib') for k in ('owner','scope')}
        self.heads={}
        if self.cfg.get('factorized'):
            for field in ('role','stage'):
                self.heads[field]=pycrfsuite.Tagger();self.heads[field].open(str(self.path/f'{field}.crfsuite'))
    def predict(self,q,nearest_owner=False):
        qs=quantities(q);context=self.cfg['context'];seq=[qfeatures(q,qs,i,context) for i in range(len(qs))]
        if not qs:return []
        tags=self.tag.tag(seq);result=[]
        if self.heads:tags=[r+'|'+s for r,s in zip(self.heads['role'].tag(seq),self.heads['stage'].tag(seq))]
        for i,tag in enumerate(tags):
            role,stage=tag.split('|');label=dict(role=role,stage=stage,owner='NIL',scope=-1)
            for kind in ('owner','scope'):
                if kind=='scope' and role!='P':continue
                opts=list(entities(q)) if kind=='owner' else [-1]+[j for j in range(len(qs)) if j!=i]
                feats=[owner_features(q,qs,i,o,context) if kind=='owner' else scope_features(q,qs,i,o,context) for o in opts];v,m=self.link[kind];z=m.decision_function(v.transform(feats));label[kind]=opts[int(np.argmax(z))]
            if nearest_owner:
                opts=[(p,n) for n,ps in entities(q).items() if n!='NIL' and "'s" not in n for p in ps if 0<=p<qs[i]['start']];label['owner']=max(opts)[1] if opts else 'NIL'
            result.append(label)
        return result
def metric(rows,predict):
    n=role=stage=owner=joint=0;prices=scope=named=namedok=temporal=temporalok=explicit=explicitok=complete=0;details=[]
    for r in rows:
        pred=predict(r['question']);qs=quantities(r['question']);assert len(pred)==len(r['labels']);allok=True;rec=[]
        for i,m in enumerate(r['mask']):
            if not m:continue
            g=r['labels'][i];p=pred[i];n+=1;role+=p['role']==g['role'];stage+=p['stage']==g['stage'];owner+=p['owner']==g['owner']
            scopeok=g['role']!='P' or (p['role']=='P' and p['scope']==g['scope']);ok=p['role']==g['role'] and p['stage']==g['stage'] and p['owner']==g['owner'] and scopeok;joint+=ok;allok &=ok
            if g['role']=='P':prices+=1;scope+=scopeok
            if g['role']=='P' and g['scope']>=0:explicit+=1;explicitok+=scopeok
            if g['owner']!='NIL':named+=1;namedok+=p['owner']==g['owner']
            if g['stage']!='N':temporal+=1;temporalok+=p['stage']==g['stage']
            rec.append(dict(mention=i,raw=qs[i]['raw'],gold=g,prediction=p,correct=bool(ok)))
        complete+=bool(allok and all(r['mask']));details.append(dict(index=r['index'],question=r['question'],all_gold_mentions_labelled=all(r['mask']),bindings=rec))
    return dict(questions=len(rows),mentions=n,role_accuracy=role/max(n,1),stage_accuracy=stage/max(n,1),owner_accuracy=owner/max(n,1),joint_binding_accuracy=joint/max(n,1),complete_questions=int(complete),price_scope_accuracy=scope/max(prices,1),price_mentions=prices,explicit_price_scope_accuracy=explicitok/max(explicit,1),explicit_price_mentions=explicit,named_owner_accuracy=namedok/max(named,1),named_owner_mentions=named,noncurrent_stage_accuracy=temporalok/max(temporal,1),noncurrent_mentions=temporal),details
def rule_predict(q):
    from round6_bindings import extract
    # Map the frozen rule outputs to the SAME reduced annotation schema.
    mapping={'count':'C','balance':'B','change':'D','unit_price':'P','total_price':'T','payment':'T','earn_rate':'P','work_rate':'C','discount':'R','age':'A','age_ratio':'R','age_difference':'D'}
    g=extract(q);qs=quantities(q);out=[]
    for i,x in enumerate(qs):
        matches=[]
        for f in g['facts']:
            v=x['value']/100 if q[x['end']:].lstrip().startswith('%') else x['value']
            if abs(abs(float(f['value']))-v)<1e-7:matches.append(f)
        if len(matches)!=1:out.append(dict(role='ABSTAIN',stage='ABSTAIN',owner='ABSTAIN',scope=-1));continue
        f=matches[0];tm=f['time'];stage=('F' if tm>0 else 'P' if tm<0 else 'N') if isinstance(tm,(int,float)) else {'initial':'I','final':'E','before_discount':'I','after_discount':'E'}.get(tm,'N');scope=-1
        if f['kind']=='unit_price':
            counts=[a for a in g['facts'] if a['kind']=='count' and a['owner']==f['owner'] and a['item']==f['item']];hits=[j for j,y in enumerate(qs) if len(counts)==1 and y['value']==counts[0]['value'] and i!=j]
            if len(hits)==1:scope=hits[0]
        out.append(dict(role=mapping.get(f['kind'],'O'),stage=stage,owner=f['owner'],scope=scope))
    return out
def train():
    rows=json.loads((OUT/'training_annotations.json').read_text(encoding='utf-8'));rng=np.random.default_rng(71014);ids=rng.permutation(len(rows));hold=set(ids[:14].tolist());tr=[r for i,r in enumerate(rows) if i not in hold];ho=[r for i,r in enumerate(rows) if i in hold]
    (OUT/'internal_split.json').write_text(json.dumps(dict(fit=[r['index'] for r in tr],tune=[r['index'] for r in ho]),indent=2));records=[];tick=time.perf_counter()
    for context,factorized in ((True,False),(False,False),(True,True)):
        for c in (.1,1,10):
            name=f'context{int(context)}_factor{int(factorized)}_C{c}';path=OUT/'internal'/name;fit(tr,path,c,context,factorized);m,_=metric(ho,Model(path).predict);records.append(dict(name=name,C=c,context=context,factorized=factorized,metrics=m));print('INTERNAL',json.dumps(records[-1]),flush=True)
    chosen=max(records,key=lambda r:r['metrics']['joint_binding_accuracy']);local=max([r for r in records if r['context']!=chosen['context']],key=lambda r:r['metrics']['joint_binding_accuracy'])
    for name,sel in [('learned',chosen),('context_ablation',local)]:fit(rows,OUT/name,sel['C'],sel['context'],sel['factorized'])
    record=dict(records=records,selected=chosen,local_ablation=local,seconds=time.perf_counter()-tick,training_questions=len(rows),python=platform.python_version(),crfsuite_version='0.9.12');(OUT/'training_selection.json').write_text(json.dumps(record,indent=2))
    files=['round7_core.py','round7_learn.py','round6_bindings.py','round5_age_graph.py','round4_quantities.py'];hashes={n:hashlib.sha256((ROOT/'research'/n).read_bytes()).hexdigest() for n in files};assets={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for name in ('learned','context_ablation') for p in (OUT/name).iterdir() if p.is_file()}
    (OUT/'frozen.json').write_text(json.dumps(dict(sources=hashes,assets=assets,plan_sha256=hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),training_labels_sha256=hashlib.sha256((OUT/'training_annotations.json').read_bytes()).hexdigest()),indent=2));print('FROZEN',chosen['name'],flush=True)
def audit():
    frozen=json.loads((OUT/'frozen.json').read_text())
    for n,h in frozen['sources'].items():assert hashlib.sha256((ROOT/'research'/n).read_bytes()).hexdigest()==h
    for n,h in frozen['assets'].items():assert hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h
    rows=json.loads((OUT/'audit_annotations.json').read_text(encoding='utf-8'));learned=Model(OUT/'learned');local=Model(OUT/'context_ablation');systems={'learned':learned.predict,'context_ablation':local.predict,'nearest_owner_ablation':lambda q:learned.predict(q,True),'round6_rules':rule_predict};result={}
    for name,predict in systems.items():
        tick=time.perf_counter();m,details=metric(rows,predict);m['seconds']=time.perf_counter()-tick;result[name]=m;(OUT/f'audit_{name}.json').write_text(json.dumps(dict(metrics=m,details=details),indent=2,ensure_ascii=False),encoding='utf-8');print('AUDIT',name,json.dumps(m),flush=True)
    gate=json.loads((OUT/'plan.json').read_text())['gates'];m=result['learned'];passed=m['complete_questions']>=gate['min_complete_audit_questions'] and all(m[k]>=gate['min_'+k] for k in ('joint_binding_accuracy','owner_accuracy','price_scope_accuracy','time_accuracy') if k!='time_accuracy') and m['stage_accuracy']>=gate['min_time_accuracy']
    (OUT/'audit_comparison.json').write_text(json.dumps(dict(systems=result,extraction_gate_passed=passed,validation_read=False,test_access=False,note='Reduced mention-binding task; does not yet establish full target/unit/equation extraction quality.'),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','audit']);a=p.parse_args();train() if a.phase=='train' else audit()
