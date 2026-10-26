from data_access import ROOT
import json,hashlib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,classification_report

OUT=ROOT/'experiments/round14'
def main():
    data=json.loads((OUT/'event_annotations.json').read_text(encoding='utf-8'))['records']
    plan=json.loads((OUT/'plan.json').read_text())
    x=[];y=[];groups=[]
    for r in data:
        assert r['split']=='train' and r['index'] not in plan['sealed_audit']
        assert hashlib.sha256(r['question'].encode()).hexdigest()==r['question_sha256']
        assert sum(a['type']=='target' for a in r['atoms'])==1
        for a in r['atoms']:
            e=a['evidence'];args=a['arguments']
            assert r['question'][e['start']:e['end']]==e['text']
            if a['type'] in ('transfer','sale','payment'):
                assert args['source']!=args['recipient'] and 'event_time' in args
            if a['type'] in ('state','target'):assert 'state_time' in args
            if a['status'] in ('ambiguous','pending_review'):assert not a['supervise']
            if a['supervise']:
                x.append(e['text']);y.append(a['type']);groups.append(r['index'])
    x=np.array(x);y=np.array(y);groups=np.array(groups);pred=np.empty(len(y),dtype=object)
    folds=[]
    for train,hold in GroupKFold(5).split(x,y,groups):
        assert not set(groups[train])&set(groups[hold])
        model=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(2,5)),LogisticRegression(C=4,max_iter=1000,class_weight='balanced'))
        model.fit(x[train],y[train]);pred[hold]=model.predict(x[hold])
        folds.append({'train_questions':sorted(set(map(int,groups[train]))),'hold_questions':sorted(set(map(int,groups[hold])))})
    result={'questions':len(data),'atoms':sum(len(r['atoms']) for r in data),'supervised_atoms':len(y),'checks_passed':True,'diagnostic':'5-fold question-disjoint classification on manually supplied evidence spans; NOT end-to-end extraction','accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),'by_class':classification_report(y,pred,output_dict=True,zero_division=0),'folds':folds,'full_extractor_gate':'NOT PASSED / not evaluated','validation_used':False,'test_used':False,'sealed_audit_read':False}
    (OUT/'quality_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('by_class','folds')}))
if __name__=='__main__':main()
