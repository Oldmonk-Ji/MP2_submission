"""Learn local object/predicate/role proposals, independently of value equality.

Lexical object heads are only a first-stage alignment. Batch, time and relation
RHS identities remain unverified; this module cannot authorize answer changes.
"""
from data_access import ROOT
from round17_labels import OUT,norm
import re,json,hashlib,time
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

STOP=set('a an the and or of for to in at on with by as than from if how much many what which it its his her their he she they them we i you each every per is are was were be been will would can could should does do did has have had cost costs worth paid pay buy bought sell sold times twice once double triple total amount number dollars dollar money profit price first second third now then so that this those these into out also all other more fewer less enough wants want gets get making make made decided decided plans plan spent spend taking take chose charge charges gain averages remaining'.split())

def words(s):return list(re.finditer(r"[A-Za-z]+(?:['’]s)?",s))
def inventory(question):
    return sorted({norm(m.group()) for m in words(question) if len(m.group())>2 and norm(m.group()) not in STOP})

def role_features(r,a):
    left=r['text'][max(a['context_start'],a['start']-90):a['start']]
    right=r['text'][a['end']:min(a['context_end'],a['end']+90)]
    f={'bias':1.,'kind':r['kind'],'dollar':bool(re.search(r'\$\W*$',left)),
       'step':bool(re.search(r'\bstep\W*$',left,re.I)),
       'fraction_fragment':bool(re.search(r'\\frac\{\d*\}\{?\d*$',left)),
       'percent':bool(re.match(r'\W*%',right)),
       'equals_before':bool(re.search(r'=\W*$',left)),
       'math_command':'\\' in a['context']}
    for key,text in [('left',left),('right',right)]:
        ws=[norm(m.group()) for m in words(text)]
        ws=ws[-5:] if key=='left' else ws[:5]
        for w in ws:f[key+':'+w]=1.
        for x,y in zip(ws,ws[1:]):f[key+':'+x+' '+y]=1.
        if ws:f[key+':nearest']=ws[-1] if key=='left' else ws[0]
    for word in ('per','each','times','cost','profit','total'):
        f['context:'+word]=bool(re.search(r'\b'+word+r'\b',a['context'],re.I))
    return f

def object_features(r,a,obj):
    f={'bias':1.};hits=[m for m in words(r['text']) if norm(m.group())==obj]
    if not hits:return dict(bias=1.,absent=True)
    hit=min(hits,key=lambda m:abs(m.start()-a['start']));gap=hit.start()-a['start']
    left=hit.end()<=a['start'];bridge=r['text'][hit.end():a['start']] if left else r['text'][a['end']:hit.start()]
    before=r['text'][max(0,hit.start()-18):hit.start()].lower()
    after=r['text'][hit.end():hit.end()+25].lower()
    f.update(distance=min(abs(gap),400)/100,before=left,same_clause=a['context_start']<=hit.start()<a['context_end'],
             near=abs(gap)<35,intervening_number=bool(re.search(r'\d',bridge)),
             sentence_break=bool(re.search(r'\n|[.!?](?!\d)',bridge)),
             immediate_after=not left and len(bridge.strip(' *:,$'))<2,
             price_subject=left and abs(gap)<60 and bool(re.match(r'\W*(?:cost|costs|price|charge|earns|has)\b',after)),
             after_per=bool(re.search(r'\bper\W*$',before)),
             after_each=bool(re.search(r'\beach\W*$',before)))
    # No lexical object identity or numeric value is a learned feature.
    return f

def fit(rows):
    models={};counts={}
    for field in ('role','predicate','object'):
        xx=[];yy=[];ww=[];missing=0
        for r in rows:
            opts=inventory(r['question'])
            for a,g in zip(r['anchors'],r['labels']):
                if field=='role':gold=g['role'] if g['valid'] else 'invalid'
                else:gold=g[field] if g['valid'] else None
                if gold is None:continue
                if field=='object':
                    if gold not in opts:missing+=1;continue
                    for opt in opts:
                        xx.append(object_features(r,a,opt));yy.append(opt==gold);ww.append(1. if opt==gold else 1./max(len(opts)-1,1))
                else:xx.append(role_features(r,a));yy.append(gold);ww.append(1.)
        v=DictVectorizer();x=v.fit_transform(xx)
        m=LogisticRegression(C=1,max_iter=1000,random_state=17017,class_weight='balanced' if field!='object' else None)
        m.fit(x,yy,sample_weight=np.asarray(ww));models[field]=(v,m)
        counts[field]=dict(examples=len(yy),features=x.shape[1],missing_proposal_gold=missing)
    return dict(models=models,counts=counts)

class Model:
    def __init__(self,path):self.obj=joblib.load(path)
    def predict(self,r,a):
        f=role_features(r,a);out={}
        for field in ('role','predicate'):
            v,m=self.obj['models'][field];p=m.predict_proba(v.transform([f]))[0];i=int(p.argmax())
            out[field]=str(m.classes_[i]);out[field+'_confidence']=float(p[i])
        opts=inventory(r['question']);v,m=self.obj['models']['object']
        if not opts:out.update(object=None,object_confidence=0.,object_margin=0.)
        else:
            p=m.predict_proba(v.transform([object_features(r,a,o) for o in opts]))[:,list(m.classes_).index(True)]
            order=sorted(range(len(opts)),key=lambda j:(-p[j],opts[j]));i=order[0]
            out.update(object=opts[i],object_confidence=float(p[i]),object_margin=float(p[i]-p[order[1]]) if len(order)>1 else 1.)
        out['valid']=out['role']!='invalid'
        out['eligible']=out['valid'] and out['predicate']!='other' and out['object_confidence']>=.6 and out['object_margin']>=.1 and min(out['role_confidence'],out['predicate_confidence'])>=.6
        return out

def main():
    assert not (OUT/'frozen.json').exists()
    rows=json.loads((OUT/'fit_annotations.json').read_text(encoding='utf-8'));start=time.perf_counter();obj=fit(rows)
    path=OUT/'local_model.joblib';joblib.dump(obj,path)
    files=['round17_prepare.py','round17_labels.py','round17_model.py','round17_evaluate.py','round7_core.py','round4_quantities.py']
    freeze=dict(source_sha256={n:hashlib.sha256((ROOT/'research'/n).read_bytes()).hexdigest() for n in files},
                model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),asset_bytes=path.stat().st_size,
                fit_sha256=hashlib.sha256((OUT/'fit_annotations.json').read_bytes()).hexdigest(),
                audit_packet_sha256=hashlib.sha256((OUT/'audit_packet.json').read_bytes()).hexdigest(),
                plan_sha256=hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),
                stage='before audit text annotation/predictions',counts=obj['counts'],training_seconds=time.perf_counter()-start,
                inference_dependencies='model only plus source code',answer_overrides=False,validation_access=False,test_access=False)
    (OUT/'frozen.json').write_text(json.dumps(freeze,indent=2));print(json.dumps(freeze,indent=2))

if __name__=='__main__':main()
