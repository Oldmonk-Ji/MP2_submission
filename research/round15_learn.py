"""Learn natural candidate bindings; compare matched question-only/no-question controls."""
from data_access import ROOT
import json,re,hashlib,time
from pathlib import Path
import numpy as np
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from round7_core import EXCLUDE, quantities

OUT=ROOT/'experiments/round15'
STOP=EXCLUDE|set('step cost total calculate find we so then let first second third given profit probability books book after before dollars dollar revenue answer question solution finally on saturday sunday friday black'.split())


def tokens(s):return list(re.finditer(r"[A-Za-z]+(?:['’]s)?",s))


def owners(r,context=True):
    text=(r['question']+'\n' if context and r['kind']=='candidate' else '')+r['text']
    return ['NIL']+sorted({m.group().lower() for m in re.finditer(r'\b[A-Z][a-z]+\b',text) if m.group().lower() not in STOP})


def scopes(r,a,context=True):
    snippets=[a['context']]
    if context:snippets.append(r['question'])
    out={'-'}
    for s in snippets:
        ts=tokens(s)
        for i in range(len(ts)):
            for size in range(1,6):
                if i+size>len(ts):break
                end=ts[i+size-1].end();start=ts[i].start();raw=s[start:end]
                if re.fullmatch(r"[A-Za-z'’]+(?:\s+[A-Za-z'’]+){0,4}",raw):out.add(raw.lower())
    return sorted(out)


def base(r,a,context=True):
    text=r['text'];s=a['context'];start=a['start'];end=a['end']
    names=set(owners(r,context));f={'bias':1.,'document_kind':r['kind'],'written':a['raw'].isalpha(),'dollar':text[:start].rstrip().endswith('$'),'slash':'/' in a['raw'],'relative_position':start/max(len(text),1)}
    def add(prefix,s):
        ws=['ENTITY' if m.group().lower() in names else m.group().lower() for m in tokens(s)]
        for w in ws:f[prefix+':'+w]=True
        for x,y in zip(ws,ws[1:]):f[prefix+':'+x+' '+y]=True
    add('sentence',s);add('left',text[max(a['context_start'],start-60):start]);add('right',text[end:min(a['context_end'],end+60)])
    if context:add('question',r['question'])
    f['step_number']=bool(re.search(r'\bStep\s*$',text[max(0,start-10):start],re.I))
    f['latex_fragment']=bool(re.search(r'\\frac\{\d*\}\{\d*$',text[max(0,start-20):start]))
    f['math_result']='=' in text[max(a['context_start'],start-50):start]
    return f


def link_features(r,a,option,field,context=True):
    f=base(r,a,context);f['nil']=option in ('NIL','-');s=r['text'];pos=a['start'];needle=option.lower()
    hits=[m for m in re.finditer(r'(?<!\w)'+re.escape(needle)+r'(?!\w)',s.lower())] if not f['nil'] else []
    f['in_question']=needle in r['question'].lower();f['in_sentence']=needle in a['context'].lower()
    if hits:
        h=min(hits,key=lambda m:abs(m.start()-pos));gap=pos-h.start()
        f.update(before=gap>=0,distance=min(abs(gap),400)/100.,same_sentence=a['context_start']<=h.start()<a['context_end'])
        f['adjacent_before']=0<=gap<25;f['adjacent_after']=-25<gap<0
        prefix=s[max(0,h.start()-24):h.start()].lower();suffix=s[h.end():h.end()+25].lower()
        for word in ('each','per','a','an','of','for','from','to','than','as'):
            f['before_'+word]=bool(re.search(r'\b'+word+r'\s*$',prefix))
        for word in ('cost','costs','sold','worth','at','is','was','will','earns','worked','has','had'):
            f['after_'+word]=bool(re.match(r'\W*'+word+r'\b',suffix))
        preceding=[(m.start(),name) for name in owners(r,context) if name!='NIL' for m in re.finditer(r'\b'+re.escape(name)+r'\b',s.lower()) if m.start()<pos]
        f['nearest_name']=bool(preceding) and max(preceding)[1]==needle
    else:f['absent_from_text']=True
    if field=='scope':
        f['phrase_length']=len(option.split());f['starts_function']=option.split()[0] in {'the','is','at','of','for','to','each','per','and'}
        for w in option.split():f['scope_word:'+w]=True
    return f


def fit(rows,path,context):
    path.mkdir(exist_ok=True,parents=True);models={};summary={}
    for field in ('valid','stage','owner','scope'):
        xx=[];yy=[];ww=[];unrepresentable=0
        for r in rows:
            for a,label in zip(r['anchors'],r['labels']):
                if field!='valid' and not label['valid']:continue
                truth=label['price_scope' if field=='scope' else field]
                if truth is None:continue
                if field in ('valid','stage'):
                    xx.append(base(r,a,context));yy.append(str(truth));ww.append(1)
                else:
                    opts=owners(r,context) if field=='owner' else scopes(r,a,context)
                    gold=truth.lower() if field=='scope' else truth
                    if gold not in opts:unrepresentable+=1;continue
                    for opt in opts:
                        xx.append(link_features(r,a,opt,field,context));yy.append(str(opt==gold));ww.append(1/len(opts))
        v=DictVectorizer();x=v.fit_transform(xx);m=LogisticRegression(C=1,max_iter=700,class_weight='balanced',random_state=15015)
        if len(set(yy))==1:m=DummyClassifier(strategy='most_frequent')
        weights=np.array(ww,dtype=float);weights*=len(weights)/weights.sum();m.fit(x,yy,sample_weight=weights)
        models[field]=(v,m);summary[field]=dict(examples=len(yy),features=x.shape[1],unrepresentable_gold=unrepresentable)
    joblib.dump(dict(context=context,models=models),path/'model.joblib')
    return summary


class Model:
    def __init__(self,path):self.obj=joblib.load(Path(path)/'model.joblib')
    def predict(self,r,a):
        ctx=self.obj['context'];out={};confidence=[]
        for field in ('valid','stage','owner','scope'):
            v,m=self.obj['models'][field]
            if field in ('valid','stage'):
                p=m.predict_proba(v.transform([base(r,a,ctx)]))[0];idx=int(p.argmax());out[field]=m.classes_[idx];confidence.append(float(p[idx]))
            else:
                opts=owners(r,ctx) if field=='owner' else scopes(r,a,ctx)
                p=m.predict_proba(v.transform([link_features(r,a,o,field,ctx) for o in opts]))[:,list(m.classes_).index('True')]
                idx=int(p.argmax());out[field]=opts[idx];confidence.append(float(p[idx]))
        out['valid']=out['valid']=='True';out['confidence']=min(confidence);return out


def rule(r,a):
    s=a['context'].lower();names=owners(r);before=[(m.start(),n) for n in names if n!='NIL' for m in re.finditer(r'\b'+re.escape(n)+r'\b',r['text'].lower()) if m.start()<a['start']]
    owner=max(before)[1] if before else (names[1] if len(names)==2 else 'NIL')
    stage='P' if re.search(r'\b(?:ago|last|previous)\b',s) else 'F' if re.search(r'\b(?:will|next|future)\b',s) else 'N'
    unit=re.search(r'\b(?:per|an)\s+([a-z]+)',s)
    return dict(valid=not bool(re.search(r'\bstep\s*$',r['text'][max(0,a['start']-10):a['start']],re.I)),owner=owner,stage=stage,scope=unit.group(1) if unit else '-',confidence=1.)


def evaluate(rows,predict):
    counts=dict(anchors=0,valid_gold=0,complete_gold=0,raw_joint_correct=0,emitted=0,emitted_correct=0,valid_correct=0,owner_correct=0,owner_n=0,stage_correct=0,stage_n=0,scope_correct=0,scope_n=0,price_correct=0,price_n=0);details=[]
    for r in rows:
        for i,(a,g) in enumerate(zip(r['anchors'],r['labels'])):
            p=predict(r,a);counts['anchors']+=1;counts['valid_correct']+=p['valid']==g['valid']
            counts['valid_gold']+=g['valid'];complete=g['valid'] and all(g[k] is not None for k in ('owner','stage','price_scope'))
            ok=p['valid'] and complete
            for key,pk in [('owner','owner'),('stage','stage'),('price_scope','scope')]:
                if g['valid'] and g[key] is not None:
                    right=str(g[key]).lower()==str(p[pk]).lower();counts[pk+'_n']+=1;counts[pk+'_correct']+=right;ok=ok and right
                    if key=='price_scope' and g[key]!='-':counts['price_n']+=1;counts['price_correct']+=right
            counts['complete_gold']+=complete;counts['raw_joint_correct']+=bool(ok)
            emitted=p['valid'] and p['confidence']>=.9
            # Unknown gold stays in coverage counts but cannot count as correct.
            counts['emitted']+=emitted;counts['emitted_correct']+=bool(emitted and ok)
            details.append(dict(record=r['id'],anchor=i,raw=a['raw'],gold=g,prediction=p,joint_correct=bool(ok)))
    counts['joint_accuracy']=counts['raw_joint_correct']/max(counts['complete_gold'],1)
    counts['correct_coverage']=counts['emitted_correct']/max(counts['complete_gold'],1)
    counts['emitted_precision']=counts['emitted_correct']/counts['emitted'] if counts['emitted'] else None
    counts['note']='Sampled natural anchors only. Not whole-trace relation recall or final-answer accuracy. Same-assistant annotation, not independent human gold.'
    return counts,details


def main():
    assert not (OUT/'frozen.json').exists()
    rows=json.loads((OUT/'fit_annotations.json').read_text(encoding='utf-8'));started=time.perf_counter();training={}
    for name,data,context in [('natural_candidates',rows,True),('question_only_training',[r for r in rows if r['kind']=='question'],True),('without_question_context',rows,False)]:
        training[name]=fit(data,OUT/name,context)
    frozen=dict(sources={n:hashlib.sha256((ROOT/'research'/n).read_bytes()).hexdigest() for n in ['round15_learn.py','round15_prepare.py','round15_annotations.py','round7_core.py','round4_quantities.py']},assets={name:hashlib.sha256((OUT/name/'model.joblib').read_bytes()).hexdigest() for name in training},plan_sha256=hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),labels_sha256=hashlib.sha256((OUT/'fit_annotations.json').read_bytes()).hexdigest(),stage='before audit text annotation and predictions')
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2));(OUT/'training.json').write_text(json.dumps(dict(systems=training,seconds=time.perf_counter()-started),indent=2));print(json.dumps(training))


if __name__=='__main__':main()
