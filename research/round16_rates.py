"""Diagnostic local rate binding. No selection overrides or answer supervision.

Separate rate existence from its denominator pointer; each positive rate
mention gives equal total weight to its positive and negative pointer choices.
Question evidence is an ablation, not presumed correct semantic support.
"""
from data_access import ROOT
from round15_learn import base, link_features, scopes, tokens
from round7_core import quantities
from fractions import Fraction
from pathlib import Path
import re, json, hashlib, time
import numpy as np
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

OUT = ROOT / 'experiments/round16'


def pointer_features(r, a, option, context=True, interactions=True):
    f = link_features(r, a, option, 'scope', context)
    if not interactions:
        return f
    # Anchor-dependent features: a shared sentence bag cannot distinguish
    # which of several prices binds to which of several noun phrases.
    def window(text, anchor, prefix):
        hits = list(re.finditer(r'(?<!\w)' + re.escape(option) + r'(?!\w)', text.lower()))
        if not hits:
            f[prefix + ':absent'] = True
            return
        h = min(hits, key=lambda m: abs(m.start()-anchor['start']))
        left = h.end() <= anchor['start']
        bridge = text[h.end():anchor['start']] if left else text[anchor['end']:h.start()]
        near = abs(h.start()-anchor['start']) < 85
        f[prefix + ':near'] = near
        f[prefix + ':before'] = left
        f[prefix + ':gap'] = min(len(bridge), 200) / 100
        f[prefix + ':intervening_number'] = bool(re.search(r'\d', bridge))
        f[prefix + ':sentence_break'] = bool(re.search(r'\n|[.!?](?!\d)', bridge))
        preceding = text[max(0,h.start()-15):h.start()].lower()
        following = text[h.end():h.end()+25].lower()
        for cue in ('per', 'each', 'every', 'a', 'an', 'of', 'for'):
            value = bool(re.search(r'\b'+cue+r'\s*$', preceding))
            f[prefix+':near_'+cue] = value and near
            f[prefix+':no_number_'+cue] = value and not re.search(r'\d', bridge)
        for cue in ('cost', 'costs', 'is', 'are', 'earns', 'sold'):
            f[prefix+':near_verb_'+cue] = near and left and bool(re.match(r'\W*'+cue+r'\b', following))
        f[prefix+':immediate_per_after_value'] = not left and bool(re.fullmatch(r'\s*(?:per|an?|/)\s*',bridge,re.I))
        f[prefix+':each_before_value'] = left and near and bool(re.search(r'\b(?:each|every)\s*$',preceding))
    window(r['text'], a, 'claim')
    if context:
        # A repeated value is not sufficient evidence for identity; expose
        # ambiguity rather than silently selecting one occurrence.
        matches = [q for q in quantities(r['question']) if q['value'] == a['value']]
        f['question_value_unique'] = len(matches)==1
        if len(matches)==1:
            window(r['question'], matches[0], 'source')
    return f


def train(rows, context=True, interactions=True):
    binary_x=[]; binary_y=[]; pointer_x=[]; pointer_y=[]; pointer_w=[]
    missing=0; rates=0
    for r in rows:
        for a,g in zip(r['anchors'],r['labels']):
            if not g['valid'] or g['price_scope'] is None:
                continue
            yes=g['price_scope']!='-'
            binary_x.append(base(r,a,context)); binary_y.append(yes)
            if not yes:
                continue
            rates+=1; opts=[x for x in scopes(r,a,context) if x!='-'];gold=g['price_scope'].lower()
            if gold not in opts:
                missing+=1;continue
            for opt in opts:
                pointer_x.append(pointer_features(r,a,opt,context,interactions))
                pointer_y.append(opt==gold)
                pointer_w.append(.5 if opt==gold else .5/max(len(opts)-1,1))
    models={}
    for name,xx,yy,weights in [('presence',binary_x,binary_y,None),('pointer',pointer_x,pointer_y,pointer_w)]:
        v=DictVectorizer();x=v.fit_transform(xx)
        m=LogisticRegression(C=1, max_iter=1000,random_state=16016,
                             class_weight='balanced' if name=='presence' else None)
        # Pointer objective is a sum over equally weighted mentions, not an
        # ever larger loss when the noun-phrase proposal inventory grows.
        w=np.asarray(weights)*2 if weights is not None else None
        m.fit(x,yy,sample_weight=w);models[name]=(v,m)
    return dict(models=models,context=context,interactions=interactions,
                training=dict(binary_mentions=len(binary_y),rate_mentions=rates,
                              pointer_pairs=len(pointer_y),unrepresentable_gold=missing))


class RateModel:
    def __init__(self, path):
        self.obj=joblib.load(path)
    def predict(self,r,a):
        ctx=self.obj['context'];v,m=self.obj['models']['presence']
        p=float(m.predict_proba(v.transform([base(r,a,ctx)]))[0,list(m.classes_).index(True)])
        opts=[o for o in scopes(r,a,ctx) if o!='-']
        if not opts:
            return dict(scope='-',rate_probability=p,pointer_score=None,runner_margin=None)
        v,m=self.obj['models']['pointer']
        score=m.decision_function(v.transform([pointer_features(r,a,o,ctx,self.obj['interactions']) for o in opts]))
        order=sorted(range(len(opts)),key=lambda j:(-float(score[j]),opts[j]))
        best=order[0]
        return dict(scope=opts[best] if p>=.5 else '-',rate_probability=p,
                    pointer_proposal=opts[best],pointer_score=float(score[best]),
                    runner_margin=float(score[best]-score[order[1]]) if len(order)>1 else None)


def evaluate(rows,predict):
    n=correct=rate_n=rate_correct=emitted=fp=proposal=invalid=0;details=[]
    for r in rows:
        if r['kind']!='candidate':continue
        for i,(a,g) in enumerate(zip(r['anchors'],r['labels'])):
            p=predict(r,a)
            if not g['valid']:
                invalid+=p['scope']!='-';continue
            if g['price_scope'] is None:continue
            gold=g['price_scope'].lower();n+=1;correct+=p['scope']==gold
            rate_n+=gold!='-';rate_correct+=gold!='-' and p['scope']==gold
            emitted+=p['scope']!='-';fp+=gold=='-' and p['scope']!='-'
            proposal+=gold!='-' and p.get('pointer_proposal')==gold
            details.append(dict(record=r['id'],anchor=i,gold=gold,prediction=p))
    return dict(valid_mentions=n,scope_correct=correct,rate_mentions=rate_n,rate_correct=rate_correct,
                emitted=emitted,false_positive_nonrates=fp,invalid_emissions=invalid,
                exact_rate_precision=rate_correct/emitted if emitted else None,
                rate_recall=rate_correct/rate_n if rate_n else None,
                oracle_presence_pointer_correct=proposal),details


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'frozen.json').exists(),'Preserve frozen run; use a new experiment for changes.'
    rows=json.loads((ROOT/'experiments/round15/fit_annotations.json').read_text(encoding='utf-8'))
    start=time.perf_counter();training={};assets={}
    for name,context,interactions in [('factor_only',True,False),('local_pointer',True,True),('without_question',False,True)]:
        obj=train(rows,context,interactions);path=OUT/(name+'.joblib');joblib.dump(obj,path)
        training[name]=obj['training'];assets[name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT/'training.json').write_text(json.dumps(dict(systems=training,seconds=time.perf_counter()-start),indent=2))
    sources=['round16_rates.py','round16_run.py','round16_prepare.py','round15_learn.py','round15_prepare.py','round7_core.py','round4_quantities.py']
    freeze=dict(sources={n:hashlib.sha256((ROOT/'research'/n).read_bytes()).hexdigest() for n in sources},assets=assets,
                fit_sha256=hashlib.sha256((ROOT/'experiments/round15/fit_annotations.json').read_bytes()).hexdigest(),
                plan_sha256=hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),
                label_provenance='Existing assistant annotations; diagnostic only, not independent human truth.',
                gate=dict(min_fresh_rate_mentions=20,min_exact_rate_precision=.95,min_rate_recall=.8),
                answer_override_enabled=False,validation_access=False,test_access=False)
    (OUT/'frozen.json').write_text(json.dumps(freeze,indent=2));print(json.dumps(training))


if __name__=='__main__':main()
