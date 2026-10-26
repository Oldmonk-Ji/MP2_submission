"""Train extraction audit and selection MUST finish before validation phase."""
from data_access import ROOT,CODE,load,read_jsonl
import argparse,json,time,hashlib
import numpy as np
from round5_graph import extract,candidate,features,NAMES
from round4_residual import fit_residual
from round4_gates import apply_gate
from features import metadata

OUT=ROOT/'experiments/round5'
GROUPS={'all':list(range(24)),'no_units':[0,1,2,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23],
        'no_target':[0,1,2,3,4,8,9,10,11,12,13,14,22,23],
        'no_entity_stage':[0,1,2,3,4,5,6,7,11,16,17,18,19,20,22,23],
        'units_only':[0,1,2,3,4,5,6,7,11,22,23]}
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def evaluate(choice,y,baseline,hard):
    c=y[np.arange(len(y)),choice];old=y[np.arange(len(y)),baseline]
    return dict(correct=int(c.sum()),n=len(y),accuracy=float(c.mean()),hard_correct=int(c[hard].sum()),hard_n=int(hard.sum()),fixed=int(((c==1)&(old==0)).sum()),lost=int(((c==0)&(old==1)).sum()),changes=int((choice!=baseline).sum()))
def choose(bank,s):return apply_gate(bank,s,'parse_and_repair',4)
def train():
    OUT.mkdir(parents=True,exist_ok=True)
    save('plan.json',dict(feature_names=NAMES,groups=GROUPS,l2=[1,.1,.01],selection='train internal holdout correct with zero residual fallback; tie prefer zero, then stronger regularization',quality='train reference consistency is a proxy, not semantic ground truth; random manual sample is development only',test_access=False))
    b,_,y=load('train');refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');x=features(b)
    np.savez(OUT/'train_graph_features.npz',features=x)
    audit=[];refx=[]
    for i,r in enumerate(refs):
        g=extract(r['question']);f,a=candidate(g,r['question'],r['answer'],reference=True);refx.append(f)
        if i in set(np.random.default_rng(51014).choice(len(b),12,replace=False)):
            audit.append(dict(index=i,question=r['question'],reference=r['answer'],graph=g,reference_binding=a,features=dict(zip(NAMES,f.astype(float)))))
    save('train_manual_audit.json',audit);refx=np.asarray(refx)
    quality=dict(train_n=len(b),reference_root_rate=float(refx[:,0].mean()),reference_target_dimension_known=int(refx[:,5].sum()),reference_target_dimension_match=int(refx[:,6].sum()),reference_target_dimension_mismatch=int(refx[:,7].sum()),reference_addition_conflicts=int((refx[:,4]>0).sum()),mean_binding_coverage=float(refx[:,23].mean()),reference_ambiguous_binding_questions=int((refx[:,1]>0).sum()))
    for label,mask in [('correct',y==1),('incorrect',y==0)]:
        z=x[mask];quality[label]=dict(n=len(z),root_rate=float(z[:,0].mean()),dimension_known=int(z[:,5].sum()),dimension_mismatch=int(z[:,7].sum()),addition_conflict=int((z[:,4]>0).sum()),mean_unused=float(z[:,8].mean()),mean_reuse=float(z[:,9].mean()))
    save('train_quality.json',quality);print('QUALITY',json.dumps(quality),flush=True)
    s=np.load(ROOT/'experiments/round3/stack_scores.npz')['meta_oof'];split=np.load(ROOT/'experiments/round4/internal_split.npz');fit=split['fit'];hold=split['holdout'];hb=[b[i] for i in hold];m=metadata(b)
    old=choose(hb,s[hold]);selections=[]
    for kind,cols in GROUPS.items():
        xx=x[:,:,cols];variants=[dict(l2=None,metrics=evaluate(old,y[hold],old,m['hard'][hold]))]
        for l2 in [1,.1,.01]:
            w=fit_residual(xx[fit],s[fit],y[fit],l2);ch=choose(hb,s[hold]+xx[hold]@w)
            variants.append(dict(l2=l2,metrics=evaluate(ch,y[hold],old,m['hard'][hold])))
        best=max(variants,key=lambda a:a['metrics']['correct']);w=np.zeros(len(cols)) if best['l2'] is None else fit_residual(xx,s,y,best['l2'])
        np.savez(OUT/f'{kind}_residual.npz',weight=w)
        selections.append(dict(kind=kind,variants=variants,best=best));print('SELECT',json.dumps(selections[-1]),flush=True)
    save('train_selection.json',selections)
    save('frozen_spec.json',dict(source_sha256=hashlib.sha256((ROOT/'research/round5_graph.py').read_bytes()).hexdigest(),validation_opened=False,limitations=['same-valued source bindings abstain','entities inferred by local lexical rules','time stages do not establish a complete event order','dimensional consistency is necessary only under correct extraction','reused prior OOF scores: not fully nested']))
def validate():
    spec=json.loads((OUT/'frozen_spec.json').read_text());assert spec['source_sha256']==hashlib.sha256((ROOT/'research/round5_graph.py').read_bytes()).hexdigest()
    assert (OUT/'manual_quality_review.json').exists(),'Review train extraction before validation'
    sels=json.loads((OUT/'train_selection.json').read_text());b,_,y=load('validation');m=metadata(b);s=np.load(ROOT/'experiments/round3/stack_scores.npz')['validation'];old=choose(b,s)
    tick=time.perf_counter();x=features(b);seconds=time.perf_counter()-tick;np.savez(OUT/'validation_graph_features.npz',features=x)
    records=[]
    for sel in sels:
        kind=sel['kind'];w=np.load(OUT/f'{kind}_residual.npz')['weight'];ch=choose(b,s+x[:,:,GROUPS[kind]]@w);rec=dict(kind=kind,train_selected_l2=sel['best']['l2'],metrics=evaluate(ch,y,old,m['hard']),changed=[dict(index=int(i),before=int(old[i]),after=int(ch[i]),correct=int(y[i,ch[i]])) for i in np.flatnonzero(ch!=old)])
        records.append(rec);print('VALIDATION',json.dumps(rec),flush=True)
    save('validation_comparison.json',dict(baseline=evaluate(old,y,old,m['hard']),records=records,feature_seconds=seconds))
    # Diagnosis only, never read by inference.
    errors=[]
    refs=read_jsonl(CODE/'data/gsm8k_validation.jsonl')
    for i in np.flatnonzero(y[np.arange(len(y)),old]==0):
        g=extract(b[i]['question']);errors.append(dict(index=int(i),question=b[i]['question'],reference=refs[i]['answer'],graph=g,selected=int(old[i]),candidates=[dict(correct=int(y[i,j]),features=dict(zip(NAMES,x[i,j].astype(float))),binding=candidate(g,b[i]['question'],c['text'])[1]) for j,c in enumerate(b[i]['candidates'])]))
    save('remaining_error_graphs.json',errors)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','validation']);a=p.parse_args();train() if a.phase=='train' else validate()
