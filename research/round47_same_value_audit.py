"""Additional process-only challenge: unseen-parent equal-answer wrong bindings."""
from data_access import ROOT
from round22_common import read,dump,sha
from round44_programs import tree,canonical,value
from round46_graph import features,csr
import numpy as np,datetime

OUT=ROOT/'experiments/round47'
CASES={
1957:(2,1,'Wednesday reduction percentage is the second 50, not Tuesday distance in kilometers.'),
2341:(1,3,'Four cakes per weekday is the production rate, not the four-week observation duration.'),
1214:(2,0,'The denominator in one third comes from the fraction, not the number of weekly gym visits.'),
4601:(2,4,'Two pages less is a page decrement, not the label of night two.'),
}

def main():
    rows={r['train_index']:r for r in read(OUT/'held_source_review_before_reference.json')};pairs=[]
    for i,(source,wrong,note) in CASES.items():
        r=rows[i];p=tree(r['program']);qs=r['quantities'];assert qs[source]['value']==qs[wrong]['value']
        def replace(n):return ('N',wrong) if n[0]=='N' and n[1]==source else n if n[0] in ('N','L') else (n[0],replace(n[1]),replace(n[2]))
        neg=canonical(replace(p));assert canonical(p)!=neg and value(p,qs)==value(neg,qs)
        pairs.append(dict(train_index=i,question=r['question'],positive=p,negative=neg,quantities=qs,adjudication=note,
            positive_source=dict(index=source,**qs[source]),negative_source=dict(index=wrong,**qs[wrong])))
    dump(OUT/'held_same_value_cases.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),
        cases=pairs,role='Post-freeze diagnostic only, no model selection or training; four correlated challenge questions, not a population estimate.'))
    xp=csr([features(r['question'],r['quantities'],r['positive']) for r in pairs]);xn=csr([features(r['question'],r['quantities'],r['negative']) for r in pairs]);records=[]
    for arm in ('noaux','sft','exact','grpo'):
        margins=[]
        for seed in (46046,46047):
            path=OUT/'no_process_supervision'/f'model_s{seed}.npz' if arm=='noaux' else OUT/'diagnostic'/f'{arm}_s{seed}'/'model.npz'
            with np.load(path) as w:
                def score(x):return (np.maximum(np.maximum(x@w['w0']+w['b0'],0)@w['w1']+w['b1'],0)@w['w2']+w['b2']).reshape(-1)
                margins.append(score(xp)-score(xn))
        m=np.mean(margins,axis=0);records.append(dict(arm=arm,correct=int((m>0).sum()),n=len(pairs),margins=m.tolist()))
    dump(OUT/'held_same_value_result.json',dict(records=records,not_used_for_tuning=True,test_opened=False));print(records,flush=True)

if __name__=='__main__':main()
