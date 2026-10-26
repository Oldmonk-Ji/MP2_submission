from data_access import ROOT,CODE,read_jsonl
from round22_common import load,read,dump,sha
from round29_programs import parse,question_tokens
from round14_prepare import family
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import json,datetime

OUT=ROOT/'experiments/round29'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'models').exists();b,_,y,ids=load('train');refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=29029,
        route='Question-to-executable-program sequence supervision from official TRAIN references; no external teacher',
        training_ids=ids.tolist(),training_n=4743,validation_n=798,test_opened=False,
        diagnostic='Grouped 80/20 TRAIN split for reference-program token loss and exact structural scoring. Final model refits all 4743.',
        question_vocabulary=4500,width=64,decoder_width=96,epochs=[15,40,80],
        variants=['reference_only','reference_and_candidate_ranking'],alphas=[.1,.25,.5],
        parsing='v3: Preserve repeated numeric occurrences and repeated computed values; ambiguous binding becomes literal. Per-line prefer annotated arithmetic, otherwise parse unannotated numeric steps. No missing trace interpreted as contradiction.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [CODE/'data/gsm8k_train.jsonl',ROOT/'research/round29_programs.py']}))
    data=[]
    for k,(row,idx) in enumerate(zip(b,ids)):
        gold=parse(row['question'],refs[int(idx)]['answer'],True)
        candidates=[parse(row['question'],c['text']) for c in row['candidates']]
        data.append(dict(train_index=int(idx),question=row['question'],reference_text=refs[int(idx)]['answer'],reference=gold,
            candidates=candidates,labels=y[k].astype(int).tolist()))
    (OUT/'training_programs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
    groups=[family(r['question']) for r in b]
    fit,held=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=29029).split(np.arange(len(b)),groups=groups))
    assert not {groups[i] for i in fit}&{groups[i] for i in held}
    np.savez(OUT/'diagnostic_split.npz',fit=fit,held=held)
    def summary(rr):
        return dict(n=len(rr),root_found=sum(r['root_found'] for r in rr),informative=sum(r['informative'] for r in rr),
            arithmetic_consistent=sum(r['arithmetic_consistent'] for r in rr),clean=sum(r['clean'] for r in rr),
            ambiguous=sum(r['ambiguous_leaves']>0 for r in rr),mean_tokens=float(np.mean([len(r['tokens']) for r in rr])))
    summary_all=dict(reference=summary([r['reference'] for r in data]),candidates=summary([c for r in data for c in r['candidates']]),
        diagnostic_fit=len(fit),diagnostic_held=len(held),test_opened=False)
    dump(OUT/'extraction_summary.json',summary_all)
    rng=np.random.default_rng(29029);sample=rng.choice(fit,32,replace=False)
    dump(OUT/'parser_audit_sample.json',[dict(train_index=data[i]['train_index'],question=data[i]['question'],
        reference_text=data[i]['reference_text'],extraction=data[i]['reference']) for i in sample])
    print(summary_all,flush=True)


if __name__=='__main__':main()
