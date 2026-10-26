from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump,sha
import numpy as np

OUT=ROOT/'experiments/round39'
# Adjudicated before reading reference answers. Amount/count/price edges are
# correct; two complete proofs fail specifically at their target relation.
VERDICTS={
453:(False,'average price per movie is total cost / total movie count, not total cost','14'),
1657:(True,'three purchases contribute to all spending','90'),
1890:(True,'four purchase batches with local unit prices contribute to all spending','33'),
2729:(True,'one computer game plus three ticket purchases; entire entertainment spending','102'),
3232:(False,'target is Coach A cost minus Coach B cost, not combined cost','237'),
4173:(True,'all three purchases; repeated count 5 belongs to separate object occurrences','160')}

def main():
    rows=read(OUT/'natural_relation_audit_packet.json');reviews=[]
    for r in rows:
        good,note,ans=VERDICTS[r['train_index']];o=r['extraction'];q=r['question']
        for f in o['facts']:
            e=f['evidence'];assert q[e['start']:e['end']]==e['text']
        reviews.append(dict(train_index=r['train_index'],fold=r['fold'],fact_edges_supported=True,target_supported=good,
            complete_proof_supported=good,adjudication=note,independently_derived_target=ans,
            provenance='assistant source-grounded review before reference comparison; not independent human gold'))
    dump(OUT/'natural_relation_adjudications.json',reviews)
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    for r in reviews:assert r['independently_derived_target']==parse_answer(refs[r['train_index']]['answer'])
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:ids=z['ids'];choices=z['choices'];scores=z['scores'];folds=z['folds']
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl');extractions={r['train_index']:r['extraction'] for r in read(OUT/'train_extractions.json')}
    changes=[]
    for k,i in enumerate(ids):
        out=extractions[int(i)];ans=out['answer']
        if ans is None:continue
        eligible=[j for j,c in enumerate(bank[int(i)]['candidates']) if parse_answer(c['text'])==ans]
        if not eligible:continue
        choice=max(eligible,key=lambda j:scores[k,j]);base=int(choices[k]);gold=parse_answer(refs[int(i)]['answer'])
        if parse_answer(bank[int(i)]['candidates'][choice]['text'])!=parse_answer(bank[int(i)]['candidates'][base]['text']):
            changes.append(dict(train_index=int(i),fold=int(folds[k]),base_correct=parse_answer(bank[int(i)]['candidates'][base]['text'])==gold,new_correct=ans==gold))
    dump(OUT/'train_effect_diagnostic.json',dict(changes=changes,test_opened=False,not_deployed=True))
    report=dict(proved_held_questions=len(rows),correct_complete_proofs=4,wrong_complete_proofs=2,
        complete_proof_precision=4/6,fact_edge_precision_observed=1.0,
        decision='Rejected as a proof authority. No official validation evaluation of v39. Target supervision was incomplete.',
        held_numeric_changes=[r for r in changes if r['fold']!=0],test_opened=False)
    dump(OUT/'result.json',report);print(report)

if __name__=='__main__':main()
