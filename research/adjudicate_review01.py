"""Record human/reference disagreement without promoting it to training gold."""
from data_access import ROOT,CODE,read_jsonl
import json,hashlib

def main():
    base=ROOT/'experiments/human_review'
    received=base/'returned/aba8c4d34c870561.json'
    human=json.loads(received.read_text(encoding='utf-8'))
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    decisions={
        1203:('A','agreement','Sydney=3*Mason; father=Sydney+6','Human and reference agree on the intended interpretation; the phrase three times younger remains imprecise.'),
        1922:('A','ambiguous_wording','initial=4; current=4+6-2','Human identified a tense inconsistency: has suggests current stock but last summer and 6 more suggest a past starting quantity. Reference assumes initial stock. Preserve the objection; do not learn has => initial as a general rule.'),
        2344:('B','agreement','camera_cost=2*110; frame_cost=3*120; payment=(camera_cost+frame_cost)*0.95','Human and reference both interpret 110 as a per-camera price.'),
        3711:('A','reference_disagreement','week1=10+5; week2=week3=2*week1; total=75','Human uses the household-only 10 kg baseline for later weeks, giving total 55. Reference includes the neighbor quantity in the baseline, giving 75. Wording does not explicitly settle whether neighbor demand persists. Exclude disputed baseline binding from hard semantic supervision pending adjudication.'),
        1201:('A','agreement','reserved=7; each_sister=(20-7-3)/2','Human and reference both use seven days.'),
        3790:('A','reference_disagreement','Puffy=3*Juniper; Scruffy=2*Puffy; Buffy=(Juniper+Puffy+Scruffy)/3','Human interprets three times more as an increment of three times, hence total four times. Reference uses total three times. Preserve both readings; do not turn this into a universal multiplier rule.'),
        1585:('A','reference_disagreement','total=30*5+4*15','Human interprets shaping price as inclusive (26*5+4*15=190); reference adds shaping to base trim (210). The question lacks an explicit additional charge statement. Preserve the price-scope ambiguity instead of declaring the human wrong.')}
    rows=[]
    for r in human['responses']:
        i=r['train_index'];ref_choice,status,equations,reason=decisions[i]
        assert hashlib.sha256(refs[i]['question'].encode()).hexdigest()==r['question_sha256']
        rows.append(dict(**r,reference_implied_choice=ref_choice,comparison=status,reference_equations=equations,reference_solution=refs[i]['answer'],adjudication_reason=reason,training_use='candidate supervision after review' if status=='agreement' else 'exclude disputed relation from hard semantic labels; retain ambiguity evidence'))
    result=dict(source_record=str(received),split='train',summary=dict(reviewed=7,agreement=3,ambiguous_wording=1,reference_disagreement=3),policy='Human responses remain unchanged. Reference-implied relations are dataset conventions, not proof of uniquely correct language interpretation. No training labels or model weights changed. No validation/test or sealed audit accessed.',rows=rows)
    (base/'review_pack01_adjudication.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result['summary']))

if __name__=='__main__':main()
