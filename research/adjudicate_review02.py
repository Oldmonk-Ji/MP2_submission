"""Preserve second-batch human judgments and reference conventions separately."""
from data_access import ROOT,CODE,read_jsonl
import json,hashlib

def main():
    base=ROOT/'experiments/human_review'
    source=base/'returned/380ef7c7eb9ec6cf.json'
    record=json.loads(source.read_text(encoding='utf-8'))
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    findings={
      2549:dict(reference_implied_choice='B',comparison='target_definition_disagreement',human_interpretation='gross receipts; no expense subtraction',human_equation='200 * 3.55 * 30 = 21300',reference_equation='200 * 3.55 * 30 - 3000 = 18300',adjudication='The question says total income, while the reference explicitly computes net income. Both readings remain recorded; providing an expense does not by itself prove the requested target is net income.',clear_constraints=['daily milk volume = 200 gallons','unit price = 3.55 per gallon','monthly expense = 3000','June duration = 30 days'],disputed_relation='target = gross receipts versus net income',training_use='Keep explicit rates and expenses. Do not promote the target-definition choice to a hard semantic gold label.'),
      4310:dict(reference_implied_choice='A (closest option; not proof of exact batch identity)',comparison='batch_identity_disagreement',human_interpretation='initial bin is a distinct additional batch',human_equation='Let b be initial personal bin and s sister donation: b + 75 + s + 2*s = 108; personal total = b + 2*s = 22 + b/3, not uniquely determined.',reference_equation='75 + s + 2*s = 108; s = 11; personal total = 22',adjudication='The reference introduces no separate initial-bin variable. This is compatible with treating the opening as an overview, but does not establish that the bin and later closet batch are literally identical. Under the human reading, the initial bin quantity is missing. Preserve that underdetermination rather than inventing a quantity.',clear_constraints=['friends donation = 18 + 42 + 2 + 13 = 75','later personal closet batch = 2 * sister batch','donated total = 108'],disputed_relation='initial bin identity / overlap with later batches',training_use='Keep explicit batch quantities and the 2:1 relation. Mask the initial-bin merge and any uniquely determined personal-total target.')
    }
    rows=[]
    for response in record['responses']:
        i=response['train_index'];assert hashlib.sha256(refs[i]['question'].encode()).hexdigest()==response['question_sha256']
        rows.append(dict(**response,**findings[i],reference_solution=refs[i]['answer']))
    out=dict(source_record=str(source),split='train',summary=dict(reviewed=2,reference_disagreements=2,unresolved_semantic_relations=2),rows=rows,policy='Original human responses unchanged. Both questions remain ambiguity examples. No frozen round8 labels or models modified; no validation/test accessed. These labels describe interpretation, not a scored human exam.')
    (base/'review_pack02_adjudication.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out['summary']))

if __name__=='__main__':main()
