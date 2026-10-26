"""Seal fresh TRAIN parents for a local rate diagnostic, no reference labels."""
from data_access import ROOT,CODE,read_jsonl
from round15_prepare import exclusions,anchors
from round14_prepare import family
import hashlib,json,re
OUT=ROOT/'experiments/round16'

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    assert not (OUT/'plan.json').exists()
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl')
    seen=exclusions()|{int(k) for k in json.loads((ROOT/'experiments/round15/plan.json').read_text())['groups']}
    families={family(bank[i]['question']) for i in seen}
    pool=[]
    for i,r in enumerate(bank):
        if i in seen or family(r['question']) in families:continue
        if not re.search(r'\$.*\b(?:each|per|apiece)\b',r['question'],re.I):continue
        if len(anchors(r['question']))<2:continue
        key=hashlib.sha256(('r16-rate-20261015:'+family(r['question'])).encode()).hexdigest()
        pool.append((key,i))
    records=[];ids=[];selected_families=set()
    for _,i in sorted(pool):
        r=bank[i];fam=family(r['question'])
        if fam in selected_families:continue
        selected_families.add(fam);ids.append(i)
        for idx,text in [(-1,r['question'])]+[(c['index'],c['text']) for c in r['candidates']]:
            records.append(dict(id=f'{i}:{idx}',train_index=i,kind='question' if idx==-1 else 'candidate',
                                candidate_index=idx,question=r['question'],text=text,anchors=anchors(text),
                                sha256=hashlib.sha256(text.encode()).hexdigest()))
        if len(ids)==8:break
    plan=dict(seed=16016,fit='round15/fit_annotations.json',development='round15/audit_annotations.json (already opened)',
              fresh_audit_parents=ids,excluded_parents=sorted(seen),
              sample='SHA-order eight fresh price-cue train parents, all four actual candidates; bounded numeric anchors, no whole-trace recall.',
              primary='local_pointer',controls=['round15','factor_only','without_question'],
              min_rate_mentions=20,min_exact_rate_precision=.95,min_rate_recall=.8,
              semantic_support_gate='Requires independently reviewed source/claim links and fair outer selection comparison; not satisfied by extraction alone.',
              no_validation=True,no_test=True,no_final_answer_references=True,
              audit_packet_sha256=hashlib.sha256(json.dumps(records,ensure_ascii=False,indent=2).encode()).hexdigest())
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'audit_packet.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(fresh_parents=len(ids),texts=len(records),anchors=sum(len(r['anchors']) for r in records))))

if __name__=='__main__':main()
