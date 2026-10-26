"""Seal TRAIN-only role/object annotation groups before reading audit texts."""
from data_access import ROOT,CODE,read_jsonl,parse_answer
from round15_prepare import exclusions
from round14_prepare import family
from round7_core import quantities
import json,hashlib,re
OUT=ROOT/'experiments/round17'

def anchors(text):
    out=[]
    for a in quantities(text):
        start=max(text.rfind('\n',0,a['start']),text.rfind('. ',0,a['start']))+1
        ends=[x for x in (text.find('\n',a['end']),text.find('. ',a['end'])) if x>=0]
        end=min(ends) if ends else len(text);context=text[start:end]
        if '####' in context or '\\boxed' in context:continue
        if len(re.findall(r'[A-Za-z]{2,}',context))<3:continue
        if re.match(r'\s*(?:\*\*)?'+re.escape(a['raw'])+r'[.)]\s',text[a['start']:]) and a['start']-start<5:continue
        out.append(dict(**a,context=context,context_start=start,context_end=end))
    # Fixed bounded sample; keep initial/final anchors and interior explicit
    # dollar/times/each mentions. This is not whole-trace extraction recall.
    if len(out)>8:
        selected=out[:2]+out[-2:]
        ranked=sorted(out,key=lambda a:(not bool(re.search(r'\$\W*$',text[max(0,a['start']-5):a['start']]) or re.match(r'\W*(?:times|each|per)\b',text[a['end']:],re.I)),a['start']))
        for a in ranked:
            if a not in selected:selected.append(a)
            if len(selected)==8:break
        out=sorted(selected,key=lambda a:a['start'])
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'plan.json').exists()
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl');seen=exclusions()
    seen|={int(k) for k in json.loads((ROOT/'experiments/round15/plan.json').read_text())['groups']}
    seen|=set(json.loads((ROOT/'experiments/round16/plan.json').read_text())['fresh_audit_parents'])
    families={family(bank[i]['question']) for i in seen};groups={};records=[]
    for category in ('multiplier','unit_price'):
        pool=[]
        for i,r in enumerate(bank):
            q=r['question'];fam=family(q)
            if i in seen or fam in families or '$' not in q:continue
            multi=bool(re.search(r'\b(?:times|twice|double|triple)\b',q,re.I))
            if (category=='multiplier')!=multi:continue
            if not multi and not re.search(r'\b(?:each|per|apiece)\b',q,re.I):continue
            cs=[c for c in r['candidates'] if anchors(c['text']) and parse_answer(c['text']) is not None]
            if len(cs)<2 or len({parse_answer(c['text']) for c in cs})<2:continue
            if len(anchors(q))<2:continue
            pool.append((hashlib.sha256(('r17-local-20261015:'+fam).encode()).hexdigest(),i,cs))
        for j,(_,i,cs) in enumerate(sorted(pool)[:8]):
            r=bank[i];split='fit' if j<4 else 'audit';seen.add(i);families.add(family(r['question']))
            first=cs[0];second=next(c for c in cs[1:] if parse_answer(c['text'])!=parse_answer(first['text']))
            groups[str(i)]=dict(split=split,category=category,family=family(r['question']))
            for idx,text in [(-1,r['question'])]+[(c['index'],c['text']) for c in (first,second)]:
                records.append(dict(id=f'{i}:{idx}',train_index=i,partition=split,kind='question' if idx==-1 else 'candidate',candidate_index=idx,
                                    question=r['question'],text=text,anchors=anchors(text),sha256=hashlib.sha256(text.encode()).hexdigest()))
    plan=dict(groups=groups,allowed_split='train',sampling='8 fit + 8 audit parents, two real conflicting-answer candidates per parent; bounded eight anchors per text.',
        primary='learned object alignment + quantity roles',controls=['same_numeric_value','nearest_shared_object'],
        labels='object head noun/name and role: count/unit_price/multiplier/total/other; unknown and invalid explicit',
        thresholds=dict(min_gold_links=30,min_role_examples_each=8,min_role_macro_accuracy=.8,min_link_precision=.95,min_link_recall=.8),
        semantic_support_gate='Object match does not prove an equation or final answer; no automatic override.',
        validation_allowed=False,test_access=False,seed=17017)
    (OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    for split in ('fit','audit'):
        subset=[r for r in records if r['partition']==split]
        (OUT/f'{split}_packet.json').write_text(json.dumps(subset,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(dict(parents=len(groups),texts=len(records),anchors=sum(len(r['anchors']) for r in records))))
    for r in records:
        if r['partition']=='fit':
            print(r['id'],r['question'] if r['kind']=='question' else '')
            for i,a in enumerate(r['anchors']):print(i,a['raw'],a['context'])

if __name__=='__main__':main()
