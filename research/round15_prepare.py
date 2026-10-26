"""Sample natural TRAIN question/candidate binding annotations before prediction.

No synthetic texts or reference answers. Frozen, bounded anchor coverage is
reported separately from whole-candidate coverage.
"""
from data_access import ROOT, CODE, read_jsonl, parse_answer
from round7_core import quantities
from round14_prepare import family
import hashlib,json,re

OUT=ROOT/'experiments/round15'


def anchors(text):
    result=[]
    for q in quantities(text):
        start=max(text.rfind('\n',0,q['start']),text.rfind('. ',0,q['start']))+1
        ends=[x for x in [text.find('\n',q['end']),text.find('. ',q['end'])] if x>=0]
        end=min(ends) if ends else len(text)
        clause=text[start:end]
        # Only verbal quantity claims. List headings / pure formulas / final
        # answer markers do not carry enough local semantic context by themselves.
        if '####' in clause or '\\boxed' in clause:continue
        if len(re.findall(r'\b[A-Za-z]{2,}\b',clause))<5:continue
        if re.match(r'\s*(?:\*\*)?'+re.escape(q['raw'])+r'[.)]\s',text[q['start']:]) and q['start']-start<5:continue
        result.append(dict(**q,context_start=start,context_end=end,context=clause))
    if len(result)>2:
        # Keep endpoints, plus two currency anchors which would otherwise drop
        # the price denominator in the middle of a multi-item sentence.
        chosen=[result[0],result[-1]]
        for item in result:
            if text[:item['start']].rstrip().endswith('$') and item not in chosen:
                chosen.append(item)
                if len(chosen)>=4:break
        result=sorted(chosen,key=lambda a:a['start'])
    return result


def exclusions():
    seen=set()
    for roundnum in (7,8):
        for name in ('training_annotations.json','audit_annotations.json'):
            path=ROOT/f'experiments/round{roundnum}'/name
            if path.exists():seen.update(r['index'] for r in json.loads(path.read_text(encoding='utf-8')))
    for name in ('fit_contrasts.json','development_contrasts.json','audit_contrasts.json'):
        path=ROOT/'experiments/round14'/name
        if path.exists():seen.update(r['train_index'] for r in json.loads(path.read_text(encoding='utf-8')))
    for path in (ROOT/'experiments/human_review').glob('review_pack*_manifest.json'):
        seen.update(v['train_index'] for v in json.loads(path.read_text())['items'].values())
    return seen


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'plan.json').exists():print('Existing split retained.');return
    bank=read_jsonl(CODE/'data/candidates/gsm8k_train.jsonl');seen=exclusions();groups={};selected=[];seenfamilies={family(bank[i]['question']) for i in seen}
    patterns=dict(coref=r'\b(?:he|she|his|her|they|their)\b',price=r'\$.*\b(?:each|per|apiece)\b',time=r'\b(?:years? ago|in \w+ years|next year|last year|will be|used to|now)\b')
    for category,pat in patterns.items():
        pool=[]
        for i,r in enumerate(bank):
            if i in seen or family(r['question']) in seenfamilies or not re.search(pat,r['question'],re.I):continue
            if category=='coref' and len(set(re.findall(r'\b[A-Z][a-z]+\b',r['question'])) )<2:continue
            if len(anchors(r['question']))<2:continue
            if sum(len(anchors(c['text']))>=2 for c in r['candidates'])<2:continue
            key=hashlib.sha256(('r15-20261015:'+category+':'+family(r['question'])).encode()).hexdigest();pool.append((key,i))
        for j,(_,i) in enumerate(sorted(pool)[:8]):
            r=bank[i];seen.add(i);seenfamilies.add(family(r['question']))
            split='fit' if j<5 else 'audit'
            valid=[c for c in r['candidates'] if len(anchors(c['text']))>=2]
            first=valid[0];second=next((c for c in valid[1:] if parse_answer(c['text'])!=parse_answer(first['text'])),valid[1])
            groups[str(i)]=dict(split=split,category=category,family_sha256=hashlib.sha256(family(r['question']).encode()).hexdigest())
            for kind,idx,text in [('question',-1,r['question'])]+[('candidate',c['index'],c['text']) for c in (first,second)]:
                selected.append(dict(id=f'{i}:{idx}',train_index=i,partition=split,category=category,kind=kind,candidate_index=idx,question=r['question'],text=text,anchors=anchors(text),source='unmodified official train input',sha256=hashlib.sha256(text.encode()).hexdigest()))
    plan=dict(groups=groups,allowed_data=['train'],label_fields=['owner','stage','price_scope'],
              annotation_truth='Assistant-authored natural-text labels; human uncertainties masked, not independent human gold.',
              sampling='8 parents per category: five fit, three audit; question + two naturally occurring candidates, prefer different parsed final answers without reading correctness.',
              anchor_scope='First and last eligible verbal quantity mention per selected text. Metrics cover these sampled anchors, not all relations in all four candidates.',
              audit_gate=dict(min_candidate_anchors=36,min_joint_precision=.95,min_correct_coverage=.80,min_each_category_joint=.90),
              validation_allowed=False,end_to_end_gate='Requires extraction gate and a fair train-side outer comparison. Existing baseline trained all train; do not reuse as independent outer fit.',
              seed=20261015)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    for split in ('fit','audit'):
        rows=[r for r in selected if r['partition']==split]
        (OUT/f'{split}_annotation_packet.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(parents=len(groups),fit_parents=sum(x['split']=='fit' for x in groups.values()),audit_parents=sum(x['split']=='audit' for x in groups.values()),text_records=len(selected),anchors=sum(len(r['anchors']) for r in selected))))
    # Audit packet exists for later annotation, but is not displayed now.
    for r in selected:
        if r['partition']=='fit':print(json.dumps(dict(id=r['id'],category=r['category'],question=r['question'],anchors=[dict(raw=a['raw'],context=a['context']) for a in r['anchors']]),ensure_ascii=True))


if __name__=='__main__':main()
