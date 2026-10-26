"""Validate returned human responses against a generated review manifest.
Never overwrites training labels or frozen model assets.
"""
from data_access import ROOT
import argparse,json,hashlib
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('answers',type=Path);ap.add_argument('--manifest',type=Path);args=ap.parse_args()
    data=json.loads(args.answers.read_text(encoding='utf-8-sig'))
    if args.manifest is not None:
        m=json.loads(args.manifest.read_text(encoding='utf-8'))
    else:
        matches=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'experiments/human_review').glob('review_pack*_manifest.json')]
        matches=[x for x in matches if x['pack_id']==data.get('pack_id') and x['version']==data.get('pack_version')]
        if len(matches)!=1:raise ValueError('No unique matching local review manifest')
        m=matches[0]
    if data.get('schema')!='mp2-review-answers/v1' or data.get('split')!='train' or data.get('pack_id')!=m['pack_id'] or data.get('pack_version')!=m['version']:raise ValueError('Wrong schema, pack, split or version')
    accepted=[];seen=set()
    for r in data['answers']:
        if r['id'] in seen:raise ValueError('Duplicate response ID')
        seen.add(r['id']);q=m['items'].get(r['id'])
        if not q or r.get('question_sha256')!=q['question_sha256']:raise ValueError('Unknown or modified question')
        if r['status'] not in ('answered','skipped','draft'):raise ValueError('Unknown response status')
        if r['choice'] is not None and r['choice'] not in q['option_ids']:raise ValueError('Invalid option')
        if r['status']=='answered' and r['choice'] is None:raise ValueError('Missing answer')
        if not isinstance(r['note'],str) or len(r['note'])>4000:raise ValueError('Invalid note')
        accepted.append(dict(id=r['id'],train_index=q['train_index'],question_sha256=q['question_sha256'],choice=r['choice'],status=r['status'],note=r['note'],updated_at=r.get('updated_at')))
    digest=hashlib.sha256(args.answers.read_bytes()).hexdigest();out=ROOT/'experiments/human_review/returned';out.mkdir(exist_ok=True);p=out/(digest[:16]+'.json')
    record=dict(source_sha256=digest,pack_id=m['pack_id'],responses=accepted,status='human input recorded; assistant adjudication pending; not automatically promoted to training gold')
    p.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(dict(saved=str(p),answered=sum(r['status']=='answered' for r in accepted),uncertain=sum(r['choice']=='U' for r in accepted)),ensure_ascii=False))
if __name__=='__main__':main()
