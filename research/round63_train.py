"""Expand local relation supervision from unambiguous reference-final traces."""
from round22_common import ROOT,CODE,read,dump,sha
from common import read_jsonl
from features import parse_answer
from round44_programs import edits,valid,tree,locations
from round56_local import focus,csr
from round50_selector import select
from collections import Counter
from fractions import Fraction as F
from scipy import sparse
import numpy as np,shutil,re
import round48_train as training
OUT=ROOT/'experiments/round63'

def literal_supported(q,v):
    if F(v) in (0,1):return True
    cues={'2':r'half|twice|double|pair','3':r'third|triple','4':r'quarter|fourth',
          '5':r'nickel','7':r'week','10':r'dime','12':r'dozen|feet|foot|inch|year|month',
          '24':r'hour|day','60':r'hour|minute|second','100':r'percent|%|cent|dollar','1000':r'kilometer|kilogram|meter|gram'}
    return str(F(v)) in cues and bool(re.search(cues[str(F(v))],q,re.I))

if __name__=='__main__':
    source=ROOT/'experiments/round56';assert not (OUT/'plan.json').exists()
    dump(OUT/'supervision_filter_freeze.json',dict(source_sha256=sha(__file__),rationale='TRAIN-fit source review identified ungrounded conversion result 24 in question16; reject constants without source lexical support. Require reference-final root.',test_opened=False))
    for name in ('train_targets.npz','node_features.npz','program_node_ptr.npy','held_source_review_before_reference.json'):
        shutil.copy2(source/name,OUT/name)
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');pairs=read(source/'constraint_pairs.json')
    excluded={r['train_index'] for r in read(OUT/'held_source_review_before_reference.json')}
    parents=set();counts=Counter();accepted=[]
    for r in read(OUT/'aligned_reference_steps.json'):
        if r['train_index'] in excluded:continue
        roots=[s for s in r['steps'] if s['value']==parse_answer(refs[r['train_index']]['answer'])]
        if not roots:continue
        s=roots[-1];p=tree(s['program'])
        if any(n[0]=='L' and not literal_supported(r['question'],n[1]) for _,n in locations(p)):continue
        per=Counter();seen=set()
        for kind,neg,decision in edits(p,r['quantities']):
            v=valid(neg,r['quantities']);key=repr(neg)
            if v is None or v==s['value'] or key in seen or per[kind]>=3:continue
            seen.add(key);per[kind]+=1
            pairs.append(dict(train_index=r['train_index'],fold=r['fold'],question=r['question'],positive=p,negative=neg,kind=kind,decision=decision,
                provenance='strict_reference_trace_not_human_gold',reference_evidence=s['evidence']))
            counts[kind]+=1;parents.add(r['train_index'])
        if seen:accepted.append(dict(train_index=r['train_index'],fold=r['fold'],program=p,reference_evidence=s['evidence']))
    dump(OUT/'accepted_reference_programs.json',accepted);dump(OUT/'constraint_pairs.json',pairs)
    np.save(OUT/'constraint_parent_ids.npy',np.array([p['train_index'] for p in pairs]))
    ff=[focus(p) for p in pairs]
    sparse.save_npz(OUT/'constraint_positive.npz',csr([p[0] for p in ff]));sparse.save_npz(OUT/'constraint_negative.npz',csr([p[1] for p in ff]))
    dump(OUT/'supervision_stats.json',dict(reference_parents=len(parents),new_pairs=sum(counts.values()),total_pairs=len(pairs),by_kind=dict(counts),held_review_parents_excluded=True,test_opened=False))
    plan=read(source/'plan.json');plan.update(objective='Matched round56 features/programs/SFT budget; add strictly reference-trace-aligned relation supervision',arms=['sft'],source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round63*.py')})
    dump(OUT/'plan.json',plan);dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(read(OUT/'supervision_stats.json'),flush=True)
    training.OUT=OUT;training.DIM=2048;training.select=select;training.main()
