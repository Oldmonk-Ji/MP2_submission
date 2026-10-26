"""TRAIN-fit-only arithmetic formatting repair audit; preserve all old roots."""
from round22_common import ROOT,load,read,dump,sha
from round152_markup import merged,checks,clean
from round46_graph import graph
from round44_programs import canonical,tree
from round109_alignment import mathematical_key
from round51_programs import value
from features import parse_answer
from collections import defaultdict,Counter
import numpy as np,time
OUT=ROOT/'experiments/round152'
def key(p):
    try:return mathematical_key(tree(p))
    except (ValueError,ZeroDivisionError):return None
if __name__=='__main__':
    OUT.mkdir(exist_ok=True);assert not (OUT/'fit_audit.json').exists();tick=time.perf_counter();dump(OUT/'plan.json',dict(checks=checks(),method='Remove properly paired Markdown strong emphasis before numeric/operator normalization. Preserve exponents and official parsed final answer. Union recovered source-root hypotheses with unchanged old hypotheses; no reference used by inference. Unsupported symbolic operators are unknown.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round152*.py')},test_opened=False))
    b,a,y,ids=load('train');old=read(ROOT/'experiments/round102/train_programs.json');refs=defaultdict(list)
    for r in read(ROOT/'experiments/round115/fit_sources.json'):refs[r['train_index']].append(r['program']);assert r['fold']!=2
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(z['ids'],ids)
    stats=Counter();rows=[];examples=[]
    for i in np.flatnonzero(f!=2):
        idx=int(ids[i]);row=b[i];g=graph(row['question']);packet=[];changed=False;stats['fit_parents']+=1
        for j,c in enumerate(row['candidates']):
            roots=merged(row['question'],c['text'],g,old[i][j]);assert len(roots)>=len(old[i][j]);packet.append(roots)
            if len(roots)>len(old[i][j]):
                changed=True;stats['changed_candidates']+=1;stats['added_roots']+=len(roots)-len(old[i][j]);oldkeys={repr(canonical(tree(p))) for p in old[i][j]}
                for p in roots:
                    if repr(canonical(tree(p))) not in oldkeys:assert str(value(p,g['quantities']))==parse_answer(c['text']);stats['new_root_execution_checks']+=1
        rows.append(dict(train_index=idx,programs=packet));stats['changed_parents']+=changed
        if changed and idx in refs:
            gold={key(p) for p in refs[idx]}-{None};before=after=False
            for j in np.flatnonzero(y[i]):
                stats['unsupported_correct_root_equivalence']+=sum(key(p) is None for p in packet[j])
                before|=any(key(p) in gold for p in old[i][j]);after|=any(key(p) in gold for p in packet[j])
            stats['changed_reference_parents']+=1;stats['previous_correct_program_covered']+=before;stats['new_correct_program_covered']+=after;stats['recovered_reference_equivalent_correct_parent']+=after and not before
            examples.append(dict(train_index=idx,before=before,after=after,question=row['question']))
        if i%500==0:print('AUDIT',i,flush=True)
    dump(OUT/'fit_programs.json',rows);dump(OUT/'fit_examples.json',examples);dump(OUT/'fit_audit.json',dict(stats=stats,fit_only=True,seconds=time.perf_counter()-tick,test_opened=False));print(stats,flush=True)
