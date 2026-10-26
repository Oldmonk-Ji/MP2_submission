from round22_common import ROOT,load,read,dump,sha
from round186_trace import extract,checks
from round109_alignment import mathematical_key
from round44_programs import tree
from collections import defaultdict,Counter
import numpy as np,time
OUT=ROOT/'experiments/round187'
def key(p):
    try:return mathematical_key(tree(p))
    except (ValueError,ZeroDivisionError):return None
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(checks=checks(),method='TRAIN-fit audit of strict optional candidate traces. Complete source/verified-intermediate lineage only; no arithmetic gap filling, no claimed-number root, bounded alternatives, improved fraction kinds. Absence is unknown, not proof of final-answer error.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round186*.py')},test_opened=False))
    b,a,y,ids=load('train');old=read(ROOT/'experiments/round102/train_programs.json');refs=defaultdict(list)
    for r in read(ROOT/'experiments/round115/fit_sources.json'):refs[r['train_index']].append(r['program']);assert r['fold']!=2
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:folds=z['folds'];assert np.array_equal(ids,z['ids'])
    stats=Counter();rows=[];examples=[];events=Counter()
    for i in np.flatnonzero(folds!=2):
        idx=int(ids[i]);row=b[i];packet=[];stats['fit_parents']+=1
        for j,c in enumerate(row['candidates']):
            roots,report=extract(row['question'],c['text']);packet.append(roots);prefix='correct_candidate' if y[i,j] else 'incorrect_candidate';stats[prefix+'_n']+=1;stats[prefix+'_legacy_root']+=bool(old[i][j]);stats[prefix+'_strict_root']+=bool(roots);stats['strict_roots']+=len(roots);events.update(report.get('events',[]))
        rows.append(dict(train_index=idx,programs=packet))
        if idx in refs:
            gold={key(p) for p in refs[idx]}-{None};before=any(key(p) in gold for j in np.flatnonzero(y[i]) for p in old[i][j]);after=any(key(p) in gold for j in np.flatnonzero(y[i]) for p in packet[j]);stats['reference_parents']+=1;stats['legacy_reference_equivalent_correct']+=before;stats['strict_reference_equivalent_correct']+=after;stats['new_only']+=after and not before;stats['old_only']+=before and not after
            if after!=before:examples.append(dict(train_index=idx,legacy=before,strict=after,question=row['question']))
        if i%500==0:print('AUDIT',i,flush=True)
    dump(OUT/'fit_programs.json',rows);dump(OUT/'fit_coverage_changes.json',examples);dump(OUT/'fit_audit.json',dict(stats=stats,events=events,fit_only=True,seconds=time.perf_counter()-tick,test_opened=False));print(stats,flush=True)
