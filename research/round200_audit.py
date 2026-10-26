"""Fit-fold audit: occurrence-span trace versus frozen line-level trace.

Quantifies how often a whole-line hint mislabels one of several same-valued
occurrences on a line, how root sets change, and whether reference-equivalent
coverage improves. Fit folds only; TEST stays sealed.
"""
from round22_common import ROOT,load,read,dump,sha
from round192_trace import extract as old_extract
from round200_trace import extract as new_extract,checks
from round200_trace import tokens,span_wanted
from round192_context import hint,names_in
from round152_markup import clean
from round109_alignment import mathematical_key
from round44_programs import canonical,tree
from fractions import Fraction as F
from collections import defaultdict,Counter
import numpy as np,time
OUT=ROOT/'experiments/round200'
def key(p):
    try:return mathematical_key(tree(p))
    except (ValueError,ZeroDivisionError):return None
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();assert read(ROOT/'experiments/round199/training_complete.json')['test_opened']==False
    dump(OUT/'plan.json',dict(checks=checks(),method='Compare frozen round192 line-level trace with round200 occurrence-span trace on TRAIN fit folds. Same prune-only-on-explicit-match policy; spans change only which context applies to which occurrence. Design evidence only; nothing frozen yet.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round200*.py')},test_opened=False))
    b,a,y,ids=load('train');refs=defaultdict(list)
    for r in read(ROOT/'experiments/round115/fit_sources.json'):refs[r['train_index']].append(r['program']);assert r['fold']!=2
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:folds=z['folds'];assert np.array_equal(ids,z['ids'])
    stats=Counter();events_new=Counter();events_old=Counter();examples=[];line_cases=[];t0=time.perf_counter()
    for i in np.flatnonzero(folds!=2):
        idx=int(ids[i]);row=b[i]
        for j,c in enumerate(row['candidates']):
            old,or_=old_extract(row['question'],c['text']);new,nr=new_extract(row['question'],c['text'])
            events_old.update(or_.get('events',[]));events_new.update(nr.get('events',[]))
            ok={key(p) for p in old}-{None};nk={key(p) for p in new}-{None}
            stats['candidates']+=1;stats['old_roots']+=len(old);stats['new_roots']+=len(new)
            if ok==nk:stats['identical']+=1
            else:
                stats['changed']+=1;stats['new_only_programs']+=len(nk-ok);stats['old_only_programs']+=len(ok-nk)
                if len(new)>len(old):stats['more_roots']+=1
                if len(new)<len(old):stats['fewer_roots']+=1
            if idx in refs:
                gold={key(p) for p in refs[idx]}-{None};bo=bool(ok&gold);bn=bool(nk&gold)
                stats['reference_parents']+=1
                if bn:stats['new_reference_equivalent_correct']+=1
                if bo:stats['old_reference_equivalent_correct']+=1
                if bn and not bo:stats['coverage_new_only']+=1;examples.append(dict(train_index=idx,candidate=j,kind='coverage_new_only',question=row['question']))
                if bo and not bn:stats['coverage_old_only']+=1;examples.append(dict(train_index=idx,candidate=j,kind='coverage_old_only',question=row['question']))
        if stats['candidates']%2000==0:print('AUDIT',stats['candidates'],time.perf_counter()-t0,flush=True)
    t1=time.perf_counter()
    # Line-level misapplication diagnostics on fit candidates.
    for i in np.flatnonzero(folds!=2)[:400]:
        row=b[i];names=names_in(row['question'])
        for c in row['candidates']:
            for line in clean(c['text']).splitlines():
                toks=tokens(line);vals=Counter(tv for tv,_,_ in toks)
                if sum(1 for v in vals.values() if v>1)<1:continue
                for k,(tv,ts,te) in enumerate(toks):
                    sw=span_wanted(line,ts,te,names)
                    lh=hint(line,tv,names)
                    if sw and sw!=lh:
                        stats['occurrences_with_differing_span_context']+=1
                        if len(line_cases)<12:
                            line_cases.append(dict(question=row['question'],line=line,value=str(tv),span_context=sw,line_context=lh))
                        break
    dump(OUT/'fit_audit.json',dict(stats=stats,events_old=events_old,events_new=events_new,seconds=time.perf_counter()-tick,test_opened=False))
    dump(OUT/'coverage_examples.json',examples[:40]);dump(OUT/'span_line_examples.json',line_cases)
    print(stats,flush=True);print('seconds',time.perf_counter()-tick,flush=True)
