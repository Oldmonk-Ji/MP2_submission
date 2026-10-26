"""Do not equate absence from one reference representation with irrelevance.

TRAIN-only conservative filter: neutral-one source omissions and any negative
contradicted by an executable correct-candidate graph become unknown, not positive.
If no correct-candidate graph is available, negative evidence remains unverified.
"""
from round22_common import ROOT,load,read,dump,sha
from round147_algebra import normalize,relation_map
from round162_decisions import used_sources
from round29_programs import quantities
from round51_programs import value
from features import parse_answer
from fractions import Fraction as F
from collections import Counter
import numpy as np,time,sys
OUT=ROOT/'experiments/round164'
if __name__=='__main__':
    mode=sys.argv[1];assert mode in ('fit','full');tick=time.perf_counter()
    if mode=='fit':
        OUT.mkdir();dump(OUT/'plan.json',dict(method='Retain all prior unanimous-reference positives. Source negative becomes unknown if value is1, any correct-candidate executable normalized root uses it, or no correct-candidate root is available. Constant negative becomes unknown if any such root contains it or no root is available. Never convert candidate hypotheses into positive semantic gold. Conditional scope labels unchanged.',motivation='TRAIN-fit audit: reference4+4 omits source two; normalization drops unit-one; subtotal and total can be alternative correct representations.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round164*.py')},test_opened=False))
    else:
        for src,h in read(OUT/'plan.json')['source_hashes'].items():assert sha(ROOT/src)==h
        assert read(OUT/'fit_stats.json')['fit_only']
    b,a,y,ids=load('train');index={int(idx):i for i,idx in enumerate(ids)};packets=read(ROOT/'experiments/round102/train_programs.json');cache={};stats=Counter();removed=[]
    def evidence(idx):
        if idx in cache:return cache[idx]
        i=index[idx];q=quantities(b[i]['question']);used=set();constants=set();count=0
        for j in np.flatnonzero(y[i]):
            answer=parse_answer(b[i]['candidates'][j]['text'])
            for p in packets[i][j]:
                try:
                    assert str(value(p,q))==answer
                    normal=normalize(p);_,cs=relation_map(normal);used.update(used_sources(normal));constants.update(cs);count+=1
                except (ValueError,ZeroDivisionError):continue
        cache[idx]=(used,constants,count);return cache[idx]
    for name in ('source','constant','scope'):
        rows=read(ROOT/f'experiments/round158/{mode}_{name}_labels.json');kept=[]
        for r in rows:
            if mode=='fit':assert r['fold']!=2
            stats[name+'_before']+=1
            if name=='scope' or r['label']==1:kept.append(r);stats[name+'_kept']+=1;continue
            idx=r['train_index'];used,constants,count=evidence(idx);reason=None
            if name=='source' and F(quantities(b[index[idx]]['question'])[r['source']]['value'])==1:reason='neutral_or_unit_one'
            elif not count:reason='no_executable_correct_candidate_evidence'
            elif name=='source' and r['source'] in used:reason='possible_source_use_in_correct_candidate'
            elif name=='constant' and r['constant'] in constants:reason='possible_constant_in_correct_candidate'
            if reason:
                stats[name+'_unknown_'+reason]+=1;removed.append(dict(head=name,reason=reason,**r));continue
            kept.append(r);stats[name+'_kept']+=1;stats[name+'_retained_negative']+=1
        dump(OUT/f'{mode}_{name}_labels.json',kept)
    dump(OUT/f'{mode}_removed.json',removed);dump(OUT/f'{mode}_stats.json',dict(stats=stats,fit_only=mode=='fit',seconds=time.perf_counter()-tick,test_opened=False));print(stats,flush=True)
