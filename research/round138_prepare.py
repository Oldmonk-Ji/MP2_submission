"""Freeze source/literal supervision from TRAIN-fit references before held use."""
from round22_common import ROOT,load,read,dump,sha
from round138_literals import normalize,relation_map,candidate_constants,checks,ABSENT,LABELS
from round29_programs import quantities
from round51_programs import value
from round101_templates import execute
from collections import defaultdict,Counter
from fractions import Fraction as F
import numpy as np,sys,time
OUT=ROOT/'experiments/round138'
if __name__=='__main__':
    mode=sys.argv[1];assert mode in ('fit','full');tick=time.perf_counter()
    if mode=='fit':
        OUT.mkdir();dump(OUT/'plan.json',dict(checks=checks(),method='Auxiliary source/literal relation head with ADD,MUL,DIV,DIV_REVERSE,ABSENT labels. Canonicalize neutral constants and constant denominators first. Candidate-only union of literal values defines questions to supervise. Retain a label only if every complete round115 reference alternative agrees; conflicting repeated paths are unknown.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round138*.py')},training_weight_policy='For each parent, split equal total mass between non-ABSENT and ABSENT examples when both exist; otherwise normalize within available group. No new candidate or answer generated.',test_opened=False))
    else:
        for src,h in read(OUT/'plan.json')['source_hashes'].items():assert sha(ROOT/src)==h,src
        assert read(OUT/'fit_stats.json')['fit_only']
    b,a,y,ids=load('train');packets=read(ROOT/'experiments/round102/train_programs.json');sources=read(ROOT/f'experiments/round115/{mode}_sources.json');refs=defaultdict(list);folds={}
    for r in sources:refs[r['train_index']].append(r['program']);folds[r['train_index']]=r['fold']
    stats=Counter();rows=[];examples=[];rng=np.random.default_rng(138138)
    for i,idx in enumerate(ids):
        idx=int(idx)
        if idx not in refs:continue
        assert mode!='fit' or folds[idx]!=2
        constants=candidate_constants(packets[i]);n=len(quantities(b[i]['question']));maps=[]
        for p in refs[idx]:
            normal=normalize(p);r,cs=relation_map(normal);maps.append(r)
            vals=[F(int(k)) for k in rng.integers(1,30,size=n)];qs=[dict(value=str(v)) for v in vals]
            try:assert execute(normal,vals)==value(p,qs);stats['exact_normalization_checks']+=1
            except ZeroDivisionError:stats['zero_denominator_skips']+=1
        accepted=0;positive=0
        for source in range(n):
            for constant in constants:
                labels={r.get((source,constant),ABSENT) for r in maps}
                if len(labels)!=1 or None in labels:stats['ambiguous_pairs_skipped']+=1;continue
                label=next(iter(labels));rows.append(dict(train_index=idx,fold=folds[idx],quantity=source,constant=constant,label=label));accepted+=1;positive+=label!=ABSENT;stats[LABELS[label]]+=1
        stats['reference_parents']+=1;stats['supervised_parents']+=accepted>0;stats['parents_with_non_absent_constraints']+=positive>0;stats['pairs']+=accepted;stats['candidate_literal_values']+=len(constants)
    if mode=='fit':
        blind=read(ROOT/'experiments/round137/fit_examples.json')
        for r in blind:
            pm,pc=relation_map(normalize(r['correct']['program']));nm,nc=relation_map(normalize(r['incorrect']['program']))
            changed=pm!=nm or pc!=nc;stats['old_blindspot_examples']+=1;stats['literal_signature_distinguishes_blindspot']+=changed
            examples.append(dict(train_index=r['train_index'],distinguished=changed,correct_constants=sorted(pc),incorrect_constants=sorted(nc)))
        dump(OUT/'fit_blindspot_check.json',examples)
    dump(OUT/f'{mode}_labels.json',rows);dump(OUT/f'{mode}_stats.json',dict(stats=stats,fit_only=mode=='fit',seconds=time.perf_counter()-tick,test_opened=False));print(stats,flush=True)
