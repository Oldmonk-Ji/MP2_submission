"""Reference-only ambiguity mask; candidate correctness is not process evidence."""
from round22_common import ROOT,load,read,dump,sha
from round147_algebra import normalize,relation_map
from round44_programs import locations
from round29_programs import quantities
from round46_graph import graph
from collections import defaultdict,Counter
from fractions import Fraction as F
import re,sys,time
OUT=ROOT/'experiments/round169_refined'
ORDINALS='first second third fourth fifth sixth seventh eighth ninth tenth'.split()

def ambiguity(q,source,programs,label=0):
    qs=quantities(q);g=graph(q);v=F(qs[source]['value'])
    if label==1:return 'unconfirmed_ordinal_operand_binding' if g['nodes'][source]['kind']=='ordinal' else None
    if v==1:return 'neutral_or_unit_one'
    literals=set()
    for p in programs:
        literals.update(abs(F(n[1])) for _,n in locations(p) if n[0]=='L')
        _,cs=relation_map(normalize(p));literals.update(abs(F(c)) for c in cs)
    if v and (abs(v) in literals or 1/abs(v) in literals):return 'possible_literal_encoding'
    if graph(q)['nodes'][source]['kind']=='percent':
        if any(abs(c) in literals for c in (v/100,1-v/100,1+v/100)):return 'possible_percent_encoding'
    ordinal_operands={F(qs[n[1]]['value']) for p in programs for _,n in locations(p) if n[0]=='N' and g['nodes'][n[1]]['kind']=='ordinal'}
    words=set(re.findall(r'[a-z]+',q.lower()))
    if v.denominator==1 and 2<=v<=10 and all(w in words for w in ORDINALS[:int(v)]) and (v-1 in literals or v-1 in ordinal_operands):return 'explicit_ordinal_enumeration'
    return None

if __name__=='__main__':
    mode=sys.argv[1];assert mode in ('fit','full');tick=time.perf_counter()
    if mode=='fit':
        OUT.mkdir();dump(OUT/'plan.json',dict(method='Mask unconfirmed positive ordinal-operand bindings, and source negatives with: unit/neutral1; possible direct/reciprocal literal encoding in raw or normalized training references; lexical percent encoding; explicit first-to-k ordinal enumeration with reference coefficient or ordinal-valued operand k-1. Absence ambiguity never becomes positive gold. Original158 constant/scope labels unchanged; no candidate graph used.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round169*.py')},test_opened=False))
    else:
        for src,h in read(OUT/'plan.json')['source_hashes'].items():assert sha(ROOT/src)==h
        assert read(OUT/'fit_stats.json')['fit_only']
    b,a,y,ids=load('train');index={int(idx):i for i,idx in enumerate(ids)};refs=defaultdict(list)
    for r in read(ROOT/f'experiments/round115/{mode}_sources.json'):
        if mode=='fit':assert r['fold']!=2
        refs[r['train_index']].append(r['program'])
    stats=Counter();kept=[];removed=[]
    for r in read(ROOT/f'experiments/round158/{mode}_source_labels.json'):
        if mode=='fit':assert r['fold']!=2
        stats['original_source_labels']+=1;stats['original_negative']+=r['label']==0;reason=ambiguity(b[index[r['train_index']]]['question'],r['source'],refs[r['train_index']],r['label'])
        if reason:removed.append(dict(**r,reason=reason));stats['unknown_'+reason]+=1
        else:kept.append(r);stats['retained_source_labels']+=1;stats['retained_negative']+=r['label']==0
    dump(OUT/f'{mode}_source_labels.json',kept);dump(OUT/f'{mode}_removed.json',removed)
    for name in ('constant','scope'):
        rows=read(ROOT/f'experiments/round158/{mode}_{name}_labels.json');dump(OUT/f'{mode}_{name}_labels.json',rows);assert sha(OUT/f'{mode}_{name}_labels.json')==sha(ROOT/f'experiments/round158/{mode}_{name}_labels.json')
    if mode=='fit':
        rm={(r['train_index'],r['source']):r['reason'] for r in removed};review=read(ROOT/'experiments/round164/fit_semantic_review.json')['examples'];dump(OUT/'fit_review_coverage.json',[dict(train_index=r['train_index'],source=r['source'],prior_review=r['category'],masked_reason=rm.get((r['train_index'],r['source']))) for r in review])
    dump(OUT/f'{mode}_stats.json',dict(stats=stats,fit_only=mode=='fit',seconds=time.perf_counter()-tick,test_opened=False));print(stats,flush=True)
