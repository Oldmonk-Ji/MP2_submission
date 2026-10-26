"""Restore explicit fraction/interval operands masked as ordinals in169."""
from round22_common import ROOT,load,read,dump,sha
from round169_reference_mask import ambiguity
from round175_roles import kind,checks
from round46_graph import graph
from collections import defaultdict,Counter
import numpy as np,sys,time
OUT=ROOT/'experiments/round175'
if __name__=='__main__':
    mode=sys.argv[1];assert mode in ('fit','full');tick=time.perf_counter()
    if mode=='fit':
        OUT.mkdir();dump(OUT/'plan.json',dict(checks=checks(),method='Same reference-only ambiguity policy as169_refined, but preserve positive source labels for explicit numerator-third/fourth/etc fractions and every-third/etc interval counts. Candidate graphs not consulted. Constant/scope labels unchanged158. Freeze before full TRAIN application.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round175*.py')},test_opened=False))
    else:
        for src,h in read(OUT/'plan.json')['source_hashes'].items():assert sha(ROOT/src)==h
        assert read(OUT/'fit_stats.json')['fit_only']
    b,a,y,ids=load('train');index={int(idx):i for i,idx in enumerate(ids)};refs=defaultdict(list);stats=Counter();rows=[];removed=[]
    for r in read(ROOT/f'experiments/round115/{mode}_sources.json'):
        if mode=='fit':assert r['fold']!=2
        refs[r['train_index']].append(r['program'])
    for r in read(ROOT/f'experiments/round158/{mode}_source_labels.json'):
        q=b[index[r['train_index']]]['question'];reason=ambiguity(q,r['source'],refs[r['train_index']],r['label']);stats['original_labels']+=1
        if reason=='unconfirmed_ordinal_operand_binding' and kind(q,r['source'])!='ordinal':reason=None;stats['explicit_fraction_or_interval_positive_restored']+=1
        if reason:removed.append(dict(**r,reason=reason));stats['unknown_'+reason]+=1
        else:rows.append(r);stats['retained_labels']+=1;stats['retained_negative']+=r['label']==0
    dump(OUT/f'{mode}_source_labels.json',rows);dump(OUT/f'{mode}_removed.json',removed)
    for name in ('constant','scope'):
        dump(OUT/f'{mode}_{name}_labels.json',read(ROOT/f'experiments/round158/{mode}_{name}_labels.json'));assert sha(OUT/f'{mode}_{name}_labels.json')==sha(ROOT/f'experiments/round158/{mode}_{name}_labels.json')
    if mode=='fit':
        with np.load(ROOT/'experiments/round77/nested/components.npz') as z:folds=z['folds'];assert np.array_equal(ids,z['ids'])
        changed=[]
        for i in np.flatnonzero(folds!=2):
            q=b[i]['question'];g=graph(q)
            for j,x in enumerate(g['quantities']):
                new=kind(q,j);old=g['nodes'][j]['kind']
                if old!=new:changed.append(dict(train_index=int(ids[i]),source=j,raw=x['raw'],old_kind=old,new_kind=new,evidence=q[max(0,x['start']-25):min(len(q),x['end']+35)]))
        dump(OUT/'fit_type_changes.json',changed);stats['fit_quantity_type_changes']=len(changed);stats['fit_type_changed_parents']=len({r['train_index'] for r in changed})
    dump(OUT/f'{mode}_stats.json',dict(stats=stats,fit_only=mode=='fit',seconds=time.perf_counter()-tick,test_opened=False));print(stats,flush=True)
