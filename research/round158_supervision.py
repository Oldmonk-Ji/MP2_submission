"""Separate source use, literal presence, and relation scope supervision.

Design/audit on TRAIN-fit only. All reference alternatives must agree; no
candidate final label is used to invent a program or a missing relation.
"""
from round22_common import ROOT,load,read,dump,sha
from round147_algebra import normalize,relation_map,candidate_constants,ABSENT
from round29_programs import quantities
from collections import defaultdict,Counter
import numpy as np,time,sys
OUT=ROOT/'experiments/round158'

def sources(p):
    if p[0]=='N':return {int(p[1])}
    if p[0]=='L':return set()
    return set().union(*(sources(q) for q in p[1:]))

if __name__=='__main__':
    mode=sys.argv[1];assert mode in ('fit','full');tick=time.perf_counter()
    if mode=='fit':
        OUT.mkdir();dump(OUT/'plan.json',dict(method='Factor source/literal absence into source-not-used OR literal-not-present. Supervise these two global decisions separately and scope only when both are present. Retain all-reference consensus; uncertainty masked. Prior five-class head provides a matched independent baseline.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round158*.py')},test_opened=False))
    else:
        for src,h in read(OUT/'plan.json')['source_hashes'].items():assert sha(ROOT/src)==h
        assert read(OUT/'fit_stats.json')['fit_only']
    b,a,y,ids=load('train');index={int(idx):i for i,idx in enumerate(ids)};programs=read(ROOT/'experiments/round102/train_programs.json');refs=defaultdict(list);folds={};stats=Counter()
    for r in read(ROOT/f'experiments/round115/{mode}_sources.json'):
        assert mode!='fit' or r['fold']!=2
        p=normalize(r['program']);mapping,cs=relation_map(p);refs[r['train_index']].append((sources(p),cs));folds[r['train_index']]=r['fold']
    source_labels=[];constant_labels=[];lookup={}
    for idx,alternatives in refs.items():
        i=index[idx];n=len(quantities(b[i]['question']));constants=candidate_constants(programs[i]);lookup[idx]={}
        for kind,items,out,at in [('source',range(n),source_labels,0),('constant',constants,constant_labels,1)]:
            for item in items:
                values={int(item in alt[at]) for alt in alternatives}
                if len(values)!=1:stats[kind+'_unknown']+=1;continue
                label=next(iter(values));out.append(dict(train_index=idx,fold=folds[idx],**{kind:item},label=label));lookup[idx][(kind,item)]=label;stats[kind+'_labels']+=1;stats[kind+'_present']+=label
    scope=[]
    for r in read(ROOT/f'experiments/round148/{mode}_labels.json'):
        if r['label']==ABSENT:continue
        idx=r['train_index'];assert lookup[idx][('source',r['quantity'])]==1;assert lookup[idx][('constant',r['constant'])]==1;scope.append(r)
    stats['scope_labels']=len(scope);stats['reference_parents']=len(refs)
    if mode=='fit':
        rows=read(ROOT/'experiments/round148/full_labels.json');table=defaultdict(list)
        with np.load(ROOT/'experiments/round149/oof_scores.npz') as z:pred=z['nested_probability'].argmax(1);assert np.array_equal(z['parents'],[r['train_index'] for r in rows])
        for k,r in enumerate(rows):
            if r['fold']!=2:table[r['train_index']].append((r['quantity'],r['constant'],r['label']!=ABSENT,pred[k]!=ABSENT))
        for idx,rr in table.items():
            n=len(quantities(b[index[idx]]['question']));cs=sorted({r[1] for r in rr})
            if len(rr)!=n*len(cs):continue
            cm={c:i for i,c in enumerate(cs)};gold=np.zeros((n,len(cs)),bool);guess=np.zeros_like(gold)
            for i,c,target,p in rr:gold[i,cm[c]]=target;guess[i,cm[c]]=p
            def coherent(m):return np.array_equal(m,m.any(1)[:,None]&m.any(0)[None,:])
            assert coherent(gold);stats['complete_fit_parent_matrices']+=1;stats['independent_argmax_nonrectangular_parents']+=not coherent(guess)
    dump(OUT/f'{mode}_source_labels.json',source_labels);dump(OUT/f'{mode}_constant_labels.json',constant_labels);dump(OUT/f'{mode}_scope_labels.json',scope);dump(OUT/f'{mode}_stats.json',dict(stats=stats,fit_only=mode=='fit',seconds=time.perf_counter()-tick,test_opened=False));print(stats,flush=True)
