"""Fit-only audit: supervise a pair only when every reference alternative agrees."""
from collections import defaultdict,Counter
from round82_relations import relations

def consensus(programs):
    table=None
    for p in programs:
        r=relations(p)
        if r is None:return {}
        table=dict(r) if table is None else {pair:label for pair,label in table.items() if r.get(pair)==label}
        if not table:return {}
    return table or {}

if __name__=='__main__':
    from round22_common import ROOT,read,dump,sha
    out=ROOT/'experiments/round119';out.mkdir();source=read(ROOT/'experiments/round115/fit_sources.json');groups=defaultdict(list);folds={}
    for r in source:groups[r['train_index']].append(r['program']);folds[r['train_index']]=r['fold'];assert r['fold']!=2
    rows=[];stats=Counter()
    for idx,programs in groups.items():
        r=consensus(programs);stats['parents']+=1;stats['supervised_parents']+=bool(r);stats['pairs']+=len(r)
        for (a,b),label in sorted(r.items()):rows.append(dict(train_index=idx,fold=folds[idx],left=a,right=b,label=label))
    old={(r['train_index'],r['left'],r['right']):r['label'] for r in read(ROOT/'experiments/round82/labels.json') if r['fold']!=2}
    for r in rows:
        k=(r['train_index'],r['left'],r['right']);stats['new_pairs']+=k not in old;stats['shared_pairs']+=k in old;stats['shared_disagreements']+=k in old and old[k]!=r['label']
    dump(out/'fit_consensus_labels.json',rows);dump(out/'fit_audit.json',dict(stats=stats,source_sha256=sha(__file__),rule='Intersection across all retained complete lineage alternatives. Any unsupported or reused-source alternative leaves this head unsupervised.',fit_only=True,test_opened=False));print(stats)
