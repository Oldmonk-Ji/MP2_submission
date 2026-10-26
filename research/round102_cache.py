"""Development-only source-program cache and TRAIN-fit representation audit.

Inference never loads these cached TRAIN/VAL programs. It extracts programs
from new inputs through the frozen round51 candidate parser.
"""
from round22_common import ROOT,load,read,dump,sha
from round51_programs import candidate_roots
from round46_graph import graph
from round44_programs import canonical as oldcanon,locations
from round101_templates import canonical
from collections import Counter
import numpy as np,time,hashlib,json
OUT=ROOT/'experiments/round102'

def extract(row):
    g=graph(row['question']);return [list({repr(oldcanon(p)):p for p,k in candidate_roots(row['question'],c['text'],g)}.values()) for c in row['candidates']]

def bag(p):
    return tuple(sorted(Counter(str(n[0])+(':'+str(n[1]) if n[0] in ('N','L') else '') for _,n in locations(p)).items()))

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);tick=time.perf_counter();sources=['round102_cache','round51_programs','round49_programs','round46_graph','round44_programs','round42_relations','round41_relations','round29_programs','round4_quantities','features','round101_templates']
    if not (OUT/'plan.json').exists():dump(OUT/'plan.json',dict(development_cache_only=True,test_opened=False,inference_cache_forbidden=True,source_hashes={str((ROOT/'research'/f'{n}.py').relative_to(ROOT)):sha(ROOT/'research'/f'{n}.py') for n in sources}))
    for split in ('train','validation'):
        b,a,y,ids=load(split)
        if (OUT/f'{split}_programs.json').exists():
            metadata=read(OUT/f'{split}_metadata.json');assert metadata['ids']==ids.tolist();assert metadata['input_sha256']==hashlib.sha256(json.dumps(b,ensure_ascii=False,sort_keys=True).encode()).hexdigest();packets=read(OUT/f'{split}_programs.json');print('REUSE_CACHE',split,flush=True)
        else:
            packets=[]
            for i,r in enumerate(b):
                packets.append(extract(r))
                if i%500==0:print('CACHE',split,i,flush=True)
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:old=z['root']
        counts=np.array([[len(rr) for rr in row] for row in packets]);assert np.allclose(np.log1p(counts),old[:,:,1])
        dump(OUT/f'{split}_programs.json',packets);dump(OUT/f'{split}_metadata.json',dict(ids=ids.tolist(),input_sha256=hashlib.sha256(json.dumps(b,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),n=len(b),program_count=int(counts.sum()),all_candidate_root_counts_match_frozen_cache=True))
        if split!='train':continue
        with np.load(ROOT/'experiments/round56/train_targets.npz') as z:folds=z['folds']
        totals=Counter();examples=[];templates=Counter()
        for i in np.flatnonzero(folds!=2):
            signatures=[];totals['fit_questions']+=1
            for j,roots in enumerate(packets[i]):
                dd={}
                for p in roots:
                    try:key=repr(canonical(p))
                    except ValueError:continue
                    dd[key]=bag(p)
                signatures.append(dd);totals['covered_candidates']+=bool(dd)
                if y[i,j]:templates.update(dd.keys())
            totals['correct_program_covered_questions']+=any(signatures[j] for j in np.flatnonzero(y[i]))
            for j in range(4):
                for k in range(j):
                    if y[i,j]==y[i,k]:continue
                    totals['opposite_label_candidate_pairs']+=1
                    same_full=set(signatures[j])&set(signatures[k]);same_bag=set(signatures[j].values())&set(signatures[k].values())
                    totals['opposite_label_same_full_template']+=bool(same_full);totals['opposite_label_same_operand_operation_bag']+=bool(same_bag)
                    if same_bag and not same_full and len(examples)<10:examples.append(dict(train_index=int(ids[i]),question=b[i]['question'],candidates=[j,k],labels=[float(y[i,j]),float(y[i,k])],programs=[list(signatures[j]),list(signatures[k])]))
        dump(OUT/'fit_audit.json',dict(counts=dict(totals),examples=examples,train_fit_only=True,test_opened=False,positive_template_occurrences=sum(templates.values()),distinct_positive_templates=len(templates)));print('FIT_AUDIT',dict(totals),flush=True)
    dump(OUT/'cache_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
