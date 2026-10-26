"""Reference-aligned positive programs versus actual incorrect candidate traces."""
from round22_common import ROOT,load,read,dump,sha
from round51_programs import candidate_roots,value
from round46_graph import graph
from round44_programs import tree,locations,canonical
from round65_local import focus,csr
from features import parse_answer
from round50_selector import select
from scipy import sparse
import numpy as np,shutil,sys
import round48_train as training
OUT=ROOT/'experiments/round67'

def prepare():
    OUT.mkdir();source=ROOT/'experiments/round65'
    for name in ('train_targets.npz','node_features.npz','program_node_ptr.npy','held_source_review_before_reference.json'):
        shutil.copy2(source/name,OUT/name)
    b,_,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)}
    pairs=read(ROOT/'experiments/round56/constraint_pairs.json');original=len(pairs);covered=set();seen=set();rejected=0
    for r in read(ROOT/'experiments/round63/accepted_reference_programs.json'):
        i=index[r['train_index']];row=b[i];q=row['question'];g=graph(q)
        for j,c in enumerate(row['candidates']):
            if y[i,j] or parse_answer(c['text']) is None:continue
            roots={repr(canonical(p)):canonical(p) for p,kind in candidate_roots(q,c['text'],g) if len(list(locations(p)))<=31}
            for key in sorted(roots,key=lambda s:(len(s),s))[:3]:
                neg=roots[key];identity=(r['train_index'],key)
                if identity in seen:continue
                assert str(value(neg,g['quantities']))==parse_answer(c['text'])
                seen.add(identity);covered.add(r['train_index'])
                pairs.append(dict(train_index=r['train_index'],fold=r['fold'],question=q,positive=r['program'],negative=neg,kind='real_candidate_error',
                    provenance='reference_trace_positive_actual_candidate_wrong_final_negative',candidate_index=j,candidate_text=c['text'],reference_evidence=r['reference_evidence']))
    dump(OUT/'constraint_pairs.json',pairs);np.save(OUT/'constraint_parent_ids.npy',np.asarray([p['train_index'] for p in pairs]))
    ff=[focus(p) for p in pairs];sparse.save_npz(OUT/'constraint_positive.npz',csr([p[0] for p in ff]));sparse.save_npz(OUT/'constraint_negative.npz',csr([p[1] for p in ff]))
    dump(OUT/'supervision_stats.json',dict(original_reviewed_pairs=original,real_candidate_pairs=len(pairs)-original,real_candidate_parents=len(covered),synthetic_reference_mutants_replaced=True,test_opened=False))
    plan=read(source/'plan.json');plan.update(objective='Matched round65 whole program model, replace synthetic reference mutants with actual incorrect candidate traces; keep original reviewed pairs',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round67*.py')})
    dump(OUT/'plan.json',plan);dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()}));print(read(OUT/'supervision_stats.json'),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
    else:training.OUT=OUT;training.DIM=2048;training.select=select;training.main()
