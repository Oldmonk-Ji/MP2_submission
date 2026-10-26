from round22_common import ROOT,load,read,dump,sha
from round65_local import packet_matrix,focus,csr
from round50_selector import select
from scipy import sparse
import numpy as np,json,shutil
import round48_train as training
OUT=ROOT/'experiments/round65'
if __name__=='__main__':
    assert not (OUT/'plan.json').exists();source=ROOT/'experiments/round63'
    for name in ('train_targets.npz','constraint_pairs.json','constraint_parent_ids.npy','held_source_review_before_reference.json'):
        shutil.copy2(source/name,OUT/name)
    plan=read(source/'plan.json');plan.update(objective='Matched round63 supervision/programs/dimension/width/budget; whole program graph instead of local soft-min node aggregation',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round65*.py')})
    dump(OUT/'plan.json',plan)
    b,_,_,ids=load('train');packets=[json.loads(s) for s in (ROOT/'experiments/round51/train_proposals.jsonl').read_text(encoding='utf-8').split('\n') if s.strip()]
    assert [p['train_index'] for p in packets]==ids.tolist()
    blocks=[];ptr=[0]
    for begin in range(0,len(b),64):
        x,p=packet_matrix(b[begin:begin+64],packets[begin:begin+64]);blocks.append(x);ptr.extend((p[1:]+ptr[-1]).tolist())
        if begin%512==0:print('PREPARED',begin,flush=True)
    sparse.save_npz(OUT/'node_features.npz',sparse.vstack(blocks));np.save(OUT/'program_node_ptr.npy',np.asarray(ptr))
    ff=[focus(p) for p in read(OUT/'constraint_pairs.json')];xp=csr([p[0] for p in ff]);xn=csr([p[1] for p in ff])
    sparse.save_npz(OUT/'constraint_positive.npz',xp);sparse.save_npz(OUT/'constraint_negative.npz',xn)
    d=xp-xn;d.eliminate_zeros();dump(OUT/'whole_alias_audit.json',dict(n=len(ff),identical=int((np.diff(d.indptr)==0).sum())))
    dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()}))
    training.OUT=OUT;training.DIM=2048;training.select=select;training.main()
