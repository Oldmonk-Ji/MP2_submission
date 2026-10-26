from data_access import ROOT
from round22_common import load,read,dump,sha
from round56_local import packet_matrix,focus,csr
from scipy import sparse
import numpy as np,json,shutil
import round48_train as training
from round50_selector import select
OUT=ROOT/'experiments/round56'
if __name__=='__main__':
    OUT.mkdir();source=ROOT/'experiments/round52'
    for name in ('train_targets.npz','constraint_pairs.json','constraint_parent_ids.npy','held_source_review_before_reference.json'):
        shutil.copy2(source/name,OUT/name)
    plan=read(source/'plan.json');plan.update(arms=['sft'],objective='Recursive intermediate dimension and owner propagation; matched round52 SFT budget; no new labels or program rules',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round56*.py')})
    dump(OUT/'plan.json',plan)
    b,_,_,ids=load('train');packets=[json.loads(s) for s in (ROOT/'experiments/round51/train_proposals.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [p['train_index'] for p in packets]==ids.tolist()
    blocks=[];ptr=[0]
    for start in range(0,len(b),64):
        x,p=packet_matrix(b[start:start+64],packets[start:start+64]);blocks.append(x);ptr.extend((p[1:]+ptr[-1]).tolist())
        if start%512==0:print('PREPARED',start,flush=True)
    sparse.save_npz(OUT/'node_features.npz',sparse.vstack(blocks));np.save(OUT/'program_node_ptr.npy',np.asarray(ptr))
    ff=[focus(p) for p in read(OUT/'constraint_pairs.json')]
    for i,name in enumerate(('positive','negative')):sparse.save_npz(OUT/f'constraint_{name}.npz',csr([p[i] for p in ff]))
    dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()}))
    training.OUT=OUT;training.DIM=2048;training.select=select;training.main()
