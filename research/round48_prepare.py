"""Freeze one compositional alternative; identical candidates and source labels."""
from data_access import ROOT
from round22_common import read,dump,sha
from round48_local import packet_matrix,csr,focus
from scipy import sparse
import numpy as np,shutil,json,time,datetime

OUT=ROOT/'experiments/round48'

def main():
    OUT.mkdir();old=ROOT/'experiments/round47'
    plan=read(old/'plan.json');plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        objective='Learn local relation decisions directly; fixed soft-min composes executable program scores. Same proposals and supervised parents as round47.',
        seeds=[48048,48049],features='1024-dimensional source-grounded local arithmetic relation,32 hidden units; no ID or final numeric value.',
        local_supervision='Focus at mutated relation, or parent of changed source leaf; old whole-program margin is replaced by local relation margin.',
        proposal_coverage_unchanged=True,source_reviewed_held_used_for_training=False,
        validation='Exactly three prespecified arms, complete798; no gate/alpha/epoch search; prior round47 scores already known. No test access.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research'/f'round48_{n}.py' for n in ('local','selector','prepare','train','evaluate')]})
    dump(OUT/'plan.json',plan)
    for name in ('train_targets.npz','constraint_pairs.json','constraint_parent_ids.npy','reviewed_programs.json','held_source_review_before_reference.json','held_execution_checks.json'):
        shutil.copy2(old/name,OUT/name)
    pairs=read(OUT/'constraint_pairs.json');positive=[];negative=[];focusrows=[]
    for p in pairs:
        a,b,path=focus(p);positive.append(a);negative.append(b);focusrows.append(dict(train_index=p['train_index'],path=path,features_different=a!=b))
    sparse.save_npz(OUT/'constraint_positive.npz',csr(positive));sparse.save_npz(OUT/'constraint_negative.npz',csr(negative));dump(OUT/'relation_focus_audit.json',focusrows)
    blocks=[];ptr=[0];start=time.perf_counter();rows=[];packets=[];n=0
    with (ROOT/'experiments/round46/train_proposals.jsonl').open(encoding='utf-8') as f:
        from round22_common import load
        bank,_,_,ids=load('train')
        for k,line in enumerate(f):
            p=json.loads(line);assert p['train_index']==int(ids[k]);rows.append(bank[k]);packets.append(p)
            if len(rows)==64 or k==len(bank)-1:
                x,local=packet_matrix(rows,packets);blocks.append(x);ptr.extend((local[1:]+ptr[-1]).tolist());n+=len(rows);rows=[];packets=[]
                if n%512==0:print(dict(local_prepared=n,seconds=time.perf_counter()-start),flush=True)
    sparse.save_npz(OUT/'node_features.npz',sparse.vstack(blocks,format='csr'));np.save(OUT/'program_node_ptr.npy',np.asarray(ptr,np.int64))
    dump(OUT/'preparation_result.json',dict(training_n=len(bank),local_relations=ptr[-1],constraint_pairs=len(pairs),
        locally_distinguishable_pairs=sum(r['features_different'] for r in focusrows),seconds=time.perf_counter()-start,test_opened=False))
    dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='prepared_freeze.json'}))
    print(read(OUT/'preparation_result.json'),flush=True)

if __name__=='__main__':main()
