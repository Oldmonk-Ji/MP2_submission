"""TRAIN-only fixed-budget coverage audit for the arithmetic-evidence correction."""
from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import read,dump,sha,load
from round49_programs import proposals,checks
from round48_local import packet_matrix,LIMIT
from scipy import sparse
import numpy as np,shutil,json,time,datetime

OUT=ROOT/'experiments/round49'

def main():
    OUT.mkdir();old=ROOT/'experiments/round48';plan=read(old/'plan.json')
    plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        objective='Repair a TRAIN-discovered feature parser bug, retaining every previous action and the64action cap; continue the same local policies with matched SFT/exact/GRPO.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research'/f'round49_{n}.py' for n in ('programs','prepare','train','selector','evaluate')]},
        warm_start='Corresponding round48 20epoch warm policy, before its continuation; identical new-bank reference and initialization for all arms.',
        discovery_train_ids=[1172,58,1809,2172,3992,1532,2821,2934],
        parser='Only candidate arithmetic feature extraction changes; official numeric answer parser/scorer unchanged.',
        proposal_coverage_unchanged=False,no_hyperparameter_sweep=True)
    dump(OUT/'plan.json',plan);dump(OUT/'parser_checks.json',checks())
    for name in ('constraint_pairs.json','constraint_parent_ids.npy','constraint_positive.npz','constraint_negative.npz','reviewed_programs.json','held_source_review_before_reference.json','held_execution_checks.json'):
        shutil.copy2(old/name,OUT/name)
    b,_,y,ids=load('train');refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    with np.load(old/'train_targets.npz') as z:folds=z['folds'];base=z['base'];bs=z['base_scores'];oldoracle=z['rewards'].any(1)
    rewards=np.zeros((len(b),LIMIT),np.float32);mask=np.zeros_like(rewards,dtype=bool);matches=np.zeros((len(b),LIMIT,4),bool)
    blocks=[];ptr=[0];start=time.perf_counter()
    with (OUT/'train_proposals.jsonl').open('w',encoding='utf-8') as out:
        for begin in range(0,len(b),64):
            packets=[]
            for k in range(begin,min(begin+64,len(b))):
                p=proposals(b[k]);packets.append(p);gold=parse_answer(refs[int(ids[k])]['answer']);out.write(json.dumps(dict(train_index=int(ids[k]),**p),ensure_ascii=False)+'\n')
                for j,h in enumerate(p['hypotheses']):mask[k,j]=True;matches[k,j,h['matches']]=True;rewards[k,j]=h['answer']==gold and bool(h['matches'])
            x,local=packet_matrix(b[begin:begin+64],packets);blocks.append(x);ptr.extend((local[1:]+ptr[-1]).tolist())
            if begin%512==0:print(dict(parser_prepared=min(begin+64,len(b)),seconds=time.perf_counter()-start),flush=True)
    sparse.save_npz(OUT/'node_features.npz',sparse.vstack(blocks,format='csr'));np.save(OUT/'program_node_ptr.npy',np.asarray(ptr,np.int64))
    np.savez(OUT/'train_targets.npz',rewards=rewards,mask=mask,matches=matches,folds=folds,ids=ids,labels=y,base=base,base_scores=bs)
    oracle=rewards.any(1);assert np.all(oracle>=oldoracle);baseok=y[np.arange(len(y)),base].astype(bool);stats={}
    for name,ix in [('all',np.ones(len(y),bool)),('held',folds==2),('held_baseline_errors',(folds==2)&~baseok),('fit_baseline_errors',(folds!=2)&~baseok)]:
        stats[name]=dict(n=int(ix.sum()),previous=int(oldoracle[ix].sum()),correct_program=int(oracle[ix].sum()),new=int((oracle&~oldoracle&ix).sum()))
    result=dict(coverage=stats,training_n=len(b),local_relations=ptr[-1],constraint_pairs=706,locally_distinguishable_pairs=read(old/'preparation_result.json')['locally_distinguishable_pairs'],seconds=time.perf_counter()-start,test_opened=False)
    dump(OUT/'preparation_result.json',result);dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='prepared_freeze.json'}));print(result,flush=True)

if __name__=='__main__':main()
