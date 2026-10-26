"""Freeze fit-derived rules, then evaluate complete TRAIN coverage once."""
from data_access import ROOT
from round22_common import read,dump,sha,load
from round51_programs import proposals,checks,value
from round52_local import packet_matrix,focus,csr,LIMIT
from scipy import sparse
import numpy as np,json,time,datetime,hashlib,shutil

OUT=ROOT/'experiments/round51'

def main():
    OUT.mkdir();b,_,y,ids=load('train');old=ROOT/'experiments/round50'
    with np.load(old/'train_targets.npz') as z:
        folds=z['folds'];base=z['base'];bs=z['base_scores'];prior=z['rewards'].any(1)
    fitmissing=np.flatnonzero((folds!=2)&~y[np.arange(len(y)),base].astype(bool)&~prior)
    discovery=sorted(fitmissing,key=lambda i:hashlib.sha256(f'round51:{ids[i]}'.encode()).digest())[:30]
    sources=[ROOT/'research'/n for n in ('round51_programs.py','round51_prepare.py','round52_local.py','round52_train.py','round52_selector.py','round52_evaluate.py')]
    plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),discovery_ids=ids[discovery].tolist(),fit_only_rule_design=True,
        held_rules_not_designed_from_text=True,program_cap=64,new_programs_are_hypotheses_not_gold=True,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},test_opened=False,
        rule_types=['source arithmetic DAG with unit normalization','linear equation','contextual time conversions','integer floor/ceiling/strict bound','one omitted source-operand arithmetic intermediate'],
        relation_features='Existing1024 lexical plus1024 identity-independent source/target, unit, rate, owner equality, state and scope features',
        seeds=[48048,48049],arms=['sft','exact','grpo'],warm_epochs=20,continuation_epochs=10,
        held_diagnostics='Fixed historical fold2. No new held process labels used in training.',
        test_protocol='After TRAIN/VAL selection, freeze best NEW feature method then one full official TEST report only. Never select incumbent from TEST.')
    dump(OUT/'plan.json',plan);dump(OUT/'parser_checks.json',checks())
    for name in ('constraint_pairs.json','constraint_parent_ids.npy','held_source_review_before_reference.json','held_execution_checks.json','reviewed_programs.json'):
        shutil.copy2(old/name,OUT/name)
    pairs=read(OUT/'constraint_pairs.json');ff=[focus(p) for p in pairs]
    sparse.save_npz(OUT/'constraint_positive.npz',csr([p[0] for p in ff]));sparse.save_npz(OUT/'constraint_negative.npz',csr([p[1] for p in ff]))
    rewards=np.zeros((len(b),LIMIT),np.float32);raw=np.zeros_like(rewards,bool);matches=np.zeros((len(b),LIMIT,4),bool)
    blocks=[];ptr=[0];start=time.perf_counter();action_counts={}
    with (OUT/'train_proposals.jsonl').open('w',encoding='utf-8') as f:
        for begin in range(0,len(b),48):
            rows=b[begin:begin+48];packets=[]
            for k,r in enumerate(rows,begin):
                p=proposals(r);packets.append(p);f.write(json.dumps(dict(train_index=int(ids[k]),**p),ensure_ascii=False)+'\n')
                for j,h in enumerate(p['hypotheses']):
                    assert str(value(h['program'],p['quantities']))==h['answer']
                    raw[k,j]=True;matches[k,j,h['matches']]=True;rewards[k,j]=any(y[k,t] for t in h['matches']);action_counts[h['action']]=action_counts.get(h['action'],0)+1
            x,pp=packet_matrix(rows,packets);blocks.append(x);ptr.extend((pp[1:]+ptr[-1]).tolist())
            if begin%480==0:print('PREPARED',begin+len(rows),'seconds',time.perf_counter()-start,flush=True)
    sparse.save_npz(OUT/'node_features.npz',sparse.vstack(blocks,format='csr'));np.save(OUT/'program_node_ptr.npy',np.asarray(ptr,np.int64))
    mask=raw&matches.any(2);np.savez(OUT/'train_targets.npz',rewards=rewards,mask=mask,raw_program_mask=raw,matches=matches,folds=folds,ids=ids,labels=y,base=base,base_scores=bs)
    oracle=rewards.any(1);assert np.all(oracle>=prior);ok=y[np.arange(len(y)),base].astype(bool);stats={}
    for name,ix in [('all',np.ones(len(y),bool)),('held',folds==2),('held_baseline_errors',(folds==2)&~ok),('fit_baseline_errors',(folds!=2)&~ok)]:
        stats[name]=dict(n=int(ix.sum()),previous=int(prior[ix].sum()),correct_program=int(oracle[ix].sum()),new=int((ix&oracle&~prior).sum()))
    dump(OUT/'preparation_result.json',dict(coverage=stats,programs=int(raw.sum()),legal_programs=int(mask.sum()),rows_without_actions=int((~mask.any(1)).sum()),nodes=ptr[-1],seconds=time.perf_counter()-start,action_counts=action_counts,test_opened=False))
    dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()}));print(stats,flush=True)

if __name__=='__main__':main()
