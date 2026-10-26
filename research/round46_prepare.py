"""Freeze protocol, derive source-grounded process pairs, prepare TRAIN only."""
from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import load,read,dump,sha
from round46_graph import proposals,matrix,features,csr,LIMIT,DIM,graph
from round44_programs import edits,valid,canonical,tree,value
from round14_prepare import family
from scipy import sparse
import numpy as np,json,datetime,time
from collections import Counter

OUT=ROOT/'experiments/round46'

def main():
    assert not (OUT/'plan.json').exists()
    sources=[ROOT/'research'/name for name in ('round46_annotations.py','round46_graph.py','round46_prepare.py','round46_train.py','round46_selector.py','round46_evaluate.py')]
    dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        objective='More source-reviewed relations and executable coverage before matched SFT/exact-reward/GRPO comparison.',
        discovery='60 deterministic historical OOF errors from folds0+1; 54 source programs,6 uncertain/unsupported. No extra training exclusion.',
        features='2048-dimensional hashed source quantity/relation graph with AST paths, typed and time-state edges; lexical hypotheses, not certified semantic parser.',
        actions='Retain every old up-to32 action, append bounded candidate arithmetic DAG and question scope/rate proposals; total64. No reference-derived program insertion.',
        seeds=[46046,46047],warm_epochs=20,continuation_epochs=10,updates_per_rollout=2,batch_size=32,
        arms=['sft','exact','grpo'],learning_rate=.0003,weight_decay=.01,kl=.02,relation_weight=.2,group_size=16,clip=.2,
        control='Frozen round25 full validation and historical grouped OOF baseline for TRAIN diagnostics.',
        diagnostic='Fixed historical grouped fold2 held, folds0+1 fit. Expanded process labels all from fit folds. TRAIN has been explored previously; not pristine.',
        final='Every model fits all4743 allowed TRAIN; held process labels excluded from diagnostic gradients.',
        validation='Complete798 selects among fixed residual .25 modes, correct then fewer harms; prefer incumbent on ties. Also report355conflict. No hyperparameter sweep.',
        executor='Exact rational arithmetic, final executed answer maps to one of original four candidates; unknown/no match uses baseline.',
        scope='Finite categorical GRPO, not LLM fine-tuning or arbitrary reasoning generation.',
        resources=dict(asset_MiB=16,reference_cpu_seconds=30,threads=4),
        test_opened=False,external_data=False,external_pretrained_weights=False,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    b,a,y,ids=load('train')
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:
        assert np.array_equal(ids,z['ids']);folds=z['folds'];base=z['choices'];bs=z['scores'];baseok=z['correct']
    assert not {family(r['question']) for r,f in zip(b,folds) if f!=2}&{family(r['question']) for r,f in zip(b,folds) if f==2}
    byid={int(i):r for i,r in zip(ids,b)};foldmap=dict(zip(map(int,ids),map(int,folds)))
    reviewed=read(ROOT/'experiments/round44/reviewed_programs.json')
    additions=read(OUT/'discovery_review_before_reference.json')
    assert all(c['execution_agrees'] for c in read(OUT/'discovery_execution_checks.json'))
    for r in additions:
        if r['status']=='source_reviewed':reviewed.append(dict(train_index=r['train_index'],question=r['question'],program=r['program'],answer=r['source_derived_answer'],provenance=r['adjudication'],relation_type=r['relation_type']))
    assert len({r['train_index'] for r in reviewed})==len(reviewed)
    pairs=[]
    for r in reviewed:
        q=r['question'];qs=graph(q)['quantities'];positive=tree(r['program']);assert q==byid[r['train_index']]['question']
        assert str(value(positive,qs))==r['answer']
        seen=set();counts=Counter()
        for action,p,decision in edits(positive,qs):
            key=repr(canonical(p));a=valid(p,qs)
            if key in seen or a is None or a==r['answer'] or counts[action]>=4:continue
            counts[action]+=1;seen.add(key)
            pairs.append(dict(train_index=r['train_index'],fold=foldmap[r['train_index']],question=q,positive=positive,negative=p,
                action=action,decision=decision,provenance='Source-reviewed positive; one executable mutation violates TRAIN final answer. Not a claim every mutated edge is independently human labeled.'))
    dump(OUT/'reviewed_programs.json',reviewed);dump(OUT/'constraint_pairs.json',pairs)
    sparse.save_npz(OUT/'constraint_positive.npz',csr([features(p['question'],graph(p['question'])['quantities'],p['positive']) for p in pairs]))
    sparse.save_npz(OUT/'constraint_negative.npz',csr([features(p['question'],graph(p['question'])['quantities'],p['negative']) for p in pairs]))
    np.save(OUT/'constraint_parent_ids.npy',[p['train_index'] for p in pairs])
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');rewards=np.zeros((len(b),LIMIT),np.float32)
    mask=np.zeros_like(rewards,dtype=bool);matches=np.zeros((len(b),LIMIT,4),bool);counts=[];blocks=[];start=time.perf_counter()
    with (OUT/'train_proposals.jsonl').open('w',encoding='utf-8') as out:
        for begin in range(0,len(b),64):
            packets=[]
            for k in range(begin,min(begin+64,len(b))):
                p=proposals(b[k]);gold=parse_answer(refs[int(ids[k])]['answer']);packets.append(p);counts.append(len(p['hypotheses']))
                out.write(json.dumps(dict(train_index=int(ids[k]),**p),ensure_ascii=False)+'\n')
                for j,h in enumerate(p['hypotheses']):
                    mask[k,j]=True;matches[k,j,h['matches']]=True;rewards[k,j]=h['answer']==gold and bool(h['matches'])
            blocks.append(matrix(b[begin:begin+64],packets))
            if begin%512==0:print(dict(prepared=min(begin+64,len(b)),seconds=round(time.perf_counter()-start,1)),flush=True)
    sparse.save_npz(OUT/'train_features.npz',sparse.vstack(blocks,format='csr'))
    np.savez(OUT/'train_targets.npz',rewards=rewards,mask=mask,matches=matches,folds=folds,ids=ids,labels=y,base=base,base_scores=bs)
    with np.load(ROOT/'experiments/round44/train_targets.npz') as z:old_oracle=z['rewards'].any(1)
    oracle=rewards.any(1);assert np.all(oracle>=old_oracle)
    strata={}
    for name,ix in [('all',np.ones(len(ids),bool)),('held',folds==2),('held_baseline_errors',(folds==2)&~baseok),('fit_baseline_errors',(folds!=2)&~baseok)]:
        strata[name]=dict(n=int(ix.sum()),old_correct_program=int(old_oracle[ix].sum()),new_correct_program=int(oracle[ix].sum()),newly_covered=int((oracle&~old_oracle&ix).sum()))
    result=dict(training_n=len(ids),source_reviewed_parents=len(reviewed),new_discovery_reviewed=60,new_source_programs=54,
        pairs=len(pairs),actions=dict(Counter(p['action'] for p in pairs)),
        fit_pair_parents=len({p['train_index'] for p in pairs if p['fold']!=2}),held_pair_parents=len({p['train_index'] for p in pairs if p['fold']==2}),
        coverage=strata,mean_proposals=float(np.mean(counts)),seconds=time.perf_counter()-start,test_opened=False)
    dump(OUT/'preparation_result.json',result)
    dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='prepared_freeze.json'}))
    print(result,flush=True)

if __name__=='__main__':main()
