"""Root-cause study of genuine held-out TRAIN errors before new model design."""
from data_access import ROOT,CODE,read_jsonl
from round22_common import load,read,dump,sha
from round22_selector import normalized
from round4_gates import apply_gate
from round14_prepare import family
import numpy as np,json,hashlib,datetime

OUT=ROOT/'experiments/round38'


def main():
    OUT.mkdir();b,a,y,ids=load('train');refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    with np.load(ROOT/'experiments/round22/oof.npz') as z:
        assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['labels']);components=z['components']
    weights=np.asarray(read(ROOT/'experiments/round22/selected_bundle/readout.json')['base_weights'])
    s=normalized(components[:,:,:5])@weights;c=apply_gate(b,s,'parse_and_repair',4);ok=y[np.arange(len(b)),c].astype(bool)
    folds={}
    for fold in range(3):
        p=read(ROOT/f'experiments/round22/oof/fold{fold}/partition.json');fit=set(p['fit_ids']);held=set(p['held_ids'])
        assert not fit&held
        assert not {family(b[k]['question']) for k,i in enumerate(ids) if int(i) in fit}&{family(b[k]['question']) for k,i in enumerate(ids) if int(i) in held}
        for i in held:folds[i]=fold
    errors=[k for k in range(len(b)) if not ok[k]]
    # Fold 0 discoveries may inform the schema. Other folds are not shown now.
    ordered=sorted([k for k in errors if folds[int(ids[k])]==0],key=lambda k:hashlib.sha256(f'root38:{int(ids[k])}'.encode()).hexdigest())
    selected=ordered[:24];dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training_n=4743,validation_n=798,test_opened=False,
        source='round22 grouped out-of-fold experts; fixed historical five-member weights and arithmetic gate, not in-sample predictions from final round25',
        historical_exploration_caveat='TRAIN and historical baseline recipes have been studied before. These are held-out expert predictions, not a pristine independent benchmark.',
        first_discovery_ids=[int(ids[k]) for k in selected],discovery_fold=0,
        root_error_count=len(errors),fold_error_counts={str(f):sum(folds[int(ids[k])]==f for k in errors) for f in range(3)},
        annotation='Read original question, wrong chosen reasoning and supplied reference; record exact evidence, explicit versus derived facts, candidate claims, uncertainty, and causal link to the wrong final answer. No numeric-match-only relation labels.',
        validation_not_used_for_root_cause_or_relation_labels=True,
        next_step='Choose a semantic subtask only after this discovery audit. Freeze its schema before new fold 1/2 natural relation evaluation.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'experiments/round22/oof.npz',ROOT/'experiments/round22/selected_bundle/readout.json']}))
    np.savez(OUT/'oof_baseline.npz',ids=ids,choices=c,scores=s,correct=ok,folds=np.asarray([folds[int(i)] for i in ids]))
    all_errors=[]
    for k in errors:
        row=dict(train_index=int(ids[k]),fold=folds[int(ids[k])],question=b[k]['question'],chosen=int(c[k]),labels=y[k].astype(int).tolist(),
            candidates=[cc['text'] for cc in b[k]['candidates']],reference=refs[int(ids[k])]['answer'])
        all_errors.append(row)
    (OUT/'all_train_oof_errors.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_errors),encoding='utf-8')
    lookup={r['train_index']:r for r in all_errors};dump(OUT/'discovery24.json',[lookup[int(ids[k])] for k in selected])
    print(dict(oof_errors=len(errors),selected=[int(ids[k]) for k in selected]),flush=True)
    for k in selected[:12]:
        r=lookup[int(ids[k])];print(json.dumps(dict(train_index=r['train_index'],question=r['question'],chosen=r['chosen'],wrong_candidate=r['candidates'][r['chosen']],reference=r['reference']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
