"""Does arithmetic contrast improvement transfer to natural candidates? TRAIN held groups only."""
from data_access import ROOT
from round22_common import read,dump
from round29_programs import encode_questions,encode_programs
from round29_model import predict_encoded
from round30_model import load
import json,numpy as np,torch


def main():
    rows=[json.loads(s) for s in open(ROOT/'experiments/round31/training_programs.jsonl',encoding='utf-8')]
    with np.load(ROOT/'experiments/round31/diagnostic_split.npz') as z:held=z['held']
    rows=[rows[i] for i in held];mask=np.asarray([r['contrast_mask'] for r in rows],bool)
    programs=encode_programs([p for r in rows for p in [r['reference']]+r['contrast_negatives']]).reshape(len(rows),4,-1)
    natural=encode_programs([p for r in rows for p in r['candidates']]).reshape(len(rows),4,-1)
    labels=np.asarray([r['labels'] for r in rows]);device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    records=[]
    for number,variant in [(30,'reference_and_candidate_ranking'),(31,'semantic_contrast')]:
        model,vocab=load(ROOT/f'experiments/round{number}/diagnostic/{variant}/epoch40');model.to(device)
        q=encode_questions(rows,vocab,256);s,_=predict_encoded(model,q,programs,device);ns,_=predict_encoded(model,q,natural,device)
        comparison=(s[:,:1]>s[:,1:]);strict=mask.any(1)&np.all(comparison|~mask,1)
        rec=dict(round=number,epoch=40,train_fit_n=3795,held_n=len(rows),
            contrast_questions=int(mask.any(1).sum()),negative_comparisons=int(mask.sum()),
            positive_above_negative=int((comparison&mask).sum()),positive_above_all_negatives=int(strict.sum()),
            natural_candidates_correct=int(labels[np.arange(len(rows)),ns.argmax(1)].sum()),
            caveat='Synthetic contrast ranking and standalone natural candidate ranking, not baseline blend accuracy and not TEST.')
        records.append(rec);print(rec,flush=True)
    dump(ROOT/'experiments/round31/contrast_transfer_diagnostic.json',dict(records=records,test_opened=False))


if __name__=='__main__':main()
