"""TRAIN diagnostic then full fit with executable-program auxiliary labels."""
from data_access import ROOT
from round22_common import load as load_bank,read,dump
from round29_programs import vocabulary,encode_questions,encode_programs
from round29_model import ProgramModel,trim,save,predict_encoded,centered
import numpy as np
import torch
from torch import nn
import argparse,json,time

OUT=ROOT/'experiments/round29'


def dataset():
    rows=[json.loads(s) for s in (OUT/'training_programs.jsonl').read_text(encoding='utf-8').split('\n') if s.strip()]
    return rows,np.asarray([r['labels'] for r in rows],np.float32)


@torch.no_grad()
def evaluate(model,q,ref,programs,y,usable,weights,device):
    model.eval();losses=[];tokenok=tokenn=exact=n=0
    for start in range(0,len(q),64):
        qt=torch.as_tensor(trim(q[start:start+64]),device=device);p=torch.as_tensor(trim(ref[start:start+64]),device=device)
        loss,_,pred=model(qt,p);losses.extend(loss.cpu().tolist());mask=p.ne(0)
        tokenok+=int(((pred==p)&mask).sum());tokenn+=int(mask.sum());exact+=int(((pred==p)|~mask).all(1).sum());n+=len(p)
    s,_=predict_encoded(model,q,programs,device)
    choice=centered(s,usable).argmax(1)
    eligible=(usable.sum(1)>=2)&((y*usable).sum(1)>0)&((1-y).sum(1)>0)
    return dict(reference_nll=float(np.average(losses,weights=weights)),reference_teacher_forced_token_accuracy=tokenok/max(1,tokenn),
        reference_teacher_forced_exact=exact,reference_n=n,candidate_correct=int(y[np.arange(len(y)),choice].sum()),
        n=len(y),eligible_comparisons=int(eligible.sum()),eligible_correct=int(y[np.arange(len(y)),choice][eligible].sum()))


def train(stage,variant):
    torch.set_num_threads(4);torch.manual_seed(29029);rng=np.random.default_rng(29029)
    rows,y=dataset();ids=np.arange(len(rows));held=None
    if stage=='diagnostic':
        with np.load(OUT/'diagnostic_split.npz') as z:ids=z['fit'];held=z['held']
    path=OUT/stage/variant;path.mkdir(parents=True);assert not (path/'complete.json').exists()
    bank=[dict(question=r['question']) for r in rows];vocab=vocabulary([bank[i] for i in ids])
    q=encode_questions(bank,vocab,max_length=256)
    ref=encode_programs([r['reference'] for r in rows])
    programs=encode_programs([c for r in rows for c in r['candidates']]).reshape(len(rows),4,-1)
    usable=np.asarray([[c['clean'] for c in r['candidates']] for r in rows])
    weights=np.asarray([1. if r['reference']['clean'] else (.25 if r['reference']['informative'] else .05) for r in rows],np.float32)
    dump(path/'vocabulary.json',vocab);dump(path/'training_ids.json',[rows[i]['train_index'] for i in ids])
    device='cuda' if torch.cuda.is_available() else 'cpu';model=ProgramModel(len(vocab)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    maxep=40 if stage=='diagnostic' else 80;history=[];tick=time.perf_counter()
    for epoch in range(1,maxep+1):
        model.train();total=0.;count=0
        for ix in np.array_split(rng.permutation(ids),int(np.ceil(len(ids)/64))):
            qt=torch.as_tensor(trim(q[ix]),device=device);rt=torch.as_tensor(trim(ref[ix]),device=device)
            w=torch.as_tensor(weights[ix],device=device);memory=model.encode(qt)
            nll,_,_=model(qt,rt,memory);loss=(nll*w).sum()/w.sum()
            if variant=='reference_and_candidate_ranking':
                cp=programs[ix];cp=torch.as_tensor(trim(cp.reshape(-1,cp.shape[-1])),device=device)
                _,cs,_=model(qt,cp,memory);cs=cs.reshape(-1,4);target=torch.as_tensor(y[ix],device=device)
                ranking=(torch.logsumexp(cs,1)-torch.logsumexp(cs.masked_fill(target<.5,-1e4),1)).mean()
                loss=loss+.5*ranking
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
            total+=float(loss.detach())*len(ix);count+=len(ix)
        rec=dict(epoch=epoch,training_loss=total/count,seconds=time.perf_counter()-tick)
        if held is not None and epoch in (5,15,25,40):rec['diagnostic']=evaluate(model,q[held],ref[held],programs[held],y[held],usable[held],weights[held],device)
        if epoch%5==0:print(stage,variant,rec,flush=True)
        history.append(rec)
        if epoch in (15,40,80):
            dest=path/f'epoch{epoch}';dest.mkdir();save(model,dest/'model.npz');dump(dest/'vocabulary.json',vocab)
            dump(dest/'config.json',dict(variant=variant,epochs=epoch,training_n=len(ids),word_width=64,decoder_width=96,seed=29029))
        dump(path/'history.json',history)
    dump(path/'complete.json',dict(n=len(ids),seconds=time.perf_counter()-tick,device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
        parameters=sum(p.numel() for p in model.parameters()),test_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['diagnostic','full']);p.add_argument('variant',choices=['reference_only','reference_and_candidate_ranking']);a=p.parse_args();train(a.stage,a.variant)
