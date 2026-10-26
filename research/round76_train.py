from round22_common import ROOT,load,read,dump,sha
from round76_text import candidate_sentences,sentence,vocab,encode
from round76_model import Model
from round76_selector import score_tokens
from round22_selector import normalized
from round25_selector import scores as baseline
from round4_gates import apply_gate
import torch,numpy as np,time,shutil
from torch import nn
OUT=ROOT/'experiments/round76'

def fit(sequences,ys,bs,ids,pairs,path,ordered):
    path.mkdir(parents=True);torch.manual_seed(76076);rng=np.random.default_rng(76076)
    seq=[s for q in sequences for s in q];pos=[sentence(p['question'],p['positive']) for p in pairs];neg=[sentence(p['question'],p['negative']) for p in pairs]
    vocabulary=vocab(seq+pos+neg);dump(path/'vocabulary.json',vocabulary);dump(path/'config.json',dict(ordered=ordered))
    xx,seg=encode(seq,vocabulary);px,ps=encode(pos,vocabulary);nx,ns=encode(neg,vocabulary)
    xx=torch.tensor(xx,device='cuda');seg=torch.tensor(seg,device='cuda');px=torch.tensor(px,device='cuda');ps=torch.tensor(ps,device='cuda');nx=torch.tensor(nx,device='cuda');ns=torch.tensor(ns,device='cuda')
    yy=torch.tensor(ys,device='cuda');base=torch.tensor(bs,dtype=torch.float32,device='cuda');m=Model(len(vocabulary),ordered).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=.01);history=[]
    start=time.perf_counter()
    for epoch in range(20):
        m.train();total=0
        for ix in np.array_split(rng.permutation(len(ids)),int(np.ceil(len(ids)/32))):
            rows=(ix[:,None]*4+np.arange(4)).reshape(-1);r=m(xx[rows],seg[rows]).reshape(-1,4);s=base[ix]+r;t=yy[ix]
            loss=(torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(t<.5,-1e9),1)).mean()+.01*r.square().mean()
            pi=rng.choice(len(pairs),16,replace=False);aux=nn.functional.softplus(m(nx[pi],ns[pi])-m(px[pi],ps[pi])).mean();loss=loss+.1*aux
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();total+=float(loss.detach())*len(ix)
        history.append(total/len(ids))
        if (epoch+1)%5==0:print('EPOCH',path.parent.name,path.name,epoch+1,'loss',history[-1],'seconds',time.perf_counter()-start,flush=True)
    m.eval();np.savez(path/'model.npz',**{k:v.detach().cpu().numpy() for k,v in m.state_dict().items()})
    dump(path/'training.json',dict(training_ids=ids.tolist(),constraint_parents=sorted({p['train_index'] for p in pairs}),history=history,seconds=time.perf_counter()-start))
    with torch.no_grad():expected=m(xx[:8],seg[:8]).cpu().numpy()
    actual=score_tokens(seq[:8],path);assert np.allclose(actual,expected,atol=2e-4)

if __name__=='__main__':
    OUT.mkdir();torch.set_num_threads(4);tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['positionless','ordered'],epochs=20,seed=76076,reference_weight=.1,residual_penalty=.01,question_tokens=128,program_tokens=96,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round76*.py')}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');tokens={}
    for split,rows in [('train',b),('validation',v)]:
        ts=[]
        for i,r in enumerate(rows):
            ts.append(candidate_sentences(r))
            if i%500==0:print('TOKENIZED',split,i,flush=True)
        tokens[split]=ts;dump(OUT/f'{split}_tokens.json',ts)
    weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']]
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(ids,z['ids']);bs=normalized(z['components'])@weights
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    pairs=read(ROOT/'experiments/round67/constraint_pairs.json');vbs=baseline(v,va,ROOT/'experiments/round25/selected_bundle');bc=apply_gate(v,vbs,'parse_and_repair',4);records=[]
    for ordered in (False,True):
        arm='ordered' if ordered else 'positionless'
        for stage in ('diagnostic','full'):
            ix=np.flatnonzero(f!=2) if stage=='diagnostic' else np.arange(len(b));keep=set(map(int,ids[ix]));pp=[p for p in pairs if p['train_index'] in keep]
            path=OUT/stage/arm;fit([tokens['train'][i] for i in ix],y[ix],bs[ix],ids[ix],pp,path,ordered)
            if stage=='diagnostic':
                held=np.flatnonzero(f==2);rr=score_tokens([s for i in held for s in tokens['train'][i]],path).reshape(-1,4);c=apply_gate([b[i] for i in held],bs[held]+rr,'parse_and_repair',4)
                dump(OUT/f'{arm}_held.json',dict(correct=int(y[held][np.arange(len(held)),c].sum()),n=len(held),baseline='actual round25 eight-expert OOF, differs from historical six-expert held baseline'))
        r=score_tokens([s for q in tokens['validation'] for s in q],OUT/'full'/arm).reshape(-1,4);c=apply_gate(v,vbs+r,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
        rec=dict(arm=arm,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()));bundle=OUT/f'bundle_{arm}';bundle.mkdir()
        for name in ('model.npz','config.json','vocabulary.json'):shutil.copy2(OUT/'full'/arm/name,bundle/name)
        shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');rec['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert rec['asset_bytes']<=16*1024**2
        pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/f'validation_{arm}_predictions.json',pred);records.append(rec);print('VAL',rec,flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle=f"experiments/round76/bundle_{best['arm']}"),selected='round76' if best['correct']>753 else 'round75',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
