from data_access import ROOT
from round22_common import read,dump,load,sha
from round34_features import transform,DIM
from round29_programs import parse
from round25_selector import scores
from round4_gates import apply_gate
from features import metadata
from scipy.optimize import minimize
from scipy import sparse
import numpy as np
import datetime,json,time,shutil,functools
import round4_gates

OUT=ROOT/'experiments/round34'


def fit(x,y,weights,reg):
    a=[];b=[];ww=[]
    for i in range(len(y)):
        pos=np.flatnonzero(y[i]);neg=np.flatnonzero(1-y[i])
        for p in pos:
            for n in neg:a.append(i*4+p);b.append(i*4+n);ww.append(weights[i]/(len(pos)*len(neg)))
    xx=x[a]-x[b];ww=np.asarray(ww);ww/=ww.sum()
    def objective(w):
        z=xx@w;grad=-1/(1+np.exp(np.clip(z,-40,40)))
        return np.dot(ww,np.logaddexp(0,-z))+reg*np.dot(w,w)/2,np.asarray(xx.T@(ww*grad)).ravel()+reg*w
    r=minimize(objective,np.zeros(DIM),jac=True,method='L-BFGS-B',options=dict(maxiter=150,ftol=1e-9,gtol=1e-7,maxcor=8))
    return r.x.astype(np.float32),dict(success=bool(r.success),iterations=int(r.nit),loss=float(r.fun))


def main():
    OUT.mkdir();dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_n=4743,validation_n=798,test_opened=False,
        method='Explicit target/entity/operand alignment ranker, trained from actual candidates and reference-derived wrong-program contrasts',
        control='auxiliary weight zero vs one, same features and regularization',auxiliary=[0.,1.],reg=[.00001,.0001,.001],alphas=[.1,.25,.5,1.],
        features='Hash of local quantity words x target words x tree role; pair relations and entity identity. No whole-question keys or IDs.',
        supervision='All eligible official training rows; unanimous-correct groups have no positive-negative ordering constraint. Derived groups are not independent questions.'))
    rows=[json.loads(s) for s in open(ROOT/'experiments/round31/training_programs.jsonl',encoding='utf-8')]
    dump(OUT/'training_ids.json',[r['train_index'] for r in rows])
    examples=[(r['question'],c) for r in rows for c in r['candidates']];y=[r['labels'] for r in rows];original_n=len(rows)
    for r in rows:
        if not any(r['contrast_mask']):continue
        examples.append((r['question'],r['reference']))
        for neg in r['contrast_negatives']:
            item=dict(r['reference'],program=neg['program'],tokens=neg['tokens']);examples.append((r['question'],item))
        y.append([1]+[1-int(k) for k in r['contrast_mask']])
    y=np.asarray(y,np.float32);tick=time.perf_counter();print('FEATURES',len(examples),flush=True)
    x=transform(examples);sparse.save_npz(OUT/'training_features.npz',x);print('FEATURES DONE',time.perf_counter()-tick,x.nnz,flush=True)
    b,a,vy,_=load('validation');base=scores(b,a,ROOT/'experiments/round25/selected_bundle')
    vx=transform([(r['question'],parse(r['question'],c['text'])) for r in b for c in r['candidates']])
    old=apply_gate(b,base,'parse_and_repair',4);oldok=vy[np.arange(len(b)),old].astype(bool);hard=metadata(b)['hard'];records=[];choices={'control':old}
    records.append(dict(name='control',model=None,alpha=0.,correct=int(oldok.sum()),n=len(b),hard_correct=int(oldok[hard].sum()),hard_n=int(hard.sum()),repaired=0,damaged=0,net=0))
    original=round4_gates.trace;round4_gates.trace=functools.lru_cache(maxsize=8000)(original)
    for aux in [0.,1.]:
        weights=np.r_[np.ones(original_n),np.full(len(y)-original_n,aux)]
        for reg in [.00001,.0001,.001]:
            name=f'aux{aux}_r{reg}';w,info=fit(x,y,weights,reg);np.savez(OUT/f'{name}.npz',weight=w)
            delta=np.asarray(vx@w).reshape(-1,4)
            for alpha in [.1,.25,.5,1.]:
                c=apply_gate(b,base+alpha*delta,'parse_and_repair',4);ok=vy[np.arange(len(b)),c].astype(bool)
                key=f'{name}_a{alpha}';choices[key]=c
                r=dict(name=key,model=name,aux=aux,reg=reg,alpha=alpha,correct=int(ok.sum()),n=len(b),hard_correct=int(ok[hard].sum()),hard_n=int(hard.sum()),
                    repaired=int((ok&~oldok).sum()),damaged=int((~ok&oldok).sum()),net=int(ok.sum()-oldok.sum()),optimizer=info)
                records.append(r);print('VAL',r,flush=True)
            dump(OUT/'validation_records.json',records)
    round4_gates.trace=original;winner=max(records,key=lambda r:(r['correct'],-r['damaged'],-r['alpha']))
    dump(OUT/'selection.json',dict(selected=winner,records=records,test_opened=False,seconds=time.perf_counter()-tick))
    np.savez(OUT/'validation_choices.npz',**choices);print('SELECTED',winner,flush=True)
    if winner['model'] is None:return
    bundle=OUT/'selected_bundle';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
    shutil.copyfile(OUT/f"{winner['model']}.npz",bundle/'model.npz');dump(bundle/'config.json',dict(alpha=winner['alpha']))
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets);assert size<=16*1024**2
    dump(OUT/'asset_manifest.json',dict(bytes=size,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets}))


if __name__=='__main__':main()
