from round22_common import ROOT,load,dump,sha
from round60_selector import documents
from round3_train import pair_data
from round25_selector import scores as baseline
from round22_selector import normalized
from round4_gates import apply_gate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np,time,shutil
OUT=ROOT/'experiments/round60'

def fit(b,y,ids,sections,p):
    p.mkdir(parents=True);vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=20000,sublinear_tf=True,dtype=np.float32)
    x=vec.fit_transform(documents(b,sections));xx,yy,ww=pair_data(x,y)
    m=LogisticRegression(C=10,fit_intercept=False,solver='liblinear',max_iter=600,random_state=60060).fit(xx,yy,sample_weight=ww)
    np.savez(p/'model.npz',idf=vec.idf_.astype(np.float32),weight=m.coef_[0].astype(np.float32));dump(p/'vocabulary.json',{k:int(v) for k,v in vec.vocabulary_.items()});dump(p/'config.json',dict(sections=sections,alpha=0))
    dump(p.parent/(p.name+'_training.json'),dict(training_ids=ids.tolist(),C=10,seed=60060))
    return lambda bb: np.asarray(vec.transform(documents(bb,sections))@m.coef_[0]).reshape(-1,4)

if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['plain','sections'],C=10,alpha=[.1,.25,.5],test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round60*.py')}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation')
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    bs=normalized(baseline(v,va,ROOT/'experiments/round25/selected_bundle'));bc=apply_gate(v,bs,'parse_and_repair',4);records=[]
    for sections in (False,True):
        arm='sections' if sections else 'plain'
        fn=fit([r for r,k in zip(b,f) if k!=2],y[f!=2],ids[f!=2],sections,OUT/'diagnostic'/arm)
        hb=[r for r,k in zip(b,f) if k==2];s=fn(hb);c=apply_gate(hb,s,'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(correct=int(y[f==2][np.arange(len(hb)),c].sum()),n=len(hb)))
        fn=fit(b,y,ids,sections,OUT/'full'/arm);s=normalized(fn(v));np.savez(OUT/f'{arm}_validation_scores.npz',scores=s)
        for alpha in (.1,.25,.5):
            c=apply_gate(v,bs+alpha*s,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
            r=dict(arm=arm,alpha=alpha,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()),predictions=c.tolist());records.append(r);print({k:v for k,v in r.items() if k!='predictions'},flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['alpha']))
    bundle=OUT/'bundle';shutil.copytree(OUT/'full'/best['arm'],bundle);shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
    dump(bundle/'config.json',dict(sections=best['arm']=='sections',alpha=best['alpha']))
    size=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert size<=16*1024**2
    dump(OUT/f"validation_{best['arm']}_predictions.json",dict(selected=best['predictions']))
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle='experiments/round60/bundle'),selected='round60' if best['correct']>752 else 'round25',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,asset_bytes=size,test_opened=False))
