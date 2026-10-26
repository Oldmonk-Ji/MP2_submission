from round22_common import ROOT,load,read,dump,sha
from round159_factors import source_features as old_source,pair_features as old_pair,matrix,probabilities
from round176_types import source_features,pair_features,checks
from round167_train import fit
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round176'
if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter()
    for src,h in read(ROOT/'experiments/round175/plan.json')['source_hashes'].items():assert sha(ROOT/src)==h
    dump(OUT/'plan.json',dict(checks=checks(),arms=['corrected_labels','corrected_types'],method='Same frozen175 labels and fixedC1/16k dimensions. Source loss class-balanced at unchanged total weight. corrected_labels uses original139 inputs; corrected_types repairs fraction/interval kind and family in source and source/literal inputs. Constant head frozen159; unmodified-scope arm also frozen159.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round176*.py')},test_opened=False))
    b,a,y,ids=load('train');index={int(idx):i for i,idx in enumerate(ids)}
    for arm in ('corrected_labels','corrected_types'):
        folder=OUT/arm;folder.mkdir();data={};oof={}
        for name in ('source','scope'):
            rows=read(ROOT/f'experiments/round175/full_{name}_labels.json');features=[]
            for r in rows:
                q=b[index[r['train_index']]]['question'];fn=(source_features if arm=='corrected_types' else old_source) if name=='source' else (pair_features if arm=='corrected_types' else old_pair);features.append(fn(q,r['source']) if name=='source' else fn(q,r['quantity'],r['constant']))
            data[name]=(matrix(features),np.array([r['label'] for r in rows]),np.array([r['train_index'] for r in rows]),np.array([r['fold'] for r in rows]));oof[name]=np.zeros((len(rows),2 if name=='source' else 4),np.float32)
        for stage in ('oof0','oof1','oof2','nested0','nested1','full'):
            held=int(stage[-1]) if stage!='full' else None
            with np.load(ROOT/f'experiments/round159/{stage}.npz') as z:w={k:z[k] for k in z.files}
            meta=read(ROOT/f'experiments/round159/{stage}.json')
            for name,d in data.items():
                mask=np.ones(len(d[1]),bool) if held is None else d[3]==1-held if stage.startswith('nested') else d[3]!=held
                if name=='source' or arm=='corrected_types':ww,mm=fit(d,mask,name);w.update(ww);meta['heads'][name]=mm
                if stage.startswith('oof'):oof[name][d[3]==held]=probabilities(d[0][d[3]==held],w,name)
            with np.load(ROOT/f'experiments/round159/{stage}.npz') as z:assert np.array_equal(w['constant_weight'],z['constant_weight']);assert np.array_equal(w['constant_bias'],z['constant_bias'])
            meta['training_parent_ids']=sorted(set(p for h in meta['heads'].values() for p in h['training_parent_ids']));meta['type_arm']=arm
            if held is not None:
                for d in data.values():assert not set(d[2][d[3]==held])&set(meta['training_parent_ids'])
            np.savez(folder/f'{stage}.npz',**w);dump(folder/f'{stage}.json',meta);print('COMPLETE',arm,stage,flush=True)
        report={}
        for name,d in data.items():
            held=d[3]==2;yy=d[1][held];pred=oof[name][held].argmax(1);report[name]=dict(n=int(held.sum()),correct=int((pred==yy).sum()),by_class={str(k):dict(n=int((yy==k).sum()),correct=int(((yy==k)&(pred==k)).sum())) for k in range(oof[name].shape[1])});np.savez(folder/f'{name}_oof.npz',probability=oof[name],labels=d[1],parents=d[2],folds=d[3])
        dump(folder/'held_heads.json',report);print('HELD',arm,report,flush=True)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
