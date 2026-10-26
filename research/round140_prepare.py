from round22_common import ROOT,load,read,dump,sha
from round140_literals import packet,checks
from round139_features import matrix
from scipy import sparse
import numpy as np,time
OUT=ROOT/'experiments/round140'
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(checks=checks(),method='Input-only literal universe and candidate program maps. Fixed27 features: availability and counts, then mean/min/max of8 root-level present/absent compatibility statistics. Unknown paths masked; absent program is not an ABSENT proof.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round140*.py')},test_opened=False))
    for split in ('train','validation'):
        b,a,y,ids=load(split);programs=read(ROOT/f'experiments/round102/{split}_programs.json');fs=[];ps=[];ptr=[0]
        for i,(row,pp) in enumerate(zip(b,programs)):
            f,p=packet(row,pp);fs.extend(f);ps.append(p);ptr.append(len(fs))
            if i%500==0:print('PREPARE',split,i,flush=True)
        x=matrix(fs);sparse.save_npz(OUT/f'{split}_features.npz',x);np.save(OUT/f'{split}_ptr.npy',np.asarray(ptr));dump(OUT/f'{split}_packets.json',ps);dump(OUT/f'{split}_metadata.json',dict(ids=ids.tolist(),pairs=x.shape[0],features=x.shape[1],nonzeros=x.nnz,test_opened=False))
    dump(OUT/'preparation_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
