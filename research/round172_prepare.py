from round22_common import ROOT,load,read,dump,sha
from round172_target_source import source_features,checks,matrix
from round29_programs import quantities
from scipy import sparse
import numpy as np,time
OUT=ROOT/'experiments/round172'
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(checks=checks(),method='Cross existing target token features with existing source local/typed features. Fixed16k hash and C1 source head; original feature block retained. Only source input block changes; constant/scope features and heads unchanged.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round172*.py')},test_opened=False))
    for split in ('train','validation'):
        b,a,y,ids=load(split);fs=[];ptr=[0]
        for row in b:
            fs.extend(source_features(row['question'],i) for i in range(len(quantities(row['question']))));ptr.append(len(fs))
        assert np.array_equal(ptr,np.load(ROOT/f'experiments/round162/{split}_uptr.npy'));x=matrix(fs);sparse.save_npz(OUT/f'{split}_source.npz',x);dump(OUT/f'{split}_metadata.json',dict(ids=ids.tolist(),source_rows=x.shape[0],nonzeros=x.nnz,test_opened=False))
    dump(OUT/'preparation_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
