"""Fresh inference checks for the current matched ablations."""
from round22_common import ROOT,load,read,dump,sha
from round104_selector import predict as unchanged
from round125_selector import predict as relative
from round129_selector import predict as pronoun_safe
from round125_features import build as relative_build
from round129_features import build as pronoun_build
from threadpoolctl import threadpool_limits
from scipy import sparse
if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('validation');b=b[:8];a={'embeddings':a['embeddings'][:8]};report=dict(source_sha256=sha(__file__),test_opened=False)
    for n,fn in [(124,relative_build),(128,pronoun_build)]:
        got=fn(b,'local')[-1];expected=sparse.load_npz(ROOT/f'experiments/round{n}/validation_features.npz')[:32];d=got-expected;assert not d.nnz or abs(d.data).max()<1e-7
        report[f'round{n}_production_features_match']=True
    for n,arm,fn in [(125,'relative',relative),(127,'marginal',unchanged),(129,'pronoun_safe',pronoun_safe),(132,'invariant',unchanged)]:
        bundle=ROOT/f'experiments/round{n}/bundle_{arm}';got=fn(b,a,bundle);assert got==read(ROOT/f'experiments/round{n}/validation_{arm}_predictions.json')['selected'][:8]
        assert got==fn(b[:3],{'embeddings':a['embeddings'][:3]},bundle)+fn(b[3:],{'embeddings':a['embeddings'][3:]},bundle)
        report[f'round{n}_production_predictions_match']=True;report[f'round{n}_batch_invariance']=True
    dump(ROOT/'experiments/round132/feature_checks.json',report);print(report)
