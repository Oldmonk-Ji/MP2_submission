from round22_common import ROOT,load,read,dump,sha
from round108_selector import predict as joint
from round104_selector import predict as consensus
from threadpoolctl import threadpool_limits
if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('validation');b=b[:8];a={'embeddings':a['embeddings'][:8]};report=dict(source_sha256=sha(__file__),test_opened=False)
    for n,arm,fn in [(118,'aligned',joint),(121,'consensus',consensus)]:
        bundle=ROOT/f'experiments/round{n}/bundle_{arm}';got=fn(b,a,bundle);assert got==read(ROOT/f'experiments/round{n}/validation_{arm}_predictions.json')['selected'][:8]
        assert got==fn(b[:3],{'embeddings':a['embeddings'][:3]},bundle)+fn(b[3:],{'embeddings':a['embeddings'][3:]},bundle)
        report[f'round{n}_fresh_predictions_match_cache']=True;report[f'round{n}_batch_invariance']=True
    dump(ROOT/'experiments/round121/feature_checks.json',report);print(report)
