"""Fresh-input smoke checks for the two failed process-supervision arms."""
from round22_common import ROOT,load,read,dump,sha
from round108_selector import predict as joint
from round113_selector import predict as decoupled
from round112_decoupled import checks
from round110_alignment import checks as loss_checks
from threadpoolctl import threadpool_limits
if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('validation');b=b[:8];a={'embeddings':a['embeddings'][:8]}
    report=dict(**checks(),**loss_checks(),source_sha256=sha(__file__),test_opened=False)
    for n,arm,fn in [(111,'aligned',joint),(113,'decoupled',decoupled)]:
        bundle=ROOT/f'experiments/round{n}/bundle_{arm}';got=fn(b,a,bundle);expected=read(ROOT/f'experiments/round{n}/validation_{arm}_predictions.json')['selected'][:8]
        assert got==expected,(n,got,expected)
        split=fn(b[:3],{'embeddings':a['embeddings'][:3]},bundle)+fn(b[3:],{'embeddings':a['embeddings'][3:]},bundle)
        assert split==got
        report[f'round{n}_fresh_predictions_match_cache']=True;report[f'round{n}_batch_invariance']=True
    dump(ROOT/'experiments/round113/feature_checks.json',report);print(report)
