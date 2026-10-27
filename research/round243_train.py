"""Same raw-text architecture; learn complementary corrections after local MLM."""
from round22_common import ROOT,dump,sha
import round241_train as protocol
from round243_residual import train

def record(path,value):
    if path.name=='plan.json':
        value.update(method='Independent locally pretrained raw-text network learns residual corrections to cross-fitted TRAIN ranking scores.',
            pretrain_epochs=6,offsets='Fit folds: round225 fit-only fusion plus round227 cross-fitted quadratic; held fold: round235 held scores. No fitting-label-dependent in-sample teacher.',
            inference='Export raw neural correction only, same architecture/assets as241; add to round235 before fixed gate.',
            source_hashes={n:sha(ROOT/'research'/n) for n in ('round241_text.py','round241_network.py','round241_train.py','round242_pretrain.py','round243_residual.py','round243_train.py')})
    dump(path,value)

if __name__=='__main__':
    protocol.OUT=ROOT/'experiments/round243'
    protocol.ALPHAS=[.25,.5,1.]
    protocol.train=train
    protocol.dump=record
    protocol.main()
