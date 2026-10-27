"""Same held/VAL gate as241; add only local TRAIN masked-word pretraining."""
from round22_common import ROOT,dump,sha
import round241_train as protocol
from round242_pretrain import train,PRETRAIN_EPOCHS

def record(path,value):
    if path.name=='plan.json':
        value.update(method='Independent raw-text comparator with TRAIN-only masked-word pretraining, then supervised ranking; no pretrained assets.',
            pretrain_epochs=PRETRAIN_EPOCHS,pretraining_population='Corresponding fitting TRAIN rows only; no held/VAL pretraining.',
            inference='Same compact word/CNN/cross-attention architecture; discard masked-word decoder.',
            source_hashes={n:sha(ROOT/'research'/n) for n in ('round241_text.py','round241_network.py','round241_train.py','round242_pretrain.py','round242_train.py')})
    dump(path,value)

if __name__=='__main__':
    protocol.OUT=ROOT/'experiments/round242'
    protocol.train=train
    protocol.dump=record
    protocol.main()
