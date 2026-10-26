from data_access import ROOT
import round32_train as training
import sys
if __name__=='__main__':
    training.OUT=ROOT/'experiments/round33'
    training.main(sys.argv[1],anonymize=True)
