"""Apply unchanged deployment checks to the local-relation selector."""
from data_access import ROOT
import round47_verify as verification
from round48_selector import predict

if __name__=='__main__':
    original_process=verification.process
    def process(args):
        args=[ROOT/'research/round48_selector.py' if str(a)==str(ROOT/'research/round46_selector.py') else a for a in args]
        return original_process(args)
    verification.OUT=ROOT/'experiments/round48'
    verification.predict=predict
    verification.process=process
    verification.main()
