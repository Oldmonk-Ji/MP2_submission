"""Fresh independent evaluation of the corrected evidence pipeline."""
from data_access import ROOT
import round47_verify as verification
from round49_selector import predict

if __name__=='__main__':
    original_process=verification.process
    def process(args):
        return original_process([ROOT/'research/round49_selector.py' if str(a)==str(ROOT/'research/round46_selector.py') else a for a in args])
    verification.OUT=ROOT/'experiments/round49';verification.predict=predict;verification.process=process;verification.main()
