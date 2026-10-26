"""Fresh-process and no-reference interface verification for the final experiment."""
from data_access import ROOT
import round47_verify as verification
from round50_selector import predict

if __name__=='__main__':
    original_process=verification.process
    def process(args):
        return original_process([ROOT/'research/round50_selector.py' if str(a)==str(ROOT/'research/round46_selector.py') else a for a in args])
    verification.OUT=ROOT/'experiments/round50';verification.predict=predict;verification.process=process;verification.main()
