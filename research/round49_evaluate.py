"""Fixed complete-VAL ablation; no official parser change or TEST access."""
from data_access import ROOT
import round48_evaluate
import round46_evaluate
from round49_selector import infer

if __name__=='__main__':
    out=ROOT/'experiments/round49'
    round48_evaluate.OUT=out;round48_evaluate.held_relations()
    round46_evaluate.OUT=out;round46_evaluate.infer=infer;round46_evaluate.main()
