"""Frozen local relation SFT/exact/GRPO with complete-answer coverage guard."""
from data_access import ROOT
import round48_evaluate
import round46_evaluate
from round50_selector import infer,select

if __name__=='__main__':
    out=ROOT/'experiments/round50'
    round48_evaluate.OUT=out;round48_evaluate.held_relations()
    round46_evaluate.OUT=out;round46_evaluate.infer=infer;round46_evaluate.select=select;round46_evaluate.main()
