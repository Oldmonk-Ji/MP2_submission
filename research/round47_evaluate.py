"""Evaluate the corrected, frozen experiment without touching TEST."""
from data_access import ROOT
import round46_evaluate
import round46_held_review

if __name__=='__main__':
    round46_held_review.OUT=ROOT/'experiments/round47'
    round46_held_review.evaluate()
    round46_evaluate.OUT=ROOT/'experiments/round47'
    round46_evaluate.main()
