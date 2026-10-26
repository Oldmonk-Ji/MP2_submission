"""Run the unchanged matched training implementation on corrected relation inputs."""
from data_access import ROOT
import round46_train

if __name__=='__main__':
    round46_train.OUT=ROOT/'experiments/round47'
    round46_train.main()
