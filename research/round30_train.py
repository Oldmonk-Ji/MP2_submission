"""Matched pointer architecture comparison; no validation or test access."""
from data_access import ROOT
from round22_common import dump,read,sha
from round30_model import ProgramModel
import round29_train as training
import shutil,sys,datetime

OUT=ROOT/'experiments/round30'


def main(stage,variant):
    OUT.mkdir(exist_ok=True)
    if not (OUT/'plan.json').exists():
        dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=29029,
            training_n=4743,validation_n=798,test_opened=False,
            hypothesis='Contextual number pointers learn which entity/time/batch each arithmetic operand refers to.',
            control='round29 fixed output slots, same data, seed, widths, objectives and epochs',
            diagnostics='same normalized-question grouped 3795/948 TRAIN split',
            variants=['reference_only','reference_and_candidate_ranking'],epochs=[15,40,80],alphas=[.1,.25,.5],
            parser_source_sha=sha(ROOT/'research/round29_programs.py')))
        for f in ['training_programs.jsonl','diagnostic_split.npz']:shutil.copyfile(ROOT/'experiments/round29'/f,OUT/f)
    training.OUT=OUT;training.ProgramModel=ProgramModel
    training.train(stage,variant)


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
