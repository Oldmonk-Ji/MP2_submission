"""All experimental data access is restricted to train and validation."""
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'DASE7506-MP2-main'/'code'
sys.path.insert(0,str(CODE))


def forbid_test(event,args):
    if event == 'open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        name=os.fsdecode(args[0]).replace('\\','/').lower()
        if name.endswith(('gsm8k_test.jsonl','gsm8k_test.npz')):
            raise PermissionError('This research run must not access test inputs or references.')


sys.addaudithook(forbid_test)

import numpy as np
from common import load_inputs,parse_answer,read_jsonl


def load(split):
    if split not in ('train','validation'):
        raise ValueError('Research restricted to train and validation')
    bank,arrays=load_inputs(split)
    refs=read_jsonl(CODE/'data'/f'gsm8k_{split}.jsonl')
    assert len(bank)==len(refs)
    labels=[]
    for row,ref in zip(bank,refs):
        assert row['id']==ref['id'] and row['question']==ref['question']
        target=parse_answer(ref['answer'])
        assert target is not None
        labels.append([int(parse_answer(c['text'])==target) for c in row['candidates']])
    return bank,arrays,np.asarray(labels,dtype=np.float32)
