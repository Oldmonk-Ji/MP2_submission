"""Question/program semantic compatibility is only a soft score adjustment."""
from pathlib import Path
import json,importlib
import numpy as np
from round25_selector import scores
from round4_gates import apply_gate


def predict(bank,arrays,checkpoint):
    if not bank:return []
    import torch
    from round29_programs import parse,encode_questions,encode_programs
    from round29_model import predict_encoded,centered
    torch.set_num_threads(4)
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text())
    base=scores(bank,arrays,path/'baseline')
    module=importlib.import_module(f"round{cfg['round']}_model");model,vocab=module.load(path/'semantic')
    programs=[[parse(r['question'],c['text']) for c in r['candidates']] for r in bank]
    p=encode_programs([c for r in programs for c in r]).reshape(len(bank),4,-1)
    q=encode_questions(bank,vocab,max_length=256);ss,_=predict_encoded(model,q,p)
    mask=np.asarray([[c[cfg['mask']] for c in r] for r in programs])
    delta=centered(ss,mask)
    return apply_gate(bank,base+cfg['alpha']*delta,'parse_and_repair',4).astype(int).tolist()
