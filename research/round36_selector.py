import json
from pathlib import Path
import torch
from round36_model import load,encode,predict_data
from round25_selector import scores
from round22_selector import normalized
from round4_gates import apply_gate


def predict(bank,arrays,checkpoint):
    if not bank:return []
    torch.set_num_threads(4);path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text());model,vocab=load(path/'semantic')
    data,_=encode(bank,arrays,vocab);s=predict_data(model,data)
    return apply_gate(bank,scores(bank,arrays,path/'baseline')+cfg['alpha']*normalized(s),'parse_and_repair',4).astype(int).tolist()
