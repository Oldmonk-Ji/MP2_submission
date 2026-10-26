import json
from pathlib import Path
import numpy as np
import torch
from round29_programs import parse,encode_questions
from round32_graph import encode_graphs
from round35_model import load,predict_encoded
from round25_selector import scores
from round22_selector import normalized
from round4_gates import apply_gate


def predict(bank,arrays,checkpoint):
    if not bank:return []
    torch.set_num_threads(4);path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text());model,vocab=load(path/'semantic')
    parsed=[parse(r['question'],c['text']) for r in bank for c in r['candidates']]
    roles,pairs=encode_graphs(parsed,[r['question'] for r in bank for c in r['candidates']]);roles=roles.reshape(len(bank),4,24);pairs=pairs.reshape(len(bank),4,24,24)
    q=encode_questions(bank,vocab,256);s=predict_encoded(model,q,roles,pairs,np.asarray(arrays['embeddings'],np.float32))
    return apply_gate(bank,scores(bank,arrays,path/'baseline')+cfg['alpha']*normalized(s),'parse_and_repair',4).astype(int).tolist()
