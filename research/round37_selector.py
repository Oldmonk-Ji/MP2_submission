import json,importlib
from pathlib import Path
import torch
from round29_programs import parse,encode_questions,encode_programs
from round29_model import predict_encoded
from round25_selector import scores
from round4_gates import apply_gate


def predict(bank,arrays,checkpoint):
    if not bank:return []
    torch.set_num_threads(4);path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text());number=cfg['source_round']
    parsed=[parse(r['question'],c['text']) for r in bank for c in r['candidates']];encoder=encode_questions
    if number<=31:
        model,vocab=importlib.import_module(f'round{number}_model').load(path/'semantic');p=encode_programs(parsed).reshape(len(bank),4,-1);q=encoder(bank,vocab,256)
        s,_=predict_encoded(model,q,p)
    else:
        from round32_graph import load,encode_graphs,predict_encoded as predict_graph
        model,vocab=load(path/'semantic');r,p=encode_graphs(parsed,[r['question'] for r in bank for c in r['candidates']]);r=r.reshape(len(bank),4,24);p=p.reshape(len(bank),4,24,24)
        if number==33:
            from round33_text import encode_questions as encoder
        q=encoder(bank,vocab,256);s=predict_graph(model,q,r,p)
    delta=s-s.mean(1,keepdims=True)
    return apply_gate(bank,scores(bank,arrays,path/'baseline')+cfg['alpha']*delta,'parse_and_repair',4).astype(int).tolist()
