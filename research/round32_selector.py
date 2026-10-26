from pathlib import Path
import json
import numpy as np
from round25_selector import scores
from round4_gates import apply_gate


def predict(bank,arrays,checkpoint):
    if not bank:return []
    import torch
    from round29_model import centered
    from round29_programs import encode_questions,parse
    from round32_graph import load,encode_graphs,predict_encoded
    torch.set_num_threads(4);path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text())
    encoder=encode_questions
    if cfg['round']==33:
        from round33_text import encode_questions as encoder
    base=scores(bank,arrays,path/'baseline');model,vocab=load(path/'semantic')
    parsed=[[parse(r['question'],c['text']) for c in r['candidates']] for r in bank]
    roles,pairs=encode_graphs([c for row in parsed for c in row],[r['question'] for r in bank for c in r['candidates']])
    roles=roles.reshape(len(bank),4,24);pairs=pairs.reshape(len(bank),4,24,24)
    ss=predict_encoded(model,encoder(bank,vocab,max_length=256),roles,pairs)
    mask=np.asarray([[c[cfg['mask']] for c in row] for row in parsed]);delta=centered(ss,mask)
    return apply_gate(bank,base+cfg['alpha']*delta,'parse_and_repair',4).astype(int).tolist()
