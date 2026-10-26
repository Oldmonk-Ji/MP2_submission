"""Local relation policy, deterministic execution and original-candidate output."""
from pathlib import Path
import json
from round48_local import infer
from round46_selector import select
from round25_selector import scores

def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
    bs=scores(bank,arrays,path/'baseline');s,mask,matches=infer(bank,[path/p for p in cfg['policies']])
    return select(bank,bs,s,mask,matches,cfg['mode']).astype(int).tolist()
