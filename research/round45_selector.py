"""CPU-only original-candidate selection; no references or IDs required."""
from pathlib import Path
import json
from round25_selector import predict as candidate_predict
from round44_selector import predict as relation_predict

def predict(bank,arrays,checkpoint):
    if not bank:return []
    cfg=json.loads((Path(checkpoint)/'config.json').read_text(encoding='utf-8'))
    return (candidate_predict if cfg['level']=='candidate' else relation_predict)(bank,arrays,checkpoint)
