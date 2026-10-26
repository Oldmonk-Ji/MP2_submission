"""New executable types and transferable relation feature policy."""
from pathlib import Path
import json,numpy as np
from round25_selector import scores as baseline_scores
from round50_selector import select
from round51_programs import proposals
from round65_local import packet_matrix as structural_matrix
from round48_local import packet_matrix as lexical_matrix
import round50_selector

def infer(bank,paths,structural=True):
    # Inference is single-threaded Python; BLAS retains the official four-thread cap.
    oldp,oldm=round50_selector.proposals,round50_selector.packet_matrix
    try:
        round50_selector.proposals=proposals
        round50_selector.packet_matrix=structural_matrix if structural else lexical_matrix
        return round50_selector.infer(bank,paths)
    finally:round50_selector.proposals,round50_selector.packet_matrix=oldp,oldm

def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
    bs=baseline_scores(bank,arrays,path/'baseline')
    s,mask,matches=infer(bank,[path/p for p in cfg['policies']],cfg.get('structural',True))
    return select(bank,bs,s,mask,matches,cfg['mode']).astype(int).tolist()
