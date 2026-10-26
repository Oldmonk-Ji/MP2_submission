"""Frozen policy weights with the fit-derived trace-completion feature ablation."""
from pathlib import Path
import json
import round52_selector
from round54_programs import proposals
from round52_selector import select,baseline_scores

def infer(bank,paths,structural=True):
    prior=round52_selector.proposals
    try:
        round52_selector.proposals=proposals
        return round52_selector.infer(bank,paths,structural)
    finally:round52_selector.proposals=prior

def predict(bank,arrays,checkpoint):
    if not bank:return []
    path=Path(checkpoint);cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
    # Older comparison bundles still use their own original program generator.
    if not cfg.get('trace_completion',False):return round52_selector.predict(bank,arrays,checkpoint)
    bs=baseline_scores(bank,arrays,path/'baseline');s,mask,matches=infer(bank,[path/p for p in cfg['policies']],cfg.get('structural',True))
    return select(bank,bs,s,mask,matches,cfg['mode']).astype(int).tolist()
