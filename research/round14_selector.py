"""Deployable, fail-closed selector: round14 has NOT earned override permission.

The experimental relation model is available separately for offline diagnosis.
It is not an additional inference dependency of the retained selector.
"""
import json
from pathlib import Path
from round4_selector import predict as incumbent_predict


def predict(bank,arrays,checkpoint):
    cfg=json.loads((Path(checkpoint)/'config.json').read_text(encoding='utf-8'))
    if cfg.get('override_enabled') is not False or cfg.get('mode')!='diagnostic_only':
        raise ValueError('This prototype has no approved semantic override implementation.')
    return incumbent_predict(bank,arrays,Path(checkpoint)/cfg['incumbent'])


def diagnose(bank):
    from common import parse_answer
    from round14_local import inspect
    return [inspect(row['question'],[c['text'] for c in row['candidates']],
                    [parse_answer(c['text']) for c in row['candidates']]) for row in bank]
