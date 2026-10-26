"""Candidate-level strict-trace event channels for the fusion readout.

Per-candidate scalars derived from the round209 strict trace: root presence,
accepted verified steps, symbolic solves, and failure-event counts. Appended
as extra channels; absence stays unknown rather than penalised.
"""
import numpy as np
from collections import Counter
from round209_trace import extract, FLAGS
from features import parse_answer

EVENT_NAMES = ('incorrect_numeric_equality', 'ungrounded_operand', 'unresolved_prior_result',
               'symbolic_equation_unknown', 'symbolic_linear_solved', 'lineage_value_mismatch',
               'unsupported_expression', 'unsupported_numeric_operator', 'ambiguity_overflow',
               'not_numeric', 'program_size', 'numeric_bound')


def candidate_vector(q, text, flags=None):
    roots, report = extract(q, text, flags)
    ev = Counter(report.get('events', []))
    v = [
        float(bool(roots)),
        np.log1p(len(roots)),
        np.log1p(report.get('accepted_steps', 0)),
        np.log1p(report.get('symbolic_solved', 0)),
        float(report.get('reason') == 'verified_trace'),
    ]
    for name in EVENT_NAMES:
        v.append(np.log1p(ev.get(name, 0)))
    return np.asarray(v, np.float32)


def build(bank, flags=None):
    flags = dict(FLAGS) if flags is None else flags
    return np.asarray([[candidate_vector(row['question'], c['text'], flags) for c in row['candidates']]
                       for row in bank], np.float32)


def checks():
    from round22_common import load
    v, _, _, _ = load('validation')
    row = v[32]
    m = build([row])
    assert m.shape == (1, 4, 5 + len(EVENT_NAMES))
    assert m[0, 0, 0] == 1.0 and m[0, 0, 2] > 0  # roots + accepted steps for the fixed trace
    return dict(shape_ok=True, vi32_roots_seen=True)
