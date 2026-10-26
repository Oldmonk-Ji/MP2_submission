"""Optional round209 strict-trace local compatibility, extracted from new inputs."""
from round209_trace import extract, FLAGS
from round103_features import row_features, matrix


def programs(row, flags=None):
    flags = dict(FLAGS) if flags is None else flags
    return [extract(row['question'], c['text'], flags)[0] for c in row['candidates']]


def build(bank, packets=None, progress=None, flags=None):
    fs = []
    for i, row in enumerate(bank):
        packet = programs(row, flags) if packets is None else packets[i]
        fs.extend(row_features(row, packet, 'local'))
        if progress is not None and i % 500 == 0:
            progress(i)
    return matrix(fs)
