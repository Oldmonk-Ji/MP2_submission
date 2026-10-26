"""Optional strict-trace local compatibility, extracted from new inputs."""
from round186_trace import extract
from round103_features import row_features,matrix

def programs(row):return [extract(row['question'],c['text'])[0] for c in row['candidates']]

def build(bank,packets=None,progress=None):
    fs=[]
    for i,row in enumerate(bank):
        packet=programs(row) if packets is None else packets[i];fs.extend(row_features(row,packet,'local'))
        if progress is not None and i%500==0:progress(i)
    return matrix(fs)
