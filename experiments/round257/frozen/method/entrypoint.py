import os,sys
def guard(event,args):
    if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0]).replace(chr(92),"/").lower()
        if any(p.endswith("/data/gsm8k_"+s+".jsonl") for s in ("train","validation","test")):
            raise PermissionError("Prediction must not open reference answers")
sys.addaudithook(guard)
_before=set(sys.modules)
from round257_selector import predict as model_predict
if "data_access" in set(sys.modules)-_before:
    raise ImportError("Inference imported research data access")
def predict(bank,arrays,checkpoint):
    return model_predict(bank,arrays,checkpoint)
