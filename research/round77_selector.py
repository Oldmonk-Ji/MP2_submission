"""Use the numerically checked attention path on CPU."""
import torch
from round76_selector import predict as original_predict

def predict(bank,arrays,checkpoint):
    previous=torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        return original_predict(bank,arrays,checkpoint)
    finally:torch.backends.mha.set_fastpath_enabled(previous)
