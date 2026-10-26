import torch
from torch import nn

class Model(nn.Module):
    def __init__(self,n,ordered):
        super().__init__();self.ordered=ordered;self.words=nn.Embedding(n,48,padding_idx=0);self.positions=nn.Embedding(240,48);self.segments=nn.Embedding(2,48)
        layer=nn.TransformerEncoderLayer(48,2,96,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Sequential(nn.LayerNorm(48),nn.Linear(48,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,segment):
        h=self.words(x)+self.segments(segment)
        if self.ordered:h=h+self.positions(torch.arange(x.shape[1],device=x.device))[None,:,:]
        h=self.encoder(h,src_key_padding_mask=x.eq(0))
        return self.head(h[:,0]).squeeze(-1)
