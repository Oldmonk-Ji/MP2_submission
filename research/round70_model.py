"""Small permutation-equivariant candidate attention; no pretrained weights."""
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,attention):
        super().__init__();self.attention=attention;self.qkv=nn.Linear(96,288);self.proj=nn.Linear(96,96)
        self.ff=nn.Sequential(nn.Linear(96,192),nn.GELU(),nn.Dropout(.1),nn.Linear(192,96));self.n1=nn.LayerNorm(96);self.n2=nn.LayerNorm(96)
        self.bias=nn.Parameter(torch.zeros(3));self.drop=nn.Dropout(.1)
    def forward(self,h,same):
        v=self.n1(h)
        if self.attention:
            q,k,v=self.qkv(v).reshape(-1,4,3,3,32).permute(2,0,3,1,4).unbind(0)
            score=(q@k.transpose(-1,-2))/(32**.5)+same[:,None,:,:]*self.bias[None,:,None,None]
            z=(score.softmax(-1)@v).transpose(1,2).reshape(-1,4,96)
            h=h+self.drop(self.proj(z))
        else:h=h+self.drop(self.proj(v))
        return h+self.drop(self.ff(self.n2(h)))

class Model(nn.Module):
    def __init__(self,d,attention):
        super().__init__();self.encoder=nn.Linear(d,96);self.blocks=nn.ModuleList([Block(attention),Block(attention)]);self.head=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x,same):
        h=self.encoder(x)
        for b in self.blocks:h=b(h,same)
        return self.head(h).squeeze(-1)
