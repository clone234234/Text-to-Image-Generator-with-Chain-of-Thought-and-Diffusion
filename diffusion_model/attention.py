import torch
from torch import nn
from torch.nn import functional as F
import math

class self_attention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias= True,out_proj_bias= True):
        super().__init__()
        self.in_proj= nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj= nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads= n_heads
        self.d_embed= d_embed // n_heads
    def forward(self, x: torch.Tensor, casual_mask=False):
        input_shape= x.shape
        batch_size, seq_len, d_embed = input_shape
        intermin_shape=(batch_size, seq_len,  self.n_heads, self.d_embed) 
        q,v,k =self.in_proj(x).chunk(3, dim=-1)
        q= q.view(intermin_shape).transpose(1, 2)
        k= k.view(intermin_shape).transpose(1, 2)
        v= v.view(intermin_shape).transpose(1, 2)
        weight=q @ k.transpose(-1, -2) 
        if casual_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask,-torch.inf)
        weight /=math.sqrt(self.d_embed)
        weight= F.softmax(weight, dim=-1)

        output= weight @ v
        output= output.transpose(1, 2).reshape(batch_size, seq_len, d_embed)
        output = self.out_proj(output)
        return output    

class CrossAttention(nn.Module):
    def __init__ (self, n_heads, d_cross, d_embed, in_proj_bias= True, out_proj_bias= True): 
        super().__init__()
        self.q_proj=nn.Linear(d_embed,d_embed, bias=in_proj_bias)
        self.k_proj=nn.Linear(d_cross,d_embed, bias=in_proj_bias)
        self.v_proj=nn.Linear(d_cross,d_embed, bias=in_proj_bias)
        self.out_proj= nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads= n_heads
        self.d_embed= d_embed // n_heads
    def forward(self,x,y):
        input_shape= x.shape
        batch_size, seq_len, d_embed = input_shape
        interim_shape=(batch_size, seq_len,  self.n_heads, self.d_embed)
        q= self.q_proj(x)
        k= self.k_proj(y)
        v= self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2) 
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2)
        
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        return output
