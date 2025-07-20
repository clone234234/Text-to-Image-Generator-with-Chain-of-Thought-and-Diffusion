import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.quantization import FakeQuantize
from transformers import CLIPTokenizer
from typing import Optional, Dict

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = dropout
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() not in [2, 3, 4]:
                raise ValueError(f"Mask dimension {mask.dim()} is not supported")
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.size(1) == 1 and scores.size(1) == self.num_heads:
                mask = mask.expand(-1, self.num_heads, -1, -1)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, -1e9)
            else:
                scores = scores + mask
        scores = torch.clamp(scores, min=-1e9, max=1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.output_linear(attention_output)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.quant_fc1 = FakeQuantize(
            observer=torch.quantization.MinMaxObserver, quant_min=-127, quant_max=127, dtype=torch.qint8
        )
    
    def forward(self, x):
        x = self.quant_fc1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout1(attn_output)
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout2(ff_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout1(attn_output)
        norm_x = self.norm2(x)
        attn_output = self.cross_attn(norm_x, memory, memory, src_mask)
        x = x + self.dropout2(attn_output)
        norm_x = self.norm3(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout3(ff_output)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        dropout=0.1,
        max_seq_length=77,  # Default to CLIP's max_length
        tokenizer: Optional[CLIPTokenizer] = None,
        vocab: Optional[Dict[str, int]] = None,
        vocab_size: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.use_tokenizer = tokenizer is not None
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.vocab_size = vocab_size if vocab_size is not None else (tokenizer.vocab_size if tokenizer else len(vocab))

        # Use embedding for custom vocab or tokenizer
        if self.use_tokenizer:
            self.embedding = nn.Embedding(self.vocab_size, d_model)
        else:
            if vocab is None:
                raise ValueError("Must provide vocab or tokenizer")
            self.embedding = nn.Embedding(self.vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.output_linear = nn.Linear(d_model, self.vocab_size)
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
    
    def encode_text(self, text: str) -> torch.Tensor:
        if self.use_tokenizer:
            tokens = self.tokenizer(
                text,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt'
            ).input_ids.to(self.embedding.weight.device)
        else:
            if self.vocab is None:
                raise ValueError("Custom vocabulary not provided")
            tokens = [self.vocab.get(char, self.vocab.get('<unk>', 0)) for char in text]
            tokens = tokens[:self.tokenizer.model_max_length] if self.use_tokenizer else tokens[:self.max_seq_length]
            tokens = torch.tensor([tokens], dtype=torch.long, device=self.embedding.weight.device)
        return tokens
    
    def encode(self, src, src_mask=None):
        if isinstance(src, str):
            src = self.encode_text(src)
        src = src.to(self.embedding.weight.device)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        if src_mask is None:
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        if isinstance(tgt, str):
            tgt = self.encode_text(tgt)
        tgt = tgt.to(self.embedding.weight.device)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        if tgt_mask is None:
            seq_len = tgt.size(1)
            device = tgt.device
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
            tgt_padding_mask = tgt_padding_mask.expand(-1, -1, seq_len, -1)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0) | ~tgt_padding_mask
        return self.decoder(tgt, memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        if isinstance(src, str):
            src = self.encode_text(src)
        if isinstance(tgt, str):
            tgt = self.encode_text(tgt)
        if tgt is None:
            tgt = src
        if src_mask is None:
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        if tgt_mask is None:
            seq_len = tgt.size(1)
            device = src.device
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
            tgt_padding_mask = tgt_padding_mask.expand(-1, -1, seq_len, -1)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0) | ~tgt_padding_mask
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        output = self.output_linear(output)
        return output