import numpy as np
import torch
import torch.nn as nn
import math


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term) # even positions
        pe[:, 0, 1::2] = torch.cos(position * div_term) # odd positions
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


# Feedforward Layer 
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.ln1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.ln2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.activation(self.ln1(x))
        x = self.ln2(x)
        return x
    




# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # 4 linear layers -> q,k,v and linear for the output
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query):
        batch_size = query.size(0)

        # Linear projections
        Q = self.linear_q(query)
        K = self.linear_k(query)
        V = self.linear_v(query)

        # Split into multiple heads. Resulting dimenstions -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  

        # Scale the attentio scores
        scores = scores/np.sqrt(self.d_k)

        # Attention(Q,K,V):
        attn_weights = torch.softmax(scores, dim=-1)  # Attention weights
        attn_output = torch.matmul(attn_weights, V)  # Weighted sum with values

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # (batch_size, seq_len, d_model)
        attn_output = self.linear_out(attn_output)  # Final projection to the dimension of the model

        return attn_output
    


# Full Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=0.1)

        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x):
        x2 = self.self_attn(x)

        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.ff(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        return x

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, d_ff, output_dim):
        super(TransformerEncoder, self).__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Stacking multiple encoder layers
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=0.1)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.projection(x).unsqueeze(1)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)

        x = x.squeeze(1)
        output = self.fc_out(self.dropout(x))
        return output
    
