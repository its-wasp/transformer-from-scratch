import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads 
        self.head_dim = d_model // n_heads 
        assert self.head_dim * self.n_heads == self.d_model, "d_model must be perfectly divisible by n_heads"

        # define 4 linear layers for Q, K, V, O
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.o_linear = nn.Linear(d_model, d_model)

        #define dropout 
        self.dropout = nn.Dropout(p=dropout)



    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):

        #Linear projections
        Q = self.q_linear(Q) # shape (batch_size, seq_len, d_model)
        K = self.k_linear(K)
        V = self.v_linear(V)

        # reshape to (batch_size, seq_len, n_heads, head_dim)
        Q = Q.view(Q.shape[0], Q.shape[1], self.n_heads, self.head_dim)
        K = K.view(K.shape[0], K.shape[1], self.n_heads, self.head_dim)
        V = V.view(V.shape[0], V.shape[1], self.n_heads, self.head_dim)

        # transpose to (batch_size, n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2) 

        # attention scores (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # apply softmax
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # multiply with V
        output = torch.matmul(probs, V) # (batch_size, n_heads, seq_len, head_dim) 

        # reshape and concat back to (batch_size, seq_len, d_model) 
        output = output.transpose(1, 2).contiguous() 
        output = output.view(output.shape[0], output.shape[1], self.d_model) 

        # apply the linear layer 

        output = self.o_linear(output) 

        return output  # (batch_size, seq_len, d_model)


