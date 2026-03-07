import torch.nn.modules.dropout
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


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU() 


    def forward(self, x: torch.Tensor):
        # (batch_size, seq_len, d_model)

        # project up through the first linear layer 
        x = self.linear1(x) # (batch_size, seq_len, d_ff)

        # activate
        x = self.relu(x)

        # dropout
        x = self.dropout(x)

        # project down 
        x = self.linear2(x) # (batch_size, seq_len, d_model)
        return x 


class ResidualConnection(nn.Module): 
    def __init__(self, d_model:int, dropout=0.1):
        super().__init__()
        self.size = d_model
        self.dropout = nn.Dropout(p=dropout) 
        self.norm = nn.LayerNorm(self.size)

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        # apply the sublayer and add the residual connection 

        return x + self.dropout(sublayer(self.norm(x))) 


class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout=0.1): 
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_foward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.res1 = ResidualConnection(d_model, dropout)
        self.res2 = ResidualConnection(d_model, dropout)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        
        # sublayer1 self attention 
        x = self.res1(x, lambda x: self.self_attention(x, x, x, mask))

        # sublayer2 feed forward 
        x = self.res2(x, self.feed_foward)

        return x #(batch_size, seq_len, d_model)


class DecoderLayer(nn.Module): 
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout=0.1):
        super().__init__()
        
        # 1. self attention (masked)
        # 2. cross attention
        # 3. feed forward 

        self.self_attention_masked = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout) 
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # three res connections 
        self.res1 = ResidualConnection(d_model, dropout)
        self.res2 = ResidualConnection(d_model, dropout)
        self.res3 = ResidualConnection(d_model, dropout)

    def forward(self, x:torch.Tensor, memory:torch.Tensor, src_mask, tgt_mask): 
        
        # self-attention with causal mask
        x = self.res1(x, lambda x_norm: self.self_attention_masked(x_norm, x_norm, x_norm, tgt_mask))

        # cross-attention with memory, Q = x_norm, K = memory, V  memory 
        # uses src_mask to mask padding in the encoder ouputs (memory)
        x = self.res2(x, lambda x_norm: self.cross_attention(x_norm, memory, memory, src_mask))

        # feed forward 
        x = self.res3(x, self.feed_forward)

        return x 


class Encoder(nn.Module):
    def __init__(self, d_model:int, n_layers:int, n_heads:int, d_ff:int, dropout=0.1): 
        super().__init__()
        
        # create a list of n_layers EncoderLayer instances 
        # use nn.ModuleList so the parameters are registered correctly 

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # Final LayerNorm (because we are using pre-LN)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x:torch.Tensor, mask:torch.Tensor): 

        # pass through each encoder layer 
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # applying final layer norm before sending to the decoder 
        x = self.norm(x) 
        return x 


class Decoder(nn.Module): 
    def __init__(self, d_model:int, n_layers:int, n_heads:int, d_ff:int, dropout=0.1):
        super().__init__() 

        # create a list of n DecoderLayer instances 
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model) 
    
    def forward(self, x:torch.Tensor, memory:torch.Tensor, src_mask:torch.Tensor, tgt_mask:torch.Tensor):

        # pass through each decoeder layer 
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        
        # final layernorm 
        x = self.norm(x)
        return x
