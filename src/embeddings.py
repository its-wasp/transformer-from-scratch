import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Standard lookup table that converts token IDs into dense vectors.
    Follows the "Attention Is All You Need" convention of scaling
    the embeddings by sqrt(d_model).
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) — integer token IDs
        Returns:
            (batch_size, seq_len, d_model) — scaled embedding vectors
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The encoding is computed once and registered as a non-trainable buffer.
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()

        # (max_seq_len, d_model) matrix of zeros
        pe = torch.zeros(max_seq_len, d_model)

        # Column vector of positions: (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Denominator term in log-space for numerical stability:
        # 10000^(2i / d_model) = exp(2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer — saved with model state but not trained
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model) — any tensor whose seq_len
               we use to slice the positional encoding
        Returns:
            (batch_size, seq_len, d_model) — the positional encoding
               (broadcast-added by the caller)
        """
        return self.pe[:, : x.size(1), :]


class TransformerEmbeddings(nn.Module):
    """
    Combines TokenEmbedding + PositionalEncoding + Dropout.

    Pipeline:  token_ids  →  scaled embeddings  +  positional encoding  →  dropout
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) — integer token IDs
        Returns:
            (batch_size, seq_len, d_model) — embedded + position-encoded + dropped-out
        """
        tok_emb = self.token_embedding(x)           # (B, S, D)
        pos_enc = self.positional_encoding(tok_emb)  # (1, S, D) broadcast-ready
        return self.dropout(tok_emb + pos_enc)
