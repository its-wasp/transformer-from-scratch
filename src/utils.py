import torch

def make_src_mask(src, pad_idx):
    """
    Creates a mask for the source sequence to ignore padding tokens.
    Shape: (batch_size, 1, 1, src_seq_len)
    """
    # Returns True for tokens that are NOT padding
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

def make_tgt_mask(tgt, pad_idx):
    """
    Creates a mask for the target sequence that combines:
    1. Padding mask (ignore <PAD>)
    2. Causal mask (prevent looking at future tokens)
    Shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)
    """
    # 1. Padding Mask: (batch_size, 1, 1, tgt_seq_len)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # 2. Causal Mask: (1, tgt_seq_len, tgt_seq_len)
    # torch.triu returns the upper triangular part of a matrix.
    # We want a lower triangular matrix, so we use diagonal=1 and look for 0s.
    seq_len = tgt.size(1)
    no_peek_mask = torch.triu(torch.ones((1, seq_len, seq_len), device=tgt.device), diagonal=1).type(torch.uint8)
    no_peek_mask = no_peek_mask == 0
    
    # 3. Combine: Both conditions must be met (Logical AND)
    tgt_mask = tgt_pad_mask & no_peek_mask
    return tgt_mask