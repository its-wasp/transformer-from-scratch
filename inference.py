import torch
from src.architecture import Transformer
from src.utils import make_src_mask, make_tgt_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    model.eval()
    # 1. Encode the source once
    with torch.no_grad():
        memory = model.encoder(model.src_embed(src), src_mask)
    
    # 2. Initialize target sequence with <SOS>
    # Shape: (batch_size, 1)
    batch_size = src.size(0)
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    
    for i in range(max_len - 1):
        # Create mask for the current generated sequence
        tgt_mask = make_tgt_mask(ys, pad_idx=0)
        
        # 3. Decode
        with torch.no_grad():
            out = model.decoder(model.tgt_embed(ys), memory, src_mask, tgt_mask)
            prob = model.generator(out[:, -1]) # Look at the last generated token's logits
            
        # 4. Pick the next word (Greedy)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(1)
        
        # 5. Append to the sequence
        ys = torch.cat([ys, next_word], dim=1)
        
        # If all batches generated <EOS>, we could stop early (optional optimization)
        if (next_word == end_symbol).all():
            break
            
    return ys

def run_example():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Must match your training hyperparameters!
    vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_seq_len = 20, 128, 3, 8, 512, 0.1, 50
    
    # Load Model
    model = Transformer(vocab_size, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_seq_len).to(device)
    model.load_state_dict(torch.load("transformer_copy_model.pth", map_location=device))
    
    # Create a dummy "unseen" sequence to copy
    # Numbers 3-19 (since 0,1,2 are reserved)
    src = torch.tensor([[10, 11, 12, 10, 11, 9, 7, 5, 6, 3]]).to(device)
    src_mask = make_src_mask(src, pad_idx=0)
    
    print(f"Input Sequence:  {src.tolist()[0]}")
    
    # Run Inference
    output_indices = greedy_decode(model, src, src_mask, max_len=12, start_symbol=1, end_symbol=2, device=device)
    
    print(f"Decoded Output: {output_indices.tolist()[0]}")

if __name__ == "__main__":
    run_example()