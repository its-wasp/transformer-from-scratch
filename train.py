import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.architecture import Transformer
from src.dataset import CopyDataset
from src.utils import make_src_mask, make_tgt_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    """
    Autoregressive decoding for inference and validation.
    """
    model.eval()
    with torch.no_grad():
        memory = model.encoder(model.src_embed(src), src_mask)
    
    batch_size = src.size(0)
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    
    for i in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=0)
        with torch.no_grad():
            out = model.decoder(model.tgt_embed(ys), memory, src_mask, tgt_mask)
            prob = model.generator(out[:, -1])
            
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
        
        if (next_word == end_symbol).all():
            break
            
    return ys

def validate_model(model, dataloader, device, pad_idx, start_symbol, end_symbol, max_len):
    model.eval()
    correct_sequences = 0
    total_sequences = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = make_src_mask(src, pad_idx)
            
            decoded = greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device)
            
            batch_size = src.size(0)
            for i in range(batch_size):
                total_sequences += 1
                actual_clean = [x for x in tgt[i].tolist() if x not in [0, 1, 2]]
                
                predicted_raw = decoded[i].tolist()
                predicted_clean = []
                for x in predicted_raw:
                    if x == 2: break 
                    if x not in [0, 1]: predicted_clean.append(x)
                
                if actual_clean == predicted_clean:
                    correct_sequences += 1
                    
    return (correct_sequences / total_sequences) * 100

def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx):
    """
    Standard training pass with Teacher Forcing.
    """
    model.train()
    total_loss = 0
    
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        
        # Teacher Forcing: tgt_input vs tgt_label
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]
        
        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx)
        
        optimizer.zero_grad()
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        # Calculate loss (flattening B, S, V -> B*S, V)
        loss = criterion(
            logits.contiguous().view(-1, logits.size(-1)), 
            tgt_label.contiguous().view(-1)
        )
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def main():
    # 1. Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 20  
    d_model = 128
    n_layers = 3
    n_heads = 8
    d_ff = 512
    dropout = 0.1
    max_seq_len = 50
    batch_size = 32
    lr = 5e-4        # Higher learning rate for the copy task
    epochs = 40      
    pad_idx = 0
    start_symbol = 1
    end_symbol = 2

    # 2. Initialization
    model = Transformer(vocab_size, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_seq_len).to(device)
    
    train_dataset = CopyDataset(vocab_size, seq_len=10, num_samples=10000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = CopyDataset(vocab_size, seq_len=10, num_samples=200)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(f"Starting training on {device}...")

    # 3. Execution Loop
    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device, pad_idx)
        
        if (epoch + 1) % 5 == 0:
            val_acc = validate_model(model, val_loader, device, pad_idx, start_symbol, end_symbol, max_len=12)
            print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        else:
            print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "transformer_copy_model.pth")
    print("Training Complete!")

if __name__ == "__main__":
    main()