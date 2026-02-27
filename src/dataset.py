import torch
from torch.utils.data import Dataset

class CopyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random indices from [2, vocab_size) 
        # (0 and 1 are usually reserved for <PAD> and <SOS>)
        data = torch.randint(2, self.vocab_size, (self.seq_len,))
        
        # In a copy task, the input (src) and the output (tgt) are identical
        # However, for the Transformer, we usually shift the target for the decoder
        src = data
        tgt = data
        return src, tgt