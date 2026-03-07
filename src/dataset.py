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
        # Reserve: 0 for <PAD>, 1 for <SOS>, 2 for <EOS>
        # Generate random numbers from [3, vocab_size)
        data = torch.randint(3, self.vocab_size, (self.seq_len,))
        
        # src: Just the random numbers
        src = data
        
        # tgt: Prepend <SOS> and Append <EOS>
        # This makes tgt_len = seq_len + 2
        sos = torch.tensor([1])
        eos = torch.tensor([2])
        tgt = torch.cat([sos, data, eos])
        
        return src, tgt