# End-to-End Transformer Dimension Check
# =======================================
# Traces tensor shapes through every stage:
#   token_ids -> Embeddings -> Encoder -> Decoder -> Generator -> logits
#
# Run from project root:
#   python experiments/e2e_dimension_check.py

import sys
sys.path.insert(0, ".")

import torch
from src.architecture import Transformer
from src.dataset import CopyDataset

# -- Hyperparameters ---------------------------------------------------
VOCAB_SIZE  = 50
D_MODEL     = 64
N_LAYERS    = 2
N_HEADS     = 4
D_FF        = 256
DROPOUT     = 0.1
MAX_SEQ_LEN = 100
SEQ_LEN     = 12
BATCH_SIZE  = 4

print("Hyperparameters:")
print(f"  vocab_size={VOCAB_SIZE}, d_model={D_MODEL}, n_layers={N_LAYERS}")
print(f"  n_heads={N_HEADS}, d_ff={D_FF}, head_dim={D_MODEL // N_HEADS}")
print(f"  batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}\n")


# -- Step 1: Generate data from CopyDataset ---------------------------
dataset = CopyDataset(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, num_samples=100)

src_batch = torch.stack([dataset[i][0] for i in range(BATCH_SIZE)])
tgt_batch = torch.stack([dataset[i][1] for i in range(BATCH_SIZE)])

print(f"src (token IDs):  {src_batch.shape}  dtype={src_batch.dtype}")
print(f"tgt (token IDs):  {tgt_batch.shape}  dtype={tgt_batch.dtype}")
print(f"Expected:         (batch_size={BATCH_SIZE}, seq_len={SEQ_LEN})\n")


# -- Step 2: Create masks ---------------------------------------------
# Padding mask for encoder: (B, 1, 1, src_len)
# No padding in copy task, so all 1s
src_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN)
print(f"src_mask:  {src_mask.shape}  (all 1s - no padding)")

# Causal mask for decoder: (B, 1, tgt_len, tgt_len)
tgt_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).unsqueeze(0).unsqueeze(0)
tgt_mask = tgt_mask.expand(BATCH_SIZE, -1, -1, -1)
print(f"tgt_mask:  {tgt_mask.shape}  (causal lower-triangular)\n")


# -- Step 3: Build model & trace shapes -------------------------------
model = Transformer(
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF,
    dropout=DROPOUT,
    max_seq_len=MAX_SEQ_LEN,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}\n")

print("=" * 55)
print(f"{'STAGE':<35} {'SHAPE'}")
print("=" * 55)

print(f"{'Input src (token IDs)':<35} {tuple(src_batch.shape)}")
print(f"{'Input tgt (token IDs)':<35} {tuple(tgt_batch.shape)}")

# Embeddings
src_emb = model.src_embed(src_batch)
tgt_emb = model.tgt_embed(tgt_batch)
print(f"{'After src Embedding':<35} {tuple(src_emb.shape)}")
print(f"{'After tgt Embedding':<35} {tuple(tgt_emb.shape)}")

# Encoder
memory = model.encoder(src_emb, src_mask)
print(f"{'Encoder output (memory)':<35} {tuple(memory.shape)}")

# Decoder
dec_out = model.decoder(tgt_emb, memory, src_mask, tgt_mask)
print(f"{'Decoder output':<35} {tuple(dec_out.shape)}")

# Generator (final linear projection)
logits = model.generator(dec_out)
print(f"{'Generator output (logits)':<35} {tuple(logits.shape)}")

print("=" * 55)
print(f"\nExpected final shape: ({BATCH_SIZE}, {SEQ_LEN}, {VOCAB_SIZE})")
assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), "SHAPE MISMATCH!"
print("[PASS] Stage-by-stage dimension check PASSED!\n")


# -- Step 4: Verify model.forward() gives same result -----------------
output = model(src_batch, tgt_batch, src_mask, tgt_mask)
print(f"model.forward() output:  {tuple(output.shape)}")
assert output.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), "SHAPE MISMATCH!"
print("[PASS] Full forward pass matches!\n")

print("[PASS] ALL END-TO-END DIMENSION CHECKS PASSED!")
