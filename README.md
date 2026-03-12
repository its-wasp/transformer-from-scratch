# Transformer from scratch

A from-scratch implementation of the Transformer architecture ("Attention Is All You Need") using PyTorch, validated on a sequence copy task.

## Architecture

The model follows the original Transformer encoder-decoder architecture:

```
Input Tokens --> [Token Embedding + Positional Encoding] --> Encoder Stack --> Memory
Target Tokens --> [Token Embedding + Positional Encoding] --> Decoder Stack --> Linear --> Logits
```

**Hyperparameters used for training:**

| Parameter | Value |
|-----------|-------|
| `d_model` | 128 |
| `n_layers` | 3 |
| `n_heads` | 8 |
| `d_ff` | 512 |
| `vocab_size` | 20 |
| `seq_len` | 10 |
| `dropout` | 0.1 |

## Project Structure

```
TransformerMadness/
├── src/
│   ├── embeddings.py      # Token embedding + sinusoidal positional encoding
│   ├── model.py           # MultiHeadAttention, FeedForward, Encoder, Decoder
│   ├── architecture.py    # Full Transformer class
│   ├── dataset.py         # CopyDataset (toy task)
│   └── utils.py           # Mask utilities (src padding mask, tgt causal mask)
├── experiments/
│   ├── tensor_pass_dimension_check.ipynb  # Interactive shape checks
│   └── e2e_dimension_check.py             # End-to-end dimension test
├── train.py               # Training loop with teacher forcing + validation
├── inference.py           # Greedy decoding on a trained model
└── transformer_copy_model.pth  # Trained model weights (~5.4 MB)
```

## The Copy Task

The model learns to copy an input sequence to the output — a standard sanity check for sequence-to-sequence models. If the Transformer can learn this, the attention mechanism is wiring correctly.

- **Input:** `[10, 11, 12, 10, 11, 9, 7, 5, 6, 3]`
- **Expected output:** `[1, 10, 11, 12, 10, 11, 9, 7, 5, 6, 3, 2]` (with `<SOS>=1` and `<EOS>=2`)

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install PyTorch (CPU)
pip install torch
```

## Usage

### Train
```bash
python train.py
```
Trains for 40 epochs with validation every 5 epochs.

### Inference
```bash
python inference.py
```
Loads the trained model and runs greedy decoding on a sample sequence.

### Dimension Check
```bash
python experiments/e2e_dimension_check.py
```
Traces tensor shapes through every stage of the pipeline to verify correctness.

## Key Implementation Details

- **Sinusoidal Positional Encoding** computed in log-space for numerical stability
- **Pre-LayerNorm** variant (norm before sublayer, not after)
- **Teacher Forcing** during training with shifted target sequences
- **Greedy Decoding** for inference (autoregressive, token-by-token)
- **Combined Target Mask** merges causal (no-peek) mask with padding mask
