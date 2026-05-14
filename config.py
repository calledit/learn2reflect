from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Tokenizer — fixed at 256 for byte-level encoding
    vocab_size: int = 256

    # Architecture
    d_model: int = 128
    n_heads: int = 8        # head_dim = d_model // n_heads
    n_layers: int = 8
    dropout: float = 0.0

    # Sequence length — attention window and training chunk size
    context_length: int = 128

    # Training
    lr: float = 1.0           # Prodigy scale factor — keep at 1.0
    max_iters: int = 4500_000
    eval_interval: int = 250
    eval_iters: int = 50      # number of context-length chunks evaluated per val check
    grad_clip: float = 1.0

    # Dataset — "wikitext103", "fineweb_edu", or "oasst2"
    dataset: str = "fineweb_edu"

    # Checkpointing
    checkpoint_interval: int = 5000
    checkpoint_dir: str = "checkpoints"

    # Inference
    max_new_tokens: int = 200
    temperature: float = 0.8

    # Batched training
    batch_size: int = 256

    # Reflection loss weight
    reflection_loss_weight: float = 1.0

    # Steps before reflector training begins
    reflection_start_iter: int = 5000

    # Reflection transformer
    reflection_d_model: int = 64
    reflection_n_heads: int = 4

    # Function groups — MLP shape per head: head_dim → fn_hidden1 → fn_hidden2 → fn_hidden1 → d_model
    fn_hidden1:               int   = 192
    fn_hidden2:               int   = 256
    fn_isolation_steps:       int   = 10         # isolated training steps per selected group per batch
    fn_isolation_lr:          float = 3e-4       # AdamW LR for isolated training
    selection_temperature:    float = 0.9        # softmax temperature when sampling group selection
    causal_finder_start_iter: int   = 10000       # gate: wait for reflector to calibrate first

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
