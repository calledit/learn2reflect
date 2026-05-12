from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Tokenizer — fixed at 256 for byte-level encoding
    vocab_size: int = 256

    # Architecture
    d_model: int = 256
    n_heads: int = 8        # head_dim = 32
    n_layers: int = 8
    dropout: float = 0.1

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
    grad_accum_steps: int = 1     # effective batch size = batch_size * grad_accum_steps

    # Reflection loss weight (primary_loss + weight * reflection_loss)
    reflection_loss_weight: float = 1.0

    # Reflection transformer
    reflection_d_model: int = 64
    reflection_n_heads: int = 4

    # Phase 2 — causal correction (active after warmup)
    phase2_start_iter: int   = 585000   # wait for reflector to calibrate
    phase2_weight:     float = 0.1      # scale of phase2 gradient relative to primary

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
