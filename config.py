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
    max_iters: int = 500_000
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

    # Phase 2 — adversarial self-stress
    phase2_start_iter:  int   = 20000   # wait for reflection head to calibrate
    phase2_interval:    int   = 10     # Phase 2 step every N Phase 1 steps
    phase2_lr:          float = 1e-4
    phase2_batch_size:       int   = 1    # 1 = idiosyncratic errors, 256 = systematic errors
    phase2_grad_accum_steps: int   = 16    # accumulate N samples before each Phase 2 step

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
