import csv
import glob
import importlib
import itertools
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from prodigyopt import Prodigy

import config as _config_module
from config import Config
from model import Generator
from data import build_dataset
from inference import generate


# ──────────────────────────────────────────────────────────────── checkpointing

def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    import re
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    if not files:
        return None
    def _step(f):
        m = re.search(r"checkpoint_(\d+)\.pt", os.path.basename(f))
        return int(m.group(1)) if m else -1
    return max(files, key=_step)


def save_checkpoint(model, optimizer, scheduler, phase2_optimizer, grad_updates, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    data = {
        "model_state":        model.state_dict(),
        "optimizer_state":    optimizer.state_dict(),
        "scheduler_state":    scheduler.state_dict(),
        "phase2_opt_state":   phase2_optimizer.state_dict(),
        "grad_updates":       grad_updates,
        "cfg":                cfg,
    }
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{grad_updates:07d}.pt")
    torch.save(data, path)
    print(f"  [ckpt] step {grad_updates} → {path}")



# ─────────────────────────────────────────────────────────────────── evaluation

@torch.no_grad()
def evaluate(model: Generator, val_data: torch.Tensor, cfg: Config) -> float:
    model.eval()
    device = next(model.parameters()).device
    total  = 0.0
    n      = 0
    for i in range(0, len(val_data) - cfg.context_length - 1, cfg.context_length):
        x = val_data[i     : i + cfg.context_length    ].unsqueeze(0).to(device)
        y = val_data[i + 1 : i + cfg.context_length + 1].unsqueeze(0).to(device)
        logits, _ = model(x)
        total += F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)).item()
        n += 1
        if n >= cfg.eval_iters:
            break
    model.train()
    return total / max(n, 1)


# ──────────────────────────────────────────────────────────────────── training

def train():
    cfg          = Config()
    _config_mtime = os.path.getmtime(_config_module.__file__)
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    train_dataset, val_data, tokenizer = build_dataset(cfg)
    print(f"Vocab: {cfg.vocab_size} | context_length: {cfg.context_length}")

    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"batch_size={cfg.batch_size} | grad_accum_steps={cfg.grad_accum_steps} | effective_batch={cfg.batch_size * cfg.grad_accum_steps}")

    model = Generator(cfg).to(device)

    n_param = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_param:,}")

    optimizer = Prodigy(
        model.parameters(), lr=cfg.lr, weight_decay=0.1,
        safeguard_warmup=True, use_bias_correction=True,
    )
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_iters)
    phase2_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.phase2_lr)

    # ── resume from checkpoint ────────────────────────────────────────────────
    grad_updates = 0
    ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if "phase2_opt_state" in ckpt:
            phase2_optimizer.load_state_dict(ckpt["phase2_opt_state"])
        grad_updates = ckpt["grad_updates"]
        print(f"  resumed at step {grad_updates}")
    else:
        print("No checkpoint found — starting from scratch")

    val_data = val_data.to(device)

    # ── logging setup ─────────────────────────────────────────────────────────
    log_path   = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    write_header = not os.path.exists(log_path)
    log_file   = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["step", "train_loss", "primary_loss", "reflection_loss", "phase2_loss", "val_loss", "lr", "elapsed_s", "tok_per_s"])

    last_checkpoint_saved = grad_updates // cfg.checkpoint_interval
    t0              = time.time()
    t_last_log      = t0
    train_loss_sum      = 0.0
    primary_loss_sum    = 0.0
    reflection_loss_sum = 0.0
    phase2_loss_sum     = 0.0
    phase2_loss_count   = 0
    train_loss_count    = 0
    tokens_since_log    = 0
    micro_step          = 0

    optimizer.zero_grad()

    for epoch in itertools.count(1):
        for batch in loader:
            if grad_updates >= cfg.max_iters:
                break

            # batch: [B, context_length + 1]
            batch = batch.to(device)
            x = batch[:, :-1]   # [B, T]
            y = batch[:, 1:]    # [B, T]

            logits, loss_pred = model(x)   # [B, T, vocab_size], [B, T]

            B, T = x.shape
            per_token_loss  = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction='none'
            ).reshape(B, T)                                          # [B, T]
            primary_loss    = per_token_loss.mean()
            reflection_loss = F.mse_loss(loss_pred, per_token_loss.detach())
            loss            = primary_loss + cfg.reflection_loss_weight * reflection_loss

            if torch.isnan(loss):
                print(f"WARNING: NaN loss at step {grad_updates + 1}, skipping micro-batch")
                optimizer.zero_grad()
                micro_step = 0
                continue

            # Scale loss so accumulated gradients match a single large-batch update
            (loss / cfg.grad_accum_steps).backward()

            # Track unscaled values for logging
            train_loss_sum      += loss.item()
            primary_loss_sum    += primary_loss.item()
            reflection_loss_sum += reflection_loss.item()
            train_loss_count    += 1
            tokens_since_log    += batch.size(0) * cfg.context_length
            micro_step          += 1

            if micro_step < cfg.grad_accum_steps:
                continue

            # ── Phase 1 optimizer step ───────────────────────────────────────
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            micro_step   = 0
            grad_updates += 1

            # ── Phase 2 step ─────────────────────────────────────────────────
            if (grad_updates >= cfg.phase2_start_iter
                    and grad_updates % cfg.phase2_interval == 0):
                phase2_optimizer.zero_grad()
                p2_loss_accum = 0.0
                n_accum = cfg.phase2_grad_accum_steps
                p2_params = list(model.parameters())
                for i in range(n_accum):
                    idx = (i * cfg.phase2_batch_size) % batch.size(0)
                    x1 = batch[idx : idx + cfg.phase2_batch_size, :-1]
                    y1 = batch[idx : idx + cfg.phase2_batch_size, 1:]
                    logits_p2, loss_pred_p2 = model(x1)
                    B2, T2 = x1.shape
                    per_token_loss_p2 = F.cross_entropy(
                        logits_p2.reshape(-1, cfg.vocab_size), y1.reshape(-1), reduction='none'
                    ).reshape(B2, T2)
                    primary_loss_p2 = per_token_loss_p2.mean()
                    # Reflection: isolate grads, scale, manually accumulate
                    r_grads = torch.autograd.grad(
                        loss_pred_p2.mean() / n_accum, p2_params,
                        retain_graph=True, allow_unused=True,
                    )
                    for (name, param), g in zip(model.named_parameters(), r_grads):
                        if g is None:
                            continue
                        if name.startswith('blocks.'):
                            li = int(name.split('.')[1])
                            scale = (cfg.n_layers - li) / cfg.n_layers
                        elif name.startswith('tok_emb') or name.startswith('pos_emb'):
                            scale = 1.0
                        else:
                            scale = 0.0
                        if param.grad is None:
                            param.grad = g * scale
                        else:
                            param.grad.add_(g * scale)
                    # Primary: normal gradient flow accumulated on top
                    (primary_loss_p2 / n_accum).backward()
                    p2_loss_accum += (primary_loss_p2 + loss_pred_p2.mean()).item() / n_accum
                phase2_optimizer.step()
                phase2_loss_sum   += p2_loss_accum
                phase2_loss_count += 1

            if grad_updates % cfg.eval_interval == 0:
                avg_train_loss      = train_loss_sum      / train_loss_count
                avg_primary_loss    = primary_loss_sum    / train_loss_count
                avg_reflection_loss = reflection_loss_sum / train_loss_count
                avg_phase2_loss     = phase2_loss_sum / phase2_loss_count if phase2_loss_count else 0.0
                train_loss_sum      = 0.0
                primary_loss_sum    = 0.0
                reflection_loss_sum = 0.0
                phase2_loss_sum     = 0.0
                phase2_loss_count   = 0
                train_loss_count    = 0

                val_loss  = evaluate(model, val_data, cfg)
                elapsed   = time.time() - t0
                lr        = scheduler.get_last_lr()[0]
                tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
                t_last_log      = time.time()
                tokens_since_log = 0

                model.eval()
                sample = generate(model, tokenizer, cfg, prompt="The history of", max_new_tokens=20)
                model.train()
                sample_line = " ".join(sample.split())

                p2_str = f"p2_loss {avg_phase2_loss:.4f} | " if phase2_loss_count else ""
                print(
                    f"step {grad_updates:7d}/{cfg.max_iters} | "
                    f"t_loss {avg_train_loss:.4f} | "
                    f"p_loss {avg_primary_loss:.4f} | "
                    f"r_loss {avg_reflection_loss:.4f} | "
                    f"{p2_str}"
                    f"v_loss {val_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"{tok_per_s:,.0f} tok/s | "
                    f"time: {elapsed:.0f}s | "
                    f"samp: {sample_line}"
                )
                log_writer.writerow([
                    grad_updates,
                    f"{avg_train_loss:.6f}",
                    f"{avg_primary_loss:.6f}",
                    f"{avg_reflection_loss:.6f}",
                    f"{avg_phase2_loss:.6f}" if phase2_loss_count else "",
                    f"{val_loss:.6f}",
                    f"{lr:.6e}",
                    f"{elapsed:.1f}",
                    f"{tok_per_s:.0f}",
                ])
                log_file.flush()

            current_interval = grad_updates // cfg.checkpoint_interval
            if current_interval > last_checkpoint_saved:
                save_checkpoint(model, optimizer, scheduler, phase2_optimizer, grad_updates, cfg)
                last_checkpoint_saved = current_interval

                new_mtime = os.path.getmtime(_config_module.__file__)
                if new_mtime != _config_mtime:
                    _config_mtime = new_mtime
                    answer = input("  [config] config.py changed — reload? [y/N] ").strip().lower()
                    if answer == "y":
                        importlib.reload(_config_module)
                        cfg = _config_module.Config()
                        print("  [config] reloaded")

        if grad_updates >= cfg.max_iters:
            break

    save_checkpoint(model, optimizer, scheduler, phase2_optimizer, grad_updates, cfg)
    log_file.close()
    print(f"\nTraining done in {time.time() - t0:.0f}s")
    return model, tokenizer, cfg


if __name__ == "__main__":
    train()
