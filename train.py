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
from model import Generator, ReflectionTransformer
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


def save_checkpoint(model, reflector, gen_optimizer, ref_optimizer, grad_updates, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    data = {
        "model_state":     model.state_dict(),
        "reflector_state": reflector.state_dict(),
        "gen_opt_state":   gen_optimizer.state_dict(),
        "ref_opt_state":   ref_optimizer.state_dict(),
        "grad_updates":    grad_updates,
        "cfg":             cfg,
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
    cfg           = Config()
    _config_mtime = os.path.getmtime(_config_module.__file__)
    device        = torch.device(cfg.device)
    print(f"Device: {device}")

    train_dataset, val_data, tokenizer = build_dataset(cfg)
    print(f"Vocab: {cfg.vocab_size} | context_length: {cfg.context_length}")

    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"batch_size={cfg.batch_size} | grad_accum_steps={cfg.grad_accum_steps} | effective_batch={cfg.batch_size * cfg.grad_accum_steps}")

    model     = Generator(cfg).to(device)
    reflector = ReflectionTransformer(cfg).to(device)

    n_param   = sum(p.numel() for p in model.parameters())
    n_param_r = sum(p.numel() for p in reflector.parameters())
    print(f"Generator parameters:   {n_param:,}")
    print(f"Reflector parameters:   {n_param_r:,}  ({100*n_param_r/n_param:.1f}% of generator)")

    gen_optimizer = Prodigy(
        model.parameters(), lr=cfg.lr, weight_decay=0.1,
        safeguard_warmup=True, use_bias_correction=True,
    )
    ref_optimizer = Prodigy(
        reflector.parameters(), lr=cfg.lr, weight_decay=0.1,
        safeguard_warmup=True, use_bias_correction=True,
    )

    # ── resume from checkpoint ────────────────────────────────────────────────
    grad_updates = 0
    ckpt_path    = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"  model load (strict=False): missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}")
        if "reflector_state" in ckpt:
            reflector.load_state_dict(ckpt["reflector_state"])
            print("  reflector state loaded")
        else:
            print("  reflector starting fresh (no prior state)")
        try:
            gen_optimizer.load_state_dict(ckpt.get("gen_opt_state") or ckpt["optimizer_state"])
        except (ValueError, KeyError, RuntimeError):
            print("  gen optimizer state incompatible — starting fresh")
        try:
            if "ref_opt_state" in ckpt:
                ref_optimizer.load_state_dict(ckpt["ref_opt_state"])
        except (ValueError, KeyError, RuntimeError):
            print("  ref optimizer state incompatible — starting fresh")
        grad_updates = ckpt["grad_updates"]
        print(f"  resumed at step {grad_updates}")
    else:
        print("No checkpoint found — starting from scratch")

    val_data = val_data.to(device)

    # ── logging ───────────────────────────────────────────────────────────────
    log_path     = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    write_header = not os.path.exists(log_path)
    log_file     = open(log_path, "a", newline="")
    log_writer   = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["step", "train_loss", "primary_loss", "pred_loss", "reflection_loss", "phase2_loss", "val_loss", "lr", "elapsed_s", "tok_per_s"])

    last_checkpoint_saved = grad_updates // cfg.checkpoint_interval
    t0                = time.time()
    t_last_log        = t0
    train_loss_sum      = 0.0
    primary_loss_sum    = 0.0
    pred_loss_sum       = 0.0
    reflection_loss_sum = 0.0
    phase2_loss_sum     = 0.0
    phase2_loss_count   = 0
    train_loss_count    = 0
    tokens_since_log    = 0
    micro_step          = 0

    gen_optimizer.zero_grad()
    ref_optimizer.zero_grad()
    gen_params = list(model.parameters())

    for epoch in itertools.count(1):
        for batch in loader:
            if grad_updates >= cfg.max_iters:
                break

            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            B, T = x.shape

            # ── Forward: generator ───────────────────────────────────────────
            logits, hidden_states = model(x)

            per_token_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction='none'
            ).reshape(B, T)
            primary_loss = per_token_loss.mean()

            if torch.isnan(primary_loss):
                print(f"WARNING: NaN loss at step {grad_updates + 1}, skipping micro-batch")
                gen_optimizer.zero_grad()
                ref_optimizer.zero_grad()
                micro_step = 0
                continue

            # ── Forward: reflector (detached) — trains reflector only ────────
            loss_pred    = reflector(x, [h.detach() for h in hidden_states])
            reflection_mse = F.mse_loss(loss_pred, per_token_loss.detach())

            # ── Phase 2: reflector (connected) — gradient into generator ─────
            phase2_active = grad_updates >= cfg.phase2_start_iter
            if phase2_active:
                loss_pred_p2 = reflector(x, hidden_states)
                # Gradient flows through reflector into generator; retain_graph
                # so the generator graph is intact for primary_loss.backward().
                p2_grads = torch.autograd.grad(
                    loss_pred_p2.mean(), gen_params,
                    retain_graph=True, allow_unused=True,
                )
                for param, g in zip(gen_params, p2_grads):
                    if g is None:
                        continue
                    g_scaled = cfg.phase2_weight * g / cfg.grad_accum_steps
                    if param.grad is None:
                        param.grad = g_scaled
                    else:
                        param.grad.add_(g_scaled)

            # ── Backward: generator (primary loss) ───────────────────────────
            (primary_loss / cfg.grad_accum_steps).backward()

            # ── Backward: reflector (MSE, detached path — ref params only) ───
            (cfg.reflection_loss_weight * reflection_mse / cfg.grad_accum_steps).backward()

            # ── Accumulators ─────────────────────────────────────────────────
            train_loss_sum      += primary_loss.item()
            primary_loss_sum    += primary_loss.item()
            pred_loss_sum       += loss_pred.detach().mean().item()
            reflection_loss_sum += reflection_mse.item()
            if phase2_active:
                phase2_loss_sum   += loss_pred_p2.detach().mean().item()
                phase2_loss_count += 1
            train_loss_count    += 1
            tokens_since_log    += B * T
            micro_step          += 1

            if micro_step < cfg.grad_accum_steps:
                continue

            # ── Optimizer steps ──────────────────────────────────────────────
            torch.nn.utils.clip_grad_norm_(model.parameters(),     cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(reflector.parameters(), cfg.grad_clip)
            gen_optimizer.step()
            ref_optimizer.step()
            gen_optimizer.zero_grad()
            ref_optimizer.zero_grad()
            micro_step   = 0
            grad_updates += 1

            # ── Eval ─────────────────────────────────────────────────────────
            if grad_updates % cfg.eval_interval == 0:
                avg_train_loss      = train_loss_sum      / train_loss_count
                avg_primary_loss    = primary_loss_sum    / train_loss_count
                avg_pred_loss       = pred_loss_sum       / train_loss_count
                avg_reflection_loss = reflection_loss_sum / train_loss_count
                avg_phase2_loss     = phase2_loss_sum / phase2_loss_count if phase2_loss_count else 0.0
                train_loss_sum      = 0.0
                primary_loss_sum    = 0.0
                pred_loss_sum       = 0.0
                reflection_loss_sum = 0.0
                phase2_loss_sum     = 0.0
                phase2_loss_count   = 0
                train_loss_count    = 0

                val_loss  = evaluate(model, val_data, cfg)
                elapsed   = time.time() - t0
                lr        = gen_optimizer.param_groups[0]['d'] * gen_optimizer.param_groups[0]['lr']
                tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
                t_last_log       = time.time()
                tokens_since_log = 0

                model.eval()
                sample = generate(model, tokenizer, cfg, prompt="The history of", max_new_tokens=20)
                model.train()
                sample_line = " ".join(sample.split())

                p2_str = f"p2 {avg_phase2_loss:.4f} | " if avg_phase2_loss else ""
                print(
                    f"{grad_updates:7d}/{cfg.max_iters} | "
                    f"pl {avg_primary_loss:.4f} | "
                    f"rl {avg_reflection_loss:.4f} | "
                    f"pdl {avg_pred_loss:.4f} | "
                    f"{p2_str}"
                    f"vl {val_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"{tok_per_s/1000:.0f}k tok/s | "
                    f"t: {elapsed:.0f}s | "
                    f"samp: {sample_line}"
                )
                log_writer.writerow([
                    grad_updates,
                    f"{avg_train_loss:.6f}",
                    f"{avg_primary_loss:.6f}",
                    f"{avg_pred_loss:.6f}",
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
                save_checkpoint(model, reflector, gen_optimizer, ref_optimizer, grad_updates, cfg)
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

    save_checkpoint(model, reflector, gen_optimizer, ref_optimizer, grad_updates, cfg)
    log_file.close()
    print(f"\nTraining done in {time.time() - t0:.0f}s")
    return model, tokenizer, cfg


if __name__ == "__main__":
    train()
