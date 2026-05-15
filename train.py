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


def save_checkpoint(model, reflector, gen_optimizer, ref_optimizer, fn_optimizer,
                    group_reward_ema, group_reward_var_ema, grad_updates, docs_consumed, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{grad_updates:07d}.pt")
    torch.save({
        "model_state":         model.state_dict(),
        "reflector_state":     reflector.state_dict(),
        "gen_opt_state":       gen_optimizer.state_dict(),
        "ref_opt_state":       ref_optimizer.state_dict(),
        "fn_opt_state":        fn_optimizer.state_dict(),
        "group_reward_ema":    group_reward_ema.cpu(),
        "group_reward_var_ema": group_reward_var_ema.cpu(),
        "grad_updates":        grad_updates,
        "docs_consumed":       docs_consumed,
        "cfg":                 cfg,
    }, path)
    print(f"  [ckpt] step {grad_updates} → {path}")


# ─────────────────────────────────────────────────────────────────── evaluation

@torch.no_grad()
def evaluate(model: Generator, val_data: torch.Tensor, cfg: Config, offset: int = 0) -> tuple[float, int]:
    model.eval()
    device   = next(model.parameters()).device
    total    = 0.0
    n_chunks = (len(val_data) - 1) // cfg.context_length
    for k in range(cfg.eval_iters):
        idxs = [((offset + k * cfg.eval_batch_size + j) % n_chunks) * cfg.context_length
                for j in range(cfg.eval_batch_size)]
        x = torch.stack([val_data[i     : i + cfg.context_length    ] for i in idxs]).to(device)
        y = torch.stack([val_data[i + 1 : i + cfg.context_length + 1] for i in idxs]).to(device)
        logits, _ = model(x)
        total += F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)).item()
    model.train()
    return total / cfg.eval_iters, (offset + cfg.eval_iters * cfg.eval_batch_size) % n_chunks


# ──────────────────────────────────────────────────────────────────── training

def train():
    cfg           = Config()
    _config_mtime = os.path.getmtime(_config_module.__file__)
    device        = torch.device(cfg.device)
    print(f"Device: {device}")

    skip_docs = 0
    ckpt_path_early = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path_early:
        _peek     = torch.load(ckpt_path_early, map_location="cpu", weights_only=False)
        skip_docs = _peek.get("docs_consumed", 0)
        del _peek
        if skip_docs:
            print(f"Checkpoint reports {skip_docs:,} documents consumed — will skip forward in stream")

    train_dataset, val_data, tokenizer = build_dataset(cfg, skip_docs)
    print(f"Vocab: {cfg.vocab_size} | context_length: {cfg.context_length}")

    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"batch_size={cfg.batch_size}")

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
    fn_params    = [p for block in model.blocks for fg in block.fn_groups for p in fg.parameters()]
    fn_optimizer = torch.optim.AdamW(fn_params, lr=cfg.fn_isolation_lr)

    n_groups             = cfg.n_layers * cfg.n_heads
    group_reward_ema     = torch.zeros(n_groups, device=device)
    group_reward_var_ema = torch.ones( n_groups, device=device)
    ema_alpha            = 0.01

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
        try:
            if "fn_opt_state" in ckpt:
                fn_optimizer.load_state_dict(ckpt["fn_opt_state"])
        except (ValueError, KeyError, RuntimeError):
            print("  fn optimizer state incompatible — starting fresh")
        if "group_reward_ema" in ckpt:
            group_reward_ema     = ckpt["group_reward_ema"].to(device)
            group_reward_var_ema = ckpt["group_reward_var_ema"].to(device)
        grad_updates = ckpt["grad_updates"]
        print(f"  resumed at step {grad_updates}")
    else:
        print("No checkpoint found — starting from scratch")

    val_data   = val_data.to(device)
    val_offset = 0

    # ── logging ───────────────────────────────────────────────────────────────
    log_path     = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    write_header = not os.path.exists(log_path)
    log_file     = open(log_path, "a", newline="")
    log_writer   = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["step", "train_loss", "primary_loss", "pred_loss", "reflection_loss", "val_loss", "lr", "elapsed_s", "tok_per_s"])

    last_checkpoint_saved = grad_updates // cfg.checkpoint_interval
    t0               = time.time()
    t_last_log       = t0
    train_loss_sum      = 0.0
    primary_loss_sum    = 0.0
    pred_loss_sum       = 0.0
    reflection_loss_sum = 0.0
    train_loss_count    = 0
    tokens_since_log    = 0
    selection_counts    = torch.zeros(n_groups, dtype=torch.long)

    gen_optimizer.zero_grad()
    ref_optimizer.zero_grad()

    for epoch in itertools.count(1):
        for batch in loader:
            if grad_updates >= cfg.max_iters:
                break

            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            B, T = x.shape

            causal_active      = (grad_updates >= cfg.reflection_start_iter and
                                  grad_updates >= cfg.causal_finder_start_iter)
            reflector_active   = grad_updates >= cfg.reflection_start_iter
            run_isolation      = causal_active and (grad_updates % cfg.isolation_interval == 0)

            # ── Stage 1: free forward/backward pass + causal finder selection ──
            logits, hidden_states = model(x)
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction='none'
            ).reshape(B, T)
            primary_loss = per_token_loss.mean()

            if torch.isnan(primary_loss):
                print(f"WARNING: NaN loss at step {grad_updates + 1}, skipping batch")
                gen_optimizer.zero_grad()
                ref_optimizer.zero_grad()
                continue

            selected_layer = selected_head = sel_idx = None
            reflection_mse = torch.tensor(0.0, device=device)
            loss_pred      = torch.zeros(B, T, device=device)
            b_hard = t_hard = None

            if reflector_active:
                if run_isolation:
                    flat_idx = per_token_loss.detach().reshape(-1).argmax()
                    b_hard   = (flat_idx // T).item()
                    t_hard   = (flat_idx  % T).item()
                    loss_pred, sel_logits = reflector(
                        x, [h.detach() for h in hidden_states],
                        return_selection=True, selection_token=(b_hard, t_hard),
                    )
                    dist           = torch.distributions.Categorical(
                        logits=sel_logits / cfg.selection_temperature
                    )
                    sel_idx        = dist.sample()
                    selected_layer = (sel_idx // cfg.n_heads).item()
                    selected_head  = (sel_idx  % cfg.n_heads).item()
                else:
                    loss_pred = reflector(x, [h.detach() for h in hidden_states])
                reflection_mse = F.mse_loss(loss_pred, per_token_loss.detach())

            primary_loss.backward()

            if reflector_active:
                ref_loss = cfg.reflection_loss_weight * reflection_mse
                if run_isolation and cfg.selector_train_grad_norm:
                    fg_grad_norms = [
                        sum(p.grad.norm().item() ** 2 for p in fg.parameters() if p.grad is not None) ** 0.5
                        for block in model.blocks for fg in block.fn_groups
                    ]
                    grad_target = int(max(range(len(fg_grad_norms)), key=lambda i: fg_grad_norms[i]))
                    sel_supervision = F.cross_entropy(
                        sel_logits.unsqueeze(0),
                        torch.tensor([grad_target], device=device),
                    )
                    ref_loss = ref_loss + cfg.selector_loss_weight * sel_supervision
                ref_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            gen_optimizer.step();  gen_optimizer.zero_grad()
            if reflector_active:
                torch.nn.utils.clip_grad_norm_(reflector.parameters(), cfg.grad_clip)
                ref_optimizer.step();  ref_optimizer.zero_grad()
            del hidden_states

            # ── Stages 2-4: isolation on single hardest token ────────────────
            fg_baseline = fg_final = 0.0
            if run_isolation:
                worst = b_hard
                sx    = x[worst:worst+1]
                sy    = y[worst:worst+1]

                # ── Stage 2: single example caching pass ──────────────────────
                with torch.no_grad():
                    logits_s, hidden_states_s, (cache_h, cache_heads, cache_fn_outs) = model(
                        sx, cache_at_layer=selected_layer
                    )
                    fg_baseline = F.cross_entropy(
                        logits_s[0, t_hard, :].unsqueeze(0),
                        sy[0, t_hard].unsqueeze(0),
                    ).item()

                sibling_sum  = sum(fo for j, fo in enumerate(cache_fn_outs) if j != selected_head)
                cached_head_ = cache_heads[:, :, selected_head, :].clone()

                # ── Fresh reflector forward on single example (live graph) ─────
                _, single_sel_logits = reflector(
                    sx, [h.detach() for h in hidden_states_s],
                    return_selection=True, selection_token=(0, t_hard),
                )
                single_dist = torch.distributions.Categorical(
                    logits=single_sel_logits / cfg.selection_temperature
                )
                log_prob = single_dist.log_prob(sel_idx)

                # ── Stage 3: isolated training of selected function group ──────
                for p in model.parameters():
                    p.requires_grad_(False)
                sel_fg = model.blocks[selected_layer].fn_groups[selected_head]
                for p in sel_fg.parameters():
                    p.requires_grad_(True)

                for _ in range(cfg.fn_isolation_steps):
                    fn_optimizer.zero_grad()
                    iso_loss = model.forward_from_cache(
                        selected_layer, cache_h, selected_head, cached_head_, sibling_sum, sy,
                        token_idx=t_hard,
                    )
                    iso_loss.backward()
                    fn_optimizer.step()

                fg_final = iso_loss.item()

                for p in model.parameters():
                    p.requires_grad_(True)

                # ── Stage 4: REINFORCE update for selection head ───────────────
                g = selected_layer * cfg.n_heads + selected_head
                selection_counts[g] += 1
                if cfg.selector_train_reinforce:
                    raw_reward = (fg_baseline - fg_final) / (abs(fg_baseline) + 1e-8)
                    group_reward_ema[g]     = (1 - ema_alpha) * group_reward_ema[g]     + ema_alpha * raw_reward
                    err                     = raw_reward - group_reward_ema[g].item()
                    group_reward_var_ema[g] = (1 - ema_alpha) * group_reward_var_ema[g] + ema_alpha * err * err
                    norm_reward             = err / (group_reward_var_ema[g].sqrt().item() + 1e-8)
                    sel_loss = -(log_prob * norm_reward) - cfg.entropy_bonus * single_dist.entropy()
                    sel_loss.backward()
                    torch.nn.utils.clip_grad_norm_(reflector.parameters(), cfg.grad_clip)
                    ref_optimizer.step()
                    ref_optimizer.zero_grad()

            # ── Accumulators ──────────────────────────────────────────────────
            train_loss_sum      += primary_loss.item()
            primary_loss_sum    += primary_loss.item()
            if reflector_active:
                pred_loss_sum       += loss_pred.detach().mean().item()
                reflection_loss_sum += reflection_mse.item()
            train_loss_count    += 1
            tokens_since_log    += B * T
            grad_updates        += 1

            # ── Eval ──────────────────────────────────────────────────────────
            if grad_updates % cfg.eval_interval == 0:
                avg_train_loss      = train_loss_sum      / train_loss_count
                avg_primary_loss    = primary_loss_sum    / train_loss_count
                avg_pred_loss       = pred_loss_sum       / train_loss_count
                avg_reflection_loss = reflection_loss_sum / train_loss_count
                train_loss_sum = primary_loss_sum = pred_loss_sum = reflection_loss_sum = 0.0
                train_loss_count = 0

                val_loss, val_offset = evaluate(model, val_data, cfg, val_offset)
                elapsed   = time.time() - t0
                lr        = gen_optimizer.param_groups[0]['d'] * gen_optimizer.param_groups[0]['lr']
                tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
                t_last_log       = time.time()
                tokens_since_log = 0

                model.eval()
                sample = generate(model, tokenizer, cfg, prompt="The history of", max_new_tokens=20)
                model.train()
                sample_line = " ".join(sample.split())

                total_selections = selection_counts.sum().item()
                if total_selections > 0:
                    top_g     = selection_counts.argmax().item()
                    top_pct   = 100.0 * selection_counts[top_g].item() / total_selections
                    top_layer = top_g // cfg.n_heads
                    top_head  = top_g  % cfg.n_heads
                    sel_str   = f"top L{top_layer}H{top_head} {top_pct:.0f}%"
                else:
                    sel_str   = "sel n/a"
                selection_counts.zero_()

                print(
                    f"{grad_updates:7d}/{cfg.max_iters} | "
                    f"pl {avg_primary_loss:.4f} | "
                    f"rl {avg_reflection_loss:.4f} | "
                    f"pdl {avg_pred_loss:.4f} | "
                    f"vl {val_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"{tok_per_s/1000:.0f}k tok/s | "
                    f"{sel_str} | "
                    f"t: {elapsed:.0f}s | "
                    f"samp: {sample_line}"
                )
                log_writer.writerow([
                    grad_updates,
                    f"{avg_train_loss:.6f}",
                    f"{avg_primary_loss:.6f}",
                    f"{avg_pred_loss:.6f}",
                    f"{avg_reflection_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{lr:.6e}",
                    f"{elapsed:.1f}",
                    f"{tok_per_s:.0f}",
                ])
                log_file.flush()

            # ── Checkpoint ────────────────────────────────────────────────────
            current_interval = grad_updates // cfg.checkpoint_interval
            if current_interval > last_checkpoint_saved:
                save_checkpoint(
                    model, reflector, gen_optimizer, ref_optimizer, fn_optimizer,
                    group_reward_ema, group_reward_var_ema,
                    grad_updates, skip_docs + train_dataset.docs_consumed, cfg,
                )
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

    save_checkpoint(
        model, reflector, gen_optimizer, ref_optimizer, fn_optimizer,
        group_reward_ema, group_reward_var_ema,
        grad_updates, skip_docs + train_dataset.docs_consumed, cfg,
    )
    log_file.close()
    print(f"\nTraining done in {time.time() - t0:.0f}s")
    return model, tokenizer, cfg


if __name__ == "__main__":
    train()
