# Architecture and system documentation

## What this system does

This is a language model training framework built around one central idea: instead of using a global loss signal to update the entire network uniformly, train a secondary network (the *reflector*) to understand where the primary network (the *generator*) is failing, and use that understanding to direct training effort to the specific part of the generator most responsible for the current error.

The result is a two-network system where the generator learns to predict text and the reflector learns to understand the generator's failures, then acts as a credit-assignment oracle that focuses isolated training on whichever part of the generator needs it most.

---

## The two networks

### Generator

The generator is a transformer language model. Its architecture departs from the standard transformer in one significant way: **there is no FFN and no output projection matrix (W_O)**. Instead, each attention head feeds directly into its own dedicated MLP called a *function group*.

A standard transformer block looks like:

```
x → LayerNorm → Attention → W_O → + → LayerNorm → FFN → +
```

A function group block looks like:

```
x → LayerNorm → Attention (8 heads, no W_O)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    FnGroup_0   FnGroup_1  ... FnGroup_7
        │           │           │
        └───────────┼───────────┘
                    ▼
                   sum
                    │
               LayerNorm
                    │
                  + x
```

Each function group is a 5-layer MLP that takes one head's output (16-dimensional at d_model=128, n_heads=8) and produces a full d_model=128 residual contribution:

```
head_dim(16) → 192 → 256 → 192 → d_model(128)
```

The eight residuals are summed and layer-normalised, then added to the stream. The next attention layer is immediately after — no FFN.

**Why this structure?** Each attention head naturally develops distinct behaviour (attending to different positions, relationships, or token types). Giving each head its own dedicated nonlinear transformation lets specialisation happen at the right granularity: the attention mechanism decides *what* to gather, the function group decides *what to do* with it. The structure also creates a clean unit for the causal finder to target — one function group = one attention head's entire downstream computation.

**Zero initialisation.** The output layer of every function group is initialised to zero weights, so all groups start silent and add nothing to the residual stream. Each group must earn its contribution through training rather than starting with a disruptive random effect.

### Reflector

The reflector is a smaller transformer that runs in parallel with the generator. It has the same number of layers but a smaller hidden dimension (d_model=64 by default). Each reflector block attends to its own sequence via causal self-attention, then cross-attends to the corresponding generator block's hidden states.

This layer-for-layer cross-attention is the key property: the reflector observes the generator's internal representations at every depth, not just the final output. This gives it everything it needs to understand where in the generator errors are originating.

The reflector has two output heads:

- **Loss prediction head** — a small MLP that produces a per-token scalar: the reflector's estimate of the generator's cross-entropy loss at each position. Trained by MSE against the actual per-token loss.
- **Selection head** — a small MLP applied to the mean-pooled final hidden state, producing one logit per function group across all layers (n_layers × n_heads = 64 logits by default). This head selects which function group to give isolated training to each step.

The reflector sees the generator's hidden states with gradients detached during the loss prediction path, so it calibrates to the generator's current behaviour without perturbing it. The selection head's gradient (from REINFORCE) flows only through the reflector's own parameters.

---

## The training loop

Every batch goes through four stages.

### Stage 1 — free forward/backward pass

Run the full generator on the batch. Compute the task loss (cross-entropy). Backprop and update the full generator via the Prodigy optimizer. This is equivalent to standard language model training.

If the reflector is past its warmup period, also run it on the same batch (with generator hidden states detached), compute MSE against the actual per-token loss, and backprop into the reflector. If the causal finder is also active, additionally run the selection head and sample a function group selection — recording the log-probability of that choice for use in stage 4.

**Important:** when the causal finder is active, the reflector optimizer is *not* stepped here. The computation graph from the reflector's forward pass is retained (via `retain_graph=True`) so stage 4 can backprop through it. Stepping the optimizer here would modify the reflector's parameters in-place, invalidating the retained graph before stage 4 can use it.

### Stage 2 — activation caching pass

Run the full generator again under `torch.no_grad()`. At the selected layer, call `get_cache()` on the block, which returns the full forward output plus the per-head attention outputs and the per-function-group residuals as separate tensors. Record the loss at the end of this pass as the **baseline loss** — this is the "before" measurement for the reward signal.

From these cached tensors, build two things for stage 3:
- `cached_head_out`: the selected head's attention output, detached. This is the input to the function group being trained.
- `sibling_sum`: the sum of all other function groups' residuals, detached. These are fixed during isolated training.

### Stage 3 — isolated training

Freeze the entire generator by setting `requires_grad=False` on all parameters. Then unfreeze only the selected function group.

Run N steps (default 10) of isolated training:
1. Feed `cached_head_out` into the selected function group (this is the only part of the computation in the gradient graph)
2. Add the fixed `sibling_sum` (detached)
3. Apply the block's LayerNorm and add to the cached layer input
4. Run all subsequent generator layers normally to compute the loss
5. Backprop and step only the function group's parameters via a separate AdamW optimizer

Gradients flow through the upper layers' *computations* (so the loss signal reaches the function group), but those layers' *parameters* don't accumulate gradients because they have `requires_grad=False`. After all N steps, record the **final loss**.

Restore `requires_grad=True` on all parameters.

### Stage 4 — causal finder REINFORCE update

Compute the reward:

```
raw_reward = (baseline_loss - final_loss) / (|baseline_loss| + ε)
```

This measures relative improvement — how much did training the selected function group actually help? Dividing by the baseline loss makes the reward scale-invariant as the overall loss decreases across training.

Normalise using a per-group exponential moving average of mean and variance:

```
normalised_reward = (raw_reward - group_ema_mean) / sqrt(group_ema_var + ε)
```

Each of the 64 function groups maintains its own EMA statistics. This removes between-group bias — groups in later layers naturally produce bigger absolute loss deltas, so without normalisation the selection head would learn to always pick those groups regardless of whether they're actually the bottleneck.

Apply REINFORCE: `selection_loss = -(log_prob × normalised_reward)`. Backprop through `log_prob`, which has a live computation graph pointing back to the reflector parameters from stage 1's retained graph. Step the reflector optimizer and zero gradients.

The net effect: if the selected group caused a big improvement (positive reward), the selection head is pushed to assign it higher probability in similar future situations. If the improvement was below average (negative normalised reward after normalisation), the head is pushed away from that selection.

---

## File reference

### `config.py`

All hyperparameters in one dataclass. Key groups:

| Parameter | Purpose |
|---|---|
| `d_model`, `n_heads`, `n_layers` | Generator size. Default: 128, 8, 8 |
| `context_length`, `batch_size` | Sequence and batch dimensions |
| `lr`, `grad_clip` | Prodigy scale factor and gradient clipping |
| `reflection_d_model`, `reflection_n_heads` | Reflector size. Default: 64, 4 |
| `reflection_start_iter` | Step at which reflector training begins |
| `fn_hidden1`, `fn_hidden2` | Function group MLP hidden dimensions (192, 256) |
| `fn_isolation_steps` | Number of isolated training steps per selected group per batch |
| `fn_isolation_lr` | AdamW learning rate for isolated training |
| `selection_temperature` | Softmax temperature for group selection sampling |
| `causal_finder_start_iter` | Step at which causal finder activates (after reflector warmup) |

### `model.py`

Contains all model code. Six classes:

**`GeneratorAttention`** — causal self-attention without an output projection. Returns per-head outputs as a `[B, T, n_heads, head_dim]` tensor instead of concatenating and projecting. The output projection is replaced by the function groups.

**`FunctionGroup`** — a single per-head MLP. Takes `[B, T, head_dim]`, applies a LayerNorm, runs through the bottleneck MLP, and returns a `[B, T, d_model]` residual. The output layer is zero-initialised.

**`FunctionGroupBlock`** — a full transformer block. Contains one `GeneratorAttention` and eight `FunctionGroup` instances. Has three forward methods:
- `forward()` — standard forward pass
- `get_cache()` — forward pass that also returns per-head attention outputs and per-function-group residuals for the caching pass
- `forward_cached()` — isolated-training forward where one group's output is live and the sibling sum is a fixed detached tensor

**`Generator`** — the main language model. Embedding → N × FunctionGroupBlock → LayerNorm → LM head (weight-tied to embedding). Also has `forward_from_cache()` which starts from a cached layer input and runs from a specified layer index through to the loss, used during isolated training.

**`CausalSelfAttention`**, **`CausalCrossAttention`**, **`ReflectionBlock`** — standard components used only by the reflector. `CausalSelfAttention` here retains the output projection unlike `GeneratorAttention`.

**`ReflectionTransformer`** — the reflector. Embedding → N × ReflectionBlock (each with self-attention, cross-attention to the corresponding generator layer, and FFN) → LayerNorm → two heads. `loss_head` produces per-token scalar predictions. `selection_head` produces `n_layers × n_heads` logits from the mean-pooled final hidden state. `forward()` takes a `return_selection` flag; when True, returns both outputs.

### `train.py`

Single `train()` function. Structure:

1. **Setup** — load config, build dataset, construct models and three optimizers: `gen_optimizer` (Prodigy, all generator params), `ref_optimizer` (Prodigy, all reflector params), `fn_optimizer` (AdamW, all function group params only).

2. **Checkpoint resume** — loads model state, reflector state, all three optimizer states, and the per-group EMA reward statistics. Uses `strict=False` on model loading for compatibility when resuming from architecturally different checkpoints.

3. **Training loop** — iterates over batches. Each batch runs the 4-stage loop described above. Eval runs every `eval_interval` steps (validation loss + a short generation sample). Checkpoints save every `checkpoint_interval` steps. Config can be hot-reloaded from disk at each checkpoint.

4. **`save_checkpoint()`** — saves model, reflector, all three optimizer states, and the EMA reward tensors.

5. **`evaluate()`** — runs the generator in eval mode on sequential chunks of the validation set, returns mean cross-entropy.

### `data.py`

Handles tokenization and dataset loading. `ByteTokenizer` encodes text as raw UTF-8 bytes (vocab size 256, no special tokens). `build_dataset()` returns a streaming training dataset and a flat validation tensor. Supports three datasets: `fineweb_edu` (default, streaming), `wikitext103`, and `oasst2`.

### `inference.py`

Loads a checkpoint and generates text. `generate()` does autoregressive sampling with temperature, maintaining a rolling context window. Can be run as a script:

```bash
python inference.py --prompt "The history of" --max_new_tokens 200
```

### `tools/plot_loss.py`

Reads `training_log.csv` from the checkpoint directory and plots training and validation loss curves.

---

## Optimizers and what they train

| Optimizer | Type | What it updates | When |
|---|---|---|---|
| `gen_optimizer` | Prodigy | All generator parameters | Stage 1, every batch |
| `ref_optimizer` | Prodigy | All reflector parameters | Stage 1 (MSE) + Stage 4 (REINFORCE), combined into one step when causal finder is active |
| `fn_optimizer` | AdamW | Function group parameters only | Stage 3, N times per batch |

Function group parameters appear in both `gen_optimizer` and `fn_optimizer`. They receive two gradient signals: the full task loss gradient (stage 1, via Prodigy), and the isolated focused gradient (stage 3, via AdamW). These are independent updates applied at different times with different learning rate regimes.

---

## Expected training dynamics

**Steps 0 to `reflection_start_iter`:** The generator trains on task loss alone. Function groups gradually specialise as the generator's ordinary gradient directs different patterns through different heads.

**Steps `reflection_start_iter` to `causal_finder_start_iter`:** The reflector begins training its loss prediction head. It observes the generator's hidden states layer by layer and learns to predict per-token loss. The selection head exists but is not yet used.

**Steps `causal_finder_start_iter` onwards:** All four stages activate. The reflector's credit assignment improves as it accumulates experience with which function groups cause which errors. The selection head learns to identify the highest-leverage group to train next. Groups that get repeatedly selected for similar error types will specialize more strongly in handling those patterns, which in turn makes them easier for the selector to identify, strengthening the specialisation further.

---

## Evaluation

Run two identical experiments — same seed, same architecture — differing only in `causal_finder_start_iter`:
- **Baseline:** set `causal_finder_start_iter` to a value larger than `max_iters` (causal finder never activates)
- **Full system:** default `causal_finder_start_iter`

Compare validation loss curves from `training_log.csv`. A secondary signal of the system working is whether function groups develop distinguishably different activation patterns over time — groups that genuinely specialise will respond to different input distributions.
