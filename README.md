# learn2reflect

A training method in which a language model develops an explicit self-model of its own error structure and uses that self-model to direct its own improvement.

## Overview

Standard training is agnostic to a model's internal error structure: the loss is computed, gradients are propagated, and weights are updated, but the model itself maintains no representation of where it fails or why. Reflection Learning addresses this by augmenting the model with a *reflector* — a secondary transformer that forms an integral part of the model, attending to its internal representations layer by layer and producing a calibrated, per-token estimate of its own cross-entropy loss.

This self-model is then used as an auxiliary training signal, directing corrective gradient pressure back through the same representational pathways the reflector observes.

## Method

```
                        tokens
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       ┌─────────────┐         ┌─────────────┐
       │  Generator  │         │  Reflector  │
       │             │         │             │
       │  Block 0  ──┼────────►│  Block 0    │  self-attn
       │             │         │  + cross-attn
       │  Block 1  ──┼────────►│  Block 1    │  self-attn
       │             │         │  + cross-attn
       │    ...    ──┼────────►│    ...      │
       │             │         │             │
       │  Block N  ──┼────────►│  Block N    │  self-attn
       │             │         │  + cross-attn
       └──────┬──────┘         └──────┬──────┘
              │                       │
              ▼                       ▼
         next-token              predicted loss
          logits                  per token
```

### Reflector — loss self-modelling

The reflector is a transformer that runs in parallel with the primary *generator*. It has the same number of layers but a smaller hidden dimension. Each reflector block contains:

- **Causal self-attention** over its own sequence
- **Causal cross-attention** to the hidden states of the corresponding generator block

The layer-for-layer cross-attention means the reflector observes not merely the generator's final output but the evolution of each token's representation across all depths. Over training, it is expected to build a calibrated internal map of the generator's error distribution.

The reflector is supervised by comparing its per-token predictions against the generator's actual per-token cross-entropy loss via MSE. The generator's hidden states are detached for this backward pass, so the reflector calibrates against the generator's current behaviour without exerting any gradient pressure on the generator.

### Corrective signal

After a warmup period, a second gradient signal is introduced. The reflector is run again on the same batch, this time with the generator's hidden states connected to the computation graph. The gradient of the reflector's mean predicted loss with respect to the generator's parameters is computed and added to the generator's update.

This gradient flows backward through the reflector's cross-attention connections into the generator's own layers. Generator parameters whose representations most strongly drive the reflector's high-loss predictions receive the strongest corrective signal. The corrective signal specifies no target output — it is, in effect, a standing instruction to the generator: *keep doing what you are doing, but try to be less uncertain about it.*

The two objectives are combined within a single forward pass per training step:

```
generator loss  =  primary loss
                +  phase2_weight × reflector(hidden_states_connected).mean()

reflector loss  =  mse(reflector(hidden_states_detached), per_token_loss)
```

Each component is updated by its own independent optimizer. The generator's update does not affect reflector weights; the reflector's update does not propagate into the generator.

## Relationship to prior work

### Generative adversarial networks

Reflection Learning is structurally analogous to GAN training. The generator is incentivised to produce hidden-state representations that the reflector reads as low-loss; the reflector is incentivised to accurately predict the generator's actual loss. These objectives are in tension, creating an implicit adversarial dynamic.

Three properties distinguish this system from a conventional GAN and are expected to provide stability.

First, the generator operates under a hard constraint — the primary language modelling loss — that cannot be sacrificed. Producing representations the reflector reads as low-loss is beneficial only insofar as those representations also support accurate next-token prediction. A generator that deceives the reflector without genuinely improving will be corrected on subsequent steps, because the reflector's MSE target is always the actual per-token loss.

Second, the corrective signal is weighted at a fraction of the primary loss (`phase2_weight = 0.1`). Where uncertainty is genuinely irreducible — inherent ambiguity in the data rather than a failure of the generator's representations — the corrective gradient opposes the primary loss gradient, and the two partially cancel. The primary loss thereby acts as a natural veto against pressure that cannot be resolved into a genuine improvement.

Third, the reflector has full visibility into every layer of the generator through layer-for-layer cross-attention. A conventional discriminator observes only the generator's output, exposing a single narrow interface that may be exploited. Here there is no such bottleneck; the reflector observes the complete computational history of every token at every depth.

Whether these constraints are sufficient to prevent the adversarial dynamic from producing unhelpful oscillations rather than genuine representational improvement remains an open empirical question.

## Limitations

The corrective signal does not direct the generator toward any specific target; it applies pressure only away from representations the reflector associates with high predicted loss. The implicit assumption is that the generator will consolidate onto more stable and generalisable patterns as a result. Whether this produces genuinely improved representations or merely representations that are harder for the reflector to assign high loss to remains an open empirical question.

## Usage

**Train:**
```bash
python train.py
```

**Generate:**
```bash
python inference.py --prompt "The history of"
```

**Plot training curves:**
```bash
python tools/plot_loss.py
```

Training resumes automatically from the latest checkpoint in `checkpoints/`.

## Requirements

```bash
pip install torch prodigyopt datasets
```
