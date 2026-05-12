# learn2reflect

A training method in which a model develops self-knowledge of its own error structure, then uses that self-knowledge to direct its own improvement.

## Overview

Standard training is agnostic to a model's own error structure — the loss is computed, gradients are propagated, and weights are updated, but the model itself has no representation of where it fails or why. Reflection Learning extends the model with a *reflector*: a second transformer that forms part of the model itself, attending to its own internal representations layer by layer and producing a calibrated, per-token estimate of its own loss.

This self-model is then used as a training signal, directing corrective pressure back through the same representational pathways the reflector observes.

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

The reflector is a second transformer that runs in parallel with the primary *generator*. It has the same number of layers but a smaller hidden dimension. Each reflector block contains:

- **Causal self-attention** over its own sequence
- **Causal cross-attention** to the corresponding generator block's hidden states

Because the cross-attention is layer-for-layer, the reflector observes not just the generator's final output but how each layer contributes to the representation at every position. Over time it builds a calibrated internal map of the generator's error distribution.

The reflector is trained by comparing its per-token predictions against the generator's actual per-token cross-entropy loss (MSE). The generator's hidden states are detached for this backward pass — the reflector calibrates on what the generator currently is, without pushing the generator in any direction.

### Causal correction — phase 2

After a warmup period, a second gradient signal is introduced. The reflector is run again on the same batch, this time with the generator's hidden states connected to the computation graph. The gradient of the reflector's predictions with respect to the generator's parameters is computed and added to the generator's update.

This gradient flows backward through the reflector's cross-attention connections, then continues up through the generator's own layers. The effect: generator parameters whose representations most strongly drive the reflector's high-loss predictions receive the strongest corrective signal. The causal structure is not imposed by a manual scaling heuristic — it emerges naturally from which layer's representations the reflector learned to rely on.

The two objectives combine in a single forward pass per step:

```
generator loss  =  primary loss
                +  phase2_weight × reflector(hidden_states_connected).mean()

reflector loss  =  mse(reflector(hidden_states_detached), per_token_loss)
```

Each is updated by its own optimizer. The generator's update does not touch reflector weights; the reflector's update does not reach generator weights.

## Relationship to prior work

### GANs

Reflection Learning is structurally similar to GAN training. There are two networks: the generator, which is trying to produce representations the reflector reads as low-loss; and the reflector, which is trying to accurately predict the generator's actual loss. These objectives are in tension — the generator is incentivised to mislead the reflector, and the reflector is incentivised to not be misled.

The key differences from a GAN concern what keeps the system stable.

In a GAN there is no constraint on the generator beyond fooling the discriminator. It can sacrifice any aspect of output quality in pursuit of that goal, which is why GAN training is fragile. Here the generator operates under a hard constraint — the primary language modelling loss — that it cannot sacrifice. Producing representations the reflector reads as low-loss is only useful if those representations also produce good predictions. If the generator manages to mislead the reflector without actually improving, the reflector's MSE loss corrects it on the next step, since the ground truth (actual per-token loss) is always available as a training target.

The reflector also has full visibility into every layer of the generator through layer-for-layer cross-attention. In a GAN the discriminator sees only the generator's output, leaving a single interface to exploit. Here there is no such bottleneck — the reflector observes the entire computational history of every token at every depth.

Whether these constraints are sufficient to prevent the adversarial dynamic from producing unhelpful oscillations rather than genuine improvement remains an open empirical question.

## Limitations

Reflection training does not direct the generator toward any specific target — it pushes only away from representations the reflector associates with high loss. The assumption is that the generator consolidates onto more stable patterns as a result. Whether this produces genuinely better representations or merely representations that are harder for the reflector to classify remains an open empirical question.

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
