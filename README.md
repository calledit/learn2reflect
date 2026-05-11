# learn2reflect

A training method for neural networks in which the model learns to predict its own loss, then uses that self-knowledge to direct its own improvement.

## Overview

Standard training is agnostic to a model's own error structure — the loss is computed, gradients are propagated, and weights are updated, but the model itself has no representation of where it fails or why. Reflection Learning introduces a lightweight auxiliary output that gives the model exactly that: a calibrated, per-sample estimate of its own loss.

This self-model is then used as a training signal in a second phase, directing corrective pressure toward the parts of the network most likely responsible for the errors.

## Method

### Phase 1 — loss self-modelling

The model is extended with a *reflection head*: a single linear layer that shares the same internal representations as the primary output and produces one scalar per position — a predicted loss value.

During Phase 1, the reflection head is trained by comparing its predictions against the model's actual per-position loss. Over time it builds a calibrated internal map of the model's own error distribution.

### Phase 2 — causal correction

Once the reflection head is calibrated, Phase 2 begins. The reflection head has identified neurons with high predicted loss — but the objective is not to suppress those neurons directly. Their elevated loss is an honest signal; suppressing it would remove information without addressing the underlying problem.

Instead, Phase 2 traces the signal upstream. In the forward pass, input enters at the top of the network and activations flow downward through the layers. A neuron with high predicted loss near the output is a symptom — the cause is most likely higher up, in an earlier layer or representation. Phase 2 sends the gradient of the reflection head's predictions back upward through the network, scaled by `(n_layers - layer_index) / n_layers`.

The scaling serves three purposes:

1. **Gradient diminishing** — gradients can weaken as they travel upward through the network. Scaling compensates so the signal reaches the upper layers where the causes are most likely to reside.
2. **Causal direction** — the location of the root cause is unknown, but we apply the heuristic that causes precede their effects. The scaling expresses this: layers closer to the input receive the strongest signal, on the grounds that they are more likely to be the source of downstream errors.
3. **Phase 1 recovery cost** — Phase 2 runs interleaved with Phase 1. The layers closest to the reflection head are left largely undisturbed so that the signal pathways remain intact and Phase 1 does not spend steps rebuilding calibration that was already in place.

The parameters below the transformer blocks — the output head and the reflection head itself — receive zero gradient from Phase 2. The output head is part of the primary objective; the reflection head is the detector and must not be disturbed.

In practice, Phase 2 does exert some pressure on the uncertain neurons themselves. The assumption is that when those neurons re-emerge through continued Phase 1 training, they will be built on more solid patterns elsewhere.

## Relationship to prior work

### GANs

Reflection Learning shares some structural similarity with GAN training in that a second objective provides a signal that drives improvement in the primary model. There are however important differences.

In a GAN, the generator and discriminator are separate networks with separate parameters in opposition, which is the source of GAN instability — if one side dominates, training collapses. Reflection Learning has no second network. The reflection head is a single linear layer on the same backbone as the primary model, and its objective is to accurately model the primary loss, not to compete with it. This shared structure provides inherent stability: the two objectives cannot diverge from each other the way a GAN's components can, and the primary loss keeps the system grounded throughout training.

It is worth noting that a very high ratio of Phase 2 to Phase 1 steps could theoretically destabilize the system. The stability advantage holds under reasonable training ratios.

## Limitations

It is worth noting that reflection training does not direct the network toward any specific target — it pushes only away from high-loss regions. The assumption is that the network consolidates onto more stable patterns as a result. In theory, however, it is possible for this to produce a training oscillation that does not settle without intervention.

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
