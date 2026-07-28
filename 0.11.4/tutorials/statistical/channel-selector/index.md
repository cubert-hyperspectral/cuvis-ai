# Learnable Channel Selection

Hyperspectral cameras often deliver 60–100 bands, but only a handful carry the signal you actually care about. The Channel Selector node learns which bands to keep using a two-phase training schedule: statistical warm-up followed by gradient-based refinement.

This tutorial walks through configuring and training a channel selector, then inspecting which wavelengths survived.

**Run the example:**

- [`examples/channel_selector.py`](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook/blob/main/examples/channel_selector.py) — cuvis-ai-cookbook

## What you'll learn

- How [two-phase training](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.4/concepts/training/index.md) combines statistical and gradient stages.
- Configuring `K` (number of channels to keep) and the temperature schedule.
- Reading the final channel weights to recover the selected wavelengths.

## When to reach for channel selection

- You want to ship a downstream model that operates on far fewer channels than the camera delivers (smaller, faster, deployable).
- You suspect only a subset of bands carry your signal and want the model to discover them.
- You need a learnable, end-to-end alternative to hand-picking bands.
