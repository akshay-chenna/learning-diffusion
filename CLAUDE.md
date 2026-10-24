# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

This project uses `uv` for dependency management (Python >= 3.12):

```bash
uv sync
source .venv/bin/activate
```

Launch notebooks:
```bash
jupyter notebook
# or
jupyter lab
```

## Structure

Foundation notebooks (math refreshers; CPU, seconds to run):

- **0a. Calculus and Vector Fields**: autograd/VJPs, divergence, curl and conservative fields, Jacobians, instantaneous change of variables + Hutchinson, ODE solvers, conditional vs. marginal velocity preview
- **0b. Probability and Score Functions**: Gaussian algebra, scores, KL/MLE/ELBO, Langevin (noise balance), Tweedie + denoising score matching
- **0c. Latent Spaces, Energy, and Stochastic Processes**: AE/VAE, Boltzmann energies, SDEs, VP-SDE = DDPM (T=1000), reverse SDE, probability-flow ODE, second-order autograd

Main notebooks, each building on prior concepts:

1. **Diffusion basics**: DDPM on 2D spiral data; forward/reverse diffusion, `DenoiserNetwork`, conditional generation with label embeddings
2. **Generating images**: U-Net architecture applied to MNIST; unconditional + conditional image denoising
3. **Classifier-Free Guidance**: joint conditional/unconditional training; CFG at inference via guidance scale
3b. **Bridge - Diffusion, Scores and Flow Matching**: analytic Gaussian-mixture study (no training): path family, ε/x̂/score/v conversions, marginal velocity, DDPM/DDIM/FM samplers, guidance math
4. **Flow Matching**: velocity field learning instead of noise prediction; continuous normalizing flows
5. **Latent Diffusion**: VAE encoder/decoder; diffusion in compressed latent space (Stable Diffusion approach)
6. **Gradient-Based Optimization**: test-time steering of generation toward objectives without retraining
7. **Diffusion EBM**: energy-based modeling with diffusion; scoring and filtering generated samples
8. **Looped Reasoning with Flows**: 6×6 Sudoku with propagation-depth labels; flow vs TRM-style loop vs Looped Flows vs DRM at equal compute; SDE sampling, fixed-point residuals, multi-solution coverage (THEORY.md §12)

Companion docs: `THEORY.md` (math, cross-referenced to notebook sections like "0b §6", "3b §5"), `PHYSICS.md` (physics analogies; marks exact vs. analogical). Keep section references in sync when editing notebooks.

Conventions to keep consistent: DDPM has data at t=0 / noise at t=T; flow matching (notebooks 4-6) has noise `x_0` at t=0 and data `x_1` at t=1.

## Architecture Patterns

All notebooks share these recurring components:

- **`SinusoidalEmbedding`** — timestep encoding (transformer-style positional embeddings), fed into a small MLP to produce a hidden-dim conditioning vector
- **Time conditioning** — additive injection of time embeddings into hidden states at each residual block
- **Beta schedule** — linear schedule `torch.linspace(1e-4, 0.02, T)`; `alpha_bars = cumprod(1 - betas)`
- **Closed-form noising** — `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps` (avoids iterating through all steps)
- **Training objective** — MSE between predicted noise `eps_theta(x_t, t)` and actual noise `eps`
- **Device detection** — notebooks check for MPS (Apple Silicon) then fall back to CPU; on this machine use CUDA if available

`0_implementation.ipynb` and `1_implementation.ipynb` are the user's working implementation copies (the numbered files with spaces are the reference versions); don't edit them unless asked.
