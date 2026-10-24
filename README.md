# Learning Diffusion Models
<p align="center">
  <img src="images/header.png" alt="Diffusion Model Spiral Example" width="500"/>
</p>


A collection of notebooks implemented from scratch to learn step-by-step major concepts of diffusion models. In this project I opted for simple implementations, derived from first principles. Starting from the basics with an unconditional DDPM to reconstruct spiral-shaped data, I go through image generation, classifier-free guidance, flow matching, latent diffusion and gradient-based optimization.
I built this over time as an excuse to learn diffusion from the basics, hopefully it will be helpful to you as well!


## Notebooks

**Suggested order:** 0a → 0b → 0c → 1 → 2 → 3 → 3b → 4 → 5 → 6 → 7 → 8.

### Foundations (0a–0c)
Self-contained math refreshers that run in seconds on a CPU.

- **0a. Calculus and Vector Fields**: derivatives vs. autograd, how autograd computes gradients (VJPs), divergence and curl, conservative fields in *n* dimensions, Jacobians and change of variables, the instantaneous change of variables with Hutchinson trace estimates, ODE solvers, and a first look at why flow-matching trajectories are curved.
- **0b. Probability and Score Functions**: Gaussians and the Gaussian algebra behind every DDPM formula, score functions, KL divergence, maximum likelihood and the ELBO, Langevin dynamics (and why its noise level is fixed), Tweedie's formula and denoising score matching, with a quantitative check of a learned score.
- **0c. Latent Spaces, Energy, and Stochastic Processes**: autoencoders and VAEs, energy functions and Boltzmann distributions, SDEs, the VP-SDE (and why it equals DDPM with T=1000), DDPM sampling as a discretised reverse SDE, the probability-flow ODE, and second-order autograd for energy-based models.

### 1. Diffusion Basics
Introduction to the fundamental concepts of diffusion models using a simple 2D spiral dataset. Learn how neural networks can reverse the forward diffusion process to generate data from noise.

### 2. Generating Images
Apply diffusion models to real image data using the MNIST dataset. Implement a U-Net architecture for image denoising and extend to conditional generation.

### 3. Classifier-Free Guidance
Explore how to improve conditional generation by training models that can operate in both conditional and unconditional modes, allowing for stronger adherence to conditions during inference.

### 3b. Bridge: Diffusion, Scores and Flow Matching
Uses a Gaussian mixture, where everything a network would learn has a closed form, to show that DDPM, score models and flow matching are one framework: one family of noising paths; noise, data, score and velocity predictions that convert into each other; why flow-matching paths are curved and its loss never reaches zero; DDPM vs. DDIM vs. flow-matching samplers; how the DDIM and flow-matching ODEs give the same noise→data map; and classifier-free guidance as score arithmetic. No training needed.

### 4. Flow Matching
Learn about an alternative to traditional diffusion models that uses continuous normalizing flows (CNFs) to define smooth, deterministic trajectories from noise to data. This approach learns velocity fields instead of noise predictions.

### 5. Latent Diffusion
Discover how to work in compressed latent spaces using Variational Autoencoders (VAEs), enabling more efficient generation of high-resolution images. This is the approach used by models like Stable Diffusion.

### 6. Gradient-Based Optimization
Implement gradient-based steering mechanisms that can guide generation toward specific properties or constraints at inference time without retraining the model.

### 7. Diffusion Energy-Based Modelling
Learn how the concept of energy can be applied to diffusion models by tweaking a simple architecture and how to score and filter generated samples by quality.


### 8. Looped Reasoning with Flows
Uses denoising steps to *solve problems* rather than generate data. On 6×6 Sudoku puzzles labelled by how many rounds of deduction they need, it trains four small models at equal compute (a plain categorical flow, a TRM-style looped model, Looped Flows, and a Denoising Recursion Model) and compares how they scale with inference steps and reasoning depth. It also covers stochastic integration, fixed points and spurious attractors (stability as a verifier, as in Flow Reasoning Models), multiple solutions from different noise seeds, and the shared-noise shortcut. Based on *Thinking with Looped Flows* (arXiv:2609.11801), *Flow Reasoning Models* (arXiv:2606.29150) and *One Step Forward and K Steps Back* (arXiv:2604.18839). About 15 minutes on a CPU.

## Companion documents

- [THEORY.md](THEORY.md): the mathematics, derived step by step and cross-referenced to the notebooks, including a "common pitfalls" list (time direction, schedule length, guidance and steering caveats).
- [PHYSICS.md](PHYSICS.md): the same material through statistical physics (Ornstein–Uhlenbeck processes, Fokker–Planck, fluctuation–dissipation, free energy, coarse-graining), marking which correspondences are exact and which are analogies.

## Next steps
It would be interesting to explore more advanced concepts like Diffusion Transformers, optimal-transport couplings and rectified flow, consistency models / distillation, and Multi-Objective Optimization.

## Requirements

- Python >= 3.12
- PyTorch >= 2.9.1
- torchvision >= 0.24.1
- numpy >= 2.3.5
- matplotlib >= 3.10.7
- ipywidgets >= 8.1.8
- scipy >= 1.16.3 (foundation notebooks)

## Setup

This project uses `uv` for dependency management. To set up the environment:

```bash
# Install dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

Alternatively, if you're using pip:

```bash
pip install torch torchvision numpy scipy matplotlib ipywidgets
```

## Dataset

The notebooks use the MNIST dataset for image generation tasks. The dataset will be automatically downloaded when you run the notebooks that require it.

## License

Feel free to use and modify the code. 

## Acknowledgments

- Visualization in the Flow Matching notebook by Alec Helbling ([Diffusion Explorer](https://github.com/helblazer811/Diffusion-Explorer))
