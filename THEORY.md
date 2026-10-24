# The Mathematics of Diffusion and Flow Models

A companion guide to the [learning-diffusion](.) notebooks. It develops the mathematics behind the code, from probability fundamentals through DDPM, classifier-free guidance, the continuous-time (SDE/ODE) view, flow matching, latent diffusion, gradient-based steering, and energy-based models.

Each section names the notebook where its ideas are implemented. Claims you can check numerically are marked with the notebook section that checks them (e.g. *[0b §6]*).

**Suggested order:** 0a → 0b → 0c → 1 → 2 → 3 → **3b** → 4 → 5 → 6 → 7 → **8**, reading the matching section here alongside each notebook. [PHYSICS.md](PHYSICS.md) gives the physical analogies for the same material.

---

## Table of Contents

0. [Notation and Conventions](#0-notation-and-conventions)
1. [Foundations: Probability, Score Functions, and Why They Matter](#1-foundations-probability-score-functions-and-why-they-matter)
2. [DDPM: Denoising Diffusion Probabilistic Models](#2-ddpm-denoising-diffusion-probabilistic-models)
3. [Sinusoidal Time Embeddings](#3-sinusoidal-time-embeddings)
4. [The U-Net Architecture](#4-the-u-net-architecture)
5. [Conditional Generation](#5-conditional-generation)
6. [Classifier-Free Guidance](#6-classifier-free-guidance)
7. [Continuous Time: SDEs, the Probability-Flow ODE, and DDIM](#7-continuous-time-sdes-the-probability-flow-ode-and-ddim)
8. [Flow Matching](#8-flow-matching)
9. [Variational Autoencoders and Latent Diffusion](#9-variational-autoencoders-and-latent-diffusion)
10. [Gradient-Based Steering](#10-gradient-based-steering)
11. [Energy-Based Diffusion Models](#11-energy-based-diffusion-models)
12. [Looped Reasoning with Flows](#12-looped-reasoning-with-flows)
13. [Unified View: How Everything Connects](#13-unified-view-how-everything-connects)
14. [Common Pitfalls](#14-common-pitfalls)
15. [References](#15-references)

---

## 0. Notation and Conventions

| Symbol | Meaning |
|---|---|
| $x_0$ (DDPM) / $x_1$ (flow matching) | clean data. **The two families index time in opposite directions**, see below |
| $\epsilon \sim \mathcal{N}(0, I)$ | standard Gaussian noise |
| $\beta_t,\ \alpha_t = 1-\beta_t,\ \bar\alpha_t = \prod_{s\le t}\alpha_s$ | DDPM schedule |
| $a_t,\ b_t$ | generic signal and noise coefficients, $x_t = a_t\,x_{\text{data}} + b_t\,\epsilon$ |
| $\text{SNR}_t = a_t^2/b_t^2$ | signal-to-noise ratio |
| $p_t$ | marginal density of $x_t$ (data convolved with noise) |
| $s(x,t) = \nabla_x \log p_t(x)$ | score of the noised marginal |

**Direction of time.**

| | Noise at | Data at | Path |
|---|---|---|---|
| DDPM (notebooks 1–3, 7) | $t = T$ | $t = 0$ | $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ |
| VP-SDE (0c, 7) | $t = 1$ | $t = 0$ | $x_t = e^{-B(t)/2}x_0 + \sqrt{1-e^{-B(t)}}\,\epsilon$ |
| Flow matching (notebooks 4–6) | $t = 0$ (called $x_0$!) | $t = 1$ (called $x_1$) | $x_t = t\,x_1 + (1-t)\,x_0$ |

So $x_0$ means *data* in DDPM and *noise* in flow matching. Most confusion when switching between papers comes from this.

---

## 1. Foundations: Probability, Score Functions, and Why They Matter

### 1.1 The generative modeling problem

We have samples from an unknown distribution $p_{\text{data}}(x)$, but no formula for it. The goal is a procedure that produces *new* samples from (approximately) the same distribution.

### 1.2 Energy, the Boltzmann form, and the partition function

Any positive density can be written via an **energy function** $E(x)$:

$$p(x) = \frac{1}{Z} \exp\!\big(-E(x)\big), \qquad Z = \int \exp(-E(x))\, dx$$

Low energy means high probability. $Z$ (the **partition function**) is almost always intractable in high dimension. Two details matter later:

- $E$ is defined only **up to an additive constant**, since $E$ and $E + c$ give the same $p$. Only energy *differences* are meaningful.
- For $p$ to exist, $Z$ must be **finite**. Being bounded below is not enough ($E \equiv 0$ gives $Z = \infty$); $E$ must grow fast enough as $\|x\| \to \infty$.

### 1.3 The score function

$$s(x) = \nabla_x \log p(x) = -\nabla_x E(x)$$

1. **$Z$ drops out**, because $\nabla_x \log Z = 0$.
2. **The score is a vector field** pointing toward higher probability.
3. **It is conservative** (a gradient), so its Jacobian is symmetric *[0a §5]*.
4. **The score is enough to sample.** Langevin dynamics, $x \leftarrow x + \tfrac{\eta}{2}s(x) + \sqrt{\eta}\,z$, has $p$ as its stationary distribution for small $\eta$. The ratio between drift and noise is fixed: scale the noise wrong and you sample the wrong distribution *[0b §4.4]*.

### 1.4 Gaussian algebra

Three facts produce every closed form in DDPM *[0b §1.4]*:

- **Sums:** independent Gaussians add their *variances*: $a x + b \epsilon \sim \mathcal{N}(a\mu, a^2\sigma^2 + b^2)$.
- **Reparameterization:** $z \sim \mathcal{N}(\mu,\sigma^2) \iff z = \mu + \sigma\epsilon$.
- **Conditioning:** for jointly Gaussian $(u,w)$, $\;\mathbb{E}[u\mid w] = \mathbb{E}u + \frac{\text{Cov}(u,w)}{\text{Var}(w)}(w - \mathbb{E}w)$ and $\text{Var}(u\mid w) = \text{Var}(u) - \frac{\text{Cov}(u,w)^2}{\text{Var}(w)}$.

### 1.5 Regression learns conditional means

If you minimise $\mathbb{E}\|f(x) - y\|^2$ over all functions $f$, the minimiser is $f^*(x) = \mathbb{E}[y\mid x]$, because

$$\mathbb{E}\|f(x) - y\|^2 = \mathbb{E}\|f(x) - \mathbb{E}[y\mid x]\|^2 + \underbrace{\mathbb{E}\|\mathbb{E}[y\mid x]-y\|^2}_{\text{independent of } f}$$

Every network in this repository is trained by MSE regression on a *noisy* target (noise, velocity, score). Two consequences:

- The network learns the **conditional mean** of the target, not the target itself.
- The **training loss has a floor** (the second term) and never reaches zero *[0b §6, 3b §5.1]*.

### 1.6 Tweedie's formula and denoising score matching

Let $x_t = a\,x_0 + b\,\epsilon$. Differentiating $p_t(x) = \int p_{\text{data}}(x_0)\,\mathcal{N}(x; a x_0, b^2 I)\,dx_0$ under the integral gives **Tweedie's formula**:

$$\boxed{\nabla_x \log p_t(x) = \frac{a\,\mathbb{E}[x_0 \mid x_t = x] - x}{b^2} = -\frac{\mathbb{E}[\epsilon\mid x_t = x]}{b}}$$

So the **optimal denoiser and the score carry the same information**. Combined with §1.5, regressing onto the per-sample target $-\epsilon/b$ (the score of $q(x_t\mid x_0)$) recovers the marginal score:

$$\mathbb{E}\Big\|s_\theta(x_t) + \frac{\epsilon}{b}\Big\|^2 = \mathbb{E}\big\|s_\theta(x_t) - \nabla\log p_t(x_t)\big\|^2 + \text{const}$$

This is **denoising score matching** (Vincent, 2011). We never need the intractable marginal score, only the noise we added. The flow-matching loss (§8) uses the same trick.

### 1.7 Why not learn the density directly?

- **Normalization:** $p_\theta$ needs $Z_\theta$.
- **Dimension:** a $32\times32$ image lives in $\mathbb{R}^{1024}$.
- **Score suffices:** sampling only needs $\nabla_x\log p$, at many noise levels (§2.5, §7).

---

## 2. DDPM: Denoising Diffusion Probabilistic Models

*Reference: Ho et al. (2020), building on Sohl-Dickstein et al. (2015). Notebooks 1–2.*

### 2.1 The forward process (data → noise)

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\Big(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t\, I\Big)$$

with the linear schedule $\beta_t = \text{linspace}(10^{-4}, 0.02, T)$.

**Why $\sqrt{1-\beta_t}$?** $\text{Var}(x_t) = (1-\beta_t)\text{Var}(x_{t-1}) + \beta_t$. This map has the fixed point $\text{Var} = 1$, and repeated application drives any initial variance toward 1. So the process is **variance preserving** for unit-variance data, and converges to $\mathcal{N}(0,I)$ for any data. Without the factor, the variance would grow without bound.

### 2.2 The closed-form shortcut

With $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$:

$$q(x_t \mid x_0) = \mathcal{N}\!\Big(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t)\, I\Big) \quad\Longrightarrow\quad \boxed{x_t = \sqrt{\bar{\alpha}_t}\; x_0 + \sqrt{1-\bar{\alpha}_t}\; \epsilon}$$

**Derivation.** Two steps give $x_2 = \sqrt{\alpha_1\alpha_2}\,x_0 + \sqrt{\alpha_2(1-\alpha_1)}\,\epsilon_1 + \sqrt{1-\alpha_2}\,\epsilon_2$. The two noise terms are independent with total variance $\alpha_2(1-\alpha_1) + 1-\alpha_2 = 1-\alpha_1\alpha_2$. Induction on $t$ gives the general formula *[0b §1.4]*.

**Is $x_T$ really pure noise?** Only if $\bar\alpha_T \approx 0$. The `linspace(1e-4, 0.02, T)` schedule was designed for $T = 1000$ ($\bar\alpha_T \approx 4\times10^{-5}$). With the $T = 200$ and $T = 300$ used in notebooks 1 and 7, $\sqrt{\bar\alpha_T} \approx 0.36$ and $0.22$: a noticeable fraction of the signal remains, but sampling starts from $\mathcal{N}(0, I)$ anyway *[3b §2.1, 0c §4.5]*. It works for the spiral, but rescale the betas when you change $T$.

### 2.3 The reverse process (noise → data)

The true reverse step $q(x_{t-1}\mid x_t)$ depends on the unknown data distribution. Conditioned additionally on $x_0$ it becomes a Gaussian, by §1.4 applied to the forward chain *[0b §1.4]*:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\Big(x_{t-1};\; \tilde{\mu}_t(x_t, x_0),\; \tilde{\beta}_t\, I\Big)$$

$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1-\bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\, x_t, \qquad \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t$$

At sampling time $x_0$ is unknown, so we substitute $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$ and replace $\epsilon$ by the network's prediction $\epsilon_\theta(x_t, t)$:

$$\boxed{x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t) \right) + \sigma_t\, z}$$

with $z\sim\mathcal{N}(0,I)$ for $t>1$ and $z=0$ at the last step. Ho et al. found $\sigma_t^2 = \beta_t$ and $\sigma_t^2 = \tilde\beta_t$ give similar results. They are the optimal choices for two extreme cases (data concentrated at a point vs. Gaussian data).

Since $\epsilon_\theta \approx \mathbb{E}[\epsilon \mid x_t]$ (§1.5), this plugs in the posterior mean of $x_0$: $\hat x_0 = \mathbb{E}[x_0\mid x_t]$.

### 2.4 The training objective

The principled objective is the **ELBO** on $\log p_\theta(x_0)$ *[0b §3.2]*, with the forward process playing the role of the encoder. It decomposes into a sum over $t$ of $D_{KL}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)$. Between two Gaussians with the same variance, this KL is a squared distance between means, which with the parameterization above is a weighted MSE on $\epsilon$. Ho et al. dropped the weights:

$$\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}\Big[\| \epsilon - \epsilon_\theta(x_t, t) \|^2\Big]}$$

Dropping the weights changes how much each noise level contributes. The loss is no longer a likelihood bound, but it gives better samples. Loss weighting is the main knob that distinguishes many later variants (§13.2).

**Training:** sample $x_0$, $t \sim \text{Uniform}\{1..T\}$, $\epsilon$; form $x_t$; regress $\epsilon_\theta(x_t,t)$ onto $\epsilon$.

### 2.5 The connection to score matching

By Tweedie (§1.6) with $a = \sqrt{\bar\alpha_t}$, $b = \sqrt{1-\bar\alpha_t}$, the optimal network satisfies

$$\boxed{s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}}$$

This holds for the score of the **marginal** $p_t$. That is the non-trivial part: the per-sample identity $\nabla_{x_t}\log q(x_t\mid x_0) = -\epsilon/\sqrt{1-\bar\alpha_t}$ is just the gradient of a Gaussian, and denoising score matching (§1.6) is what connects the two. Because of this, the noise-prediction network is a score estimator, and DDPM fits into the SDE framework of §7.

---

## 3. Sinusoidal Time Embeddings

The network must know the noise level. Instead of feeding the raw integer $t$, we map it to a vector of sinusoids:

$$\text{emb}_{2i}(t) = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad \text{emb}_{2i+1}(t) = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

1. **Multiple frequency scales.** Frequencies $\omega_i = 10000^{-2i/d}$ range from fast (resolving neighbouring $t$) to slow (separating coarse ranges of $t$).
2. **Smoothness.** Nearby $t$ get similar embeddings, which helps generalisation across noise levels.
3. **No parameters.** A learnable MLP (`time_mlp`) follows and projects to the hidden dimension.
4. **From Transformers** (Vaswani et al., 2017), where it encoded sequence positions.

**Continuous time caveat.** The frequencies are designed for $t$ in the hundreds. In flow matching (notebook 4) $t \in [0,1]$, so most sinusoids barely move over the whole range. Implementations commonly multiply $t$ by ~1000 before embedding.

The embedding is **added** to hidden features in every block (additive conditioning), so each layer can adapt to the noise level.

---

## 4. The U-Net Architecture

*Notebooks 2–6.* For images, the network must output a tensor of the same shape as its input while using context at several scales.

- **Down path:** convolution blocks + downsampling, $32\to16\to8\to4$ pixels, with increasing channels. Captures increasingly global structure.
- **Bottleneck:** the coarsest representation ("this is a 3").
- **Up path:** upsampling back to $32\times32$.
- **Skip connections:** at each resolution, encoder features are *concatenated* to the decoder input. This preserves fine detail (edges, stroke positions) that the bottleneck discards. Without skips, outputs are blurry.
- **Time conditioning:** the time embedding is broadcast to $(B, C, 1, 1)$ and added to every block's features:

```python
time_emb = self.time_mlp(t)                 # (B, hidden_dim)
h = h + time_emb[(...,) + (None,) * 2]       # broadcast over H, W
```

- **Residual connections** ($h_{\text{out}} = h + \text{Block}(h)$, used in the spiral MLP) ease optimisation: each block learns a correction to the identity.

Modern large models replace the U-Net with **Diffusion Transformers (DiT)**, but the conditioning ideas (time embedding, label/text embedding) carry over.

---

## 5. Conditional Generation

*Notebooks 2–3.* To control *what* is generated, the network also receives a label $y$:

$$h \leftarrow h + \text{MLP}(t) + e(y), \qquad e(y) = \text{Embedding}(y) \in \mathbb{R}^d$$

The network becomes $\epsilon_\theta(x_t,t,y)$. By §2.5 it estimates the **conditional score** $\nabla\log p_t(x\mid y)$, and the same sampler then draws from $p(x\mid y)$. Nothing else changes.

In practice, plain conditioning can follow the label only weakly. A big network can model $p(x\mid y)$ partly by modeling $p(x)$, and conditioning signals are easy to underuse. Guidance (§6) addresses this.

---

## 6. Classifier-Free Guidance

*Reference: Ho & Salimans (2022), arXiv:2207.12598. Notebook 3; exact demonstration in 3b §8.*

### 6.1 Guidance as score arithmetic

Bayes' rule $p(x\mid y) \propto p(x)\,p(y\mid x)$ becomes **addition** after $\nabla_x\log$:

$$\nabla_x \log p_t(x \mid y) = \nabla_x \log p_t(x) + \nabla_x \log p_t(y \mid x)$$

**Classifier guidance** (Dhariwal & Nichol, 2021) trains a classifier on noisy images and samples with $\nabla\log p_t(x) + w\,\nabla\log p_t(y\mid x)$, where $w>1$ amplifies the class signal.

**Classifier-free guidance** gets the classifier gradient for free from one network trained in two modes:

$$\nabla_x \log p_t(y\mid x) = \nabla_x\log p_t(x\mid y) - \nabla_x\log p_t(x)$$

### 6.2 Training with label dropout

With probability $p_{\text{drop}}$ (0.1–0.2), replace the label with a **null token** $\varnothing$ (index 10 for MNIST). The same network then learns both $\epsilon_\theta(x_t,t,y)$ and $\epsilon_\theta(x_t,t,\varnothing) \approx$ unconditional.

### 6.3 The guidance formula

$$\boxed{\epsilon_{w} = \epsilon_\theta(x_t, t, \varnothing) + w \cdot \Big(\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \varnothing)\Big)}$$

- $w = 0$: unconditional; $w = 1$: plain conditional; $w > 1$: amplified.
- Because $\epsilon$ is a scaled score (§2.5), this is exactly score-space guidance. The **same formula** works on velocities in flow matching (§8.6) and on $\hat x_0$ predictions, since all are affine in the score at fixed $(x_t, t)$.
- Both predictions come from one batched forward pass (`torch.cat([x, x])` with real and null labels).

### 6.4 What CFG does and does not sample

It is tempting to say CFG samples $\tilde p_w(x) \propto p(x)\,p(y\mid x)^w$. **It does not.** Guidance tilts every *noisy* marginal $p_t$, and tilting doesn't commute with adding noise. At high noise the noisy classifier $p_t(y\mid x)$ is soft and its gradient acts over large regions, so CFG pushes samples away from other classes and distorts their shape more than the clean-data tilt would. The resulting distribution is not defined by any simple formula *[3b §8.1]*.

That is the real source of the **quality–diversity trade-off**. Higher $w$ gives more prototypical, higher-contrast samples (often preferred perceptually), less diversity, and, at very large $w$, artifacts and off-manifold samples. On images $w \in [2, 7]$ is typical.

---

## 7. Continuous Time: SDEs, the Probability-Flow ODE, and DDIM

*Notebooks 0c §4, 3b, 7. References: Song et al. (2021) "Score-Based Generative Modeling through SDEs"; Song, Meng & Ermon (2021) "DDIM".*

### 7.1 The VP-SDE is DDPM in continuous time

$$dx = -\tfrac12\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dW, \qquad \beta(t) = \beta_{\min} + t(\beta_{\max}-\beta_{\min})$$

This is an Ornstein–Uhlenbeck process with stationary distribution $\mathcal{N}(0, I)$, and it has the exact solution

$$x_t = e^{-B(t)/2}\,x_0 + \sqrt{1-e^{-B(t)}}\;\epsilon, \qquad B(t) = \int_0^t\beta(s)\,ds = \beta_{\min}t + \tfrac{t^2}{2}(\beta_{\max}-\beta_{\min})$$

Taking DDPM with $N$ steps, $\beta(t_i) \approx N\beta_i$, so $\prod(1-\beta_i)\approx e^{-\sum\beta_i}\to e^{-B(t)}$. With $N = 1000$ and `linspace(1e-4, 0.02)`, this gives $\beta_{\min} = 0.1$ and $\beta_{\max} = 20$, exactly the constants used in notebook 7 *[0c §4.5]*.

(Notebook 7 defines the VP-SDE helpers, but trains and samples with the discrete $T = 300$ schedule, which corresponds to $\beta_{\max}\approx 6$.)

### 7.2 Reverse-time SDE

For a forward SDE $dx = f(x,t)\,dt + g(t)\,dW$, Anderson (1982) showed that the time-reversed process is also a diffusion:

$$dx = \big[f(x,t) - g(t)^2\,\nabla_x\log p_t(x)\big]\,dt + g(t)\,d\bar W \qquad (\text{time running backward})$$

The only unknown is the **score**, which is what the network learns. **DDPM's sampling loop is an Euler–Maruyama discretisation of this SDE**: expand $1/\sqrt{\alpha_t}\approx 1+\beta_t/2$ in §2.3 and the update becomes $x - [f - g^2 s]\Delta t + g\sqrt{\Delta t}\,z$ *[0c §4.7]*.

It is *not* Langevin MCMC. Langevin targets a fixed distribution and needs many steps per noise level; the reverse SDE moves through the noise levels, one step each. (Song et al.'s "predictor–corrector" samplers add Langevin *corrector* steps on top of the reverse-SDE *predictor*.)

### 7.3 The probability-flow ODE

Song et al. (2021) showed that the deterministic ODE

$$\frac{dx}{dt} = f(x,t) - \tfrac12\,g(t)^2\,\nabla_x\log p_t(x)$$

has **the same marginals $p_t$** as the SDE *[0c §4.8]*. Its solutions are smooth, deterministic and invertible, which gives:

- **Exact likelihoods**, via the instantaneous change of variables $\frac{d}{dt}\log p_t(x(t)) = -\nabla\cdot v$ and Hutchinson trace estimates *[0a §6]*.
- **Few-step sampling** with good ODE solvers.
- A **deterministic encoding** of each data point to a noise vector and back.

Why it works: the Fokker–Planck equation for the SDE, $\partial_t p = -\nabla\cdot(fp) + \tfrac{g^2}{2}\Delta p$, can be rewritten as a pure transport (continuity) equation $\partial_t p = -\nabla\cdot(\tilde v\,p)$ with $\tilde v = f - \tfrac{g^2}{2}\nabla\log p$, using $\Delta p = \nabla\cdot(p\,\nabla \log p)$.

### 7.4 DDIM: the probability-flow ODE for a trained DDPM

DDIM uses the *same* trained $\epsilon_\theta$ and a deterministic update (with $t'$ any earlier timestep, so steps can be skipped):

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}, \qquad x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t'}}\;\epsilon_\theta$$

Predict the clean image, then re-noise it *deterministically* with the predicted noise. This is a first-order discretisation of the probability-flow ODE (an Euler step in the variables $x/\sqrt{\bar\alpha}$ and $\sqrt{(1-\bar\alpha)/\bar\alpha}$). A parameter $\eta\in[0,1]$ re-injects noise, and $\eta=1$ recovers ancestral sampling. With a perfect model, deterministic samplers degrade far more gracefully as the number of steps drops *[3b §6]*.

### 7.5 One path family, four predictions

Every method uses $x_t = a_t\,x + b_t\,\epsilon$. The optimal predictions are conditional means, linked by fixed linear maps *[3b §4]*:

$$\hat\epsilon = \frac{x_t - a_t\hat x}{b_t}, \qquad s = -\frac{\hat\epsilon}{b_t} = \frac{a_t\hat x - x_t}{b_t^2}, \qquad v = \dot a_t\,\hat x + \dot b_t\,\hat\epsilon$$

where $\hat x = \mathbb{E}[x\mid x_t]$, $\hat\epsilon = \mathbb{E}[\epsilon\mid x_t]$, $v = \mathbb{E}[\dot x_t\mid x_t]$. "Predict noise", "predict data", "predict score" and "predict velocity" are different **output parameterizations and loss weightings** of the same information. They matter for optimisation, but not for what the ideal model represents.

---

## 8. Flow Matching

*References: Lipman et al. (2023); Liu et al. (2023, rectified flow); Albergo & Vanden-Eijnden (2023, stochastic interpolants). Notebooks 4–6; 3b.*

### 8.1 Continuous normalizing flows

A time-dependent **velocity field** $v_t(x)$ defines an ODE $\frac{dx}{dt} = v_t(x)$. If the density of particles obeys the **continuity equation**

$$\partial_t p_t + \nabla\cdot(p_t\,v_t) = 0,$$

then integrating from $p_0 = \mathcal{N}(0, I)$ yields $p_1$. Learning $v$ so that $p_1 = p_{\text{data}}$ gives a **continuous normalizing flow**. Older CNFs trained $v$ by maximum likelihood, which needs an ODE solve per training step. Flow matching avoids simulation entirely.

### 8.2 Conditional paths and the conditional velocity

Pick a noise sample $x_0\sim\mathcal{N}(0, I)$ and a data sample $x_1$, independently, and join them by the straight line (notebook 4's convention)

$$x_t = (1-t)\,x_0 + t\,x_1 \qquad\Longrightarrow\qquad \frac{dx_t}{dt} = x_1 - x_0 .$$

This **conditional** velocity is constant along each line. But many lines pass through the same point $x$ at time $t$, heading in different directions. The lines **cross**, so no single velocity field generates them.

### 8.3 The marginal velocity: what the network actually learns

The network sees only $(x_t, t)$. Trained with MSE on targets $x_1 - x_0$, it learns their conditional mean (§1.5):

$$\boxed{v^*(x,t) = \mathbb{E}\,[\,x_1 - x_0 \mid x_t = x\,]}$$

**Flow-matching theorem.** $v^*$ satisfies the continuity equation for the marginal path $p_t$ (the distribution of $x_t$), so integrating $v^*$ from noise produces exactly $p_{\text{data}}$. Properties of $v^*$ *[0a §8.5, 3b §5]*:

- Its trajectories **do not cross** and are generally **curved**. They straighten only where the posterior over "which $x_1$ am I heading to" is already sharp.
- At $t\approx 0$, $v^*$ points toward $\mathbb{E}[x_1]$, the data mean. One or two Euler steps therefore produce mean-like, blurry samples.
- The noise→data pairing realised by the ODE is **not** the random pairing used in training.

So the statement "the velocity is constant along the path" is true of the training targets, not of the learned field.

### 8.4 The training objective and its floor

$$\boxed{\mathcal{L}_{\text{CFM}} = \mathbb{E}_{x_0, x_1, t}\Big[\| v_\theta(x_t, t) - (x_1 - x_0) \|^2\Big]}$$

As in §1.5,

$$\mathcal{L}_{\text{CFM}}(v_\theta) = \underbrace{\mathbb{E}\|v_\theta - v^*\|^2}_{\mathcal{L}_{\text{FM}}} + \underbrace{\mathbb{E}\|(x_1-x_0) - v^*\|^2}_{\text{const}}$$

so the cheap conditional loss has the **same gradients** as the intractable marginal one. It also plateaus well above zero, and that is expected *[3b §5.1]*.

Compared with DDPM: there is no discrete schedule, time is continuous in $[0,1]$, the target is $x_1 - x_0$ rather than $\epsilon$, and the path is not variance-preserving ($a_t^2 + b_t^2 = t^2 + (1-t)^2 < 1$ in the middle).

### 8.5 Relation to scores and to DDPM

With $a_t = t$, $b_t = 1-t$ in §7.5:

$$v^*(x,t) = \hat x - \hat\epsilon = \frac{x - \hat\epsilon}{t} = \frac{x + (1-t)\,s(x,t)}{t}$$

where $\hat\epsilon = \mathbb{E}[x_0\mid x_t = x]$ (the noise endpoint, in FM notation) and $s = \nabla\log p_t$. So a velocity network is also a score estimator, and the flow-matching ODE is the **probability-flow ODE of the linear Gaussian path**.

Two Gaussian paths that visit the same SNR values are the same path up to rescaling $x_t$ and reparameterizing time. Hence the DDIM ODE and the flow-matching ODE realise **the same deterministic noise→data map**, just on different clocks *[3b §7, where they agree to ~$10^{-3}$]*. The practical differences between DDPM-style and FM-style training are the loss weighting across noise levels, the regression target, and the time discretisation used for sampling.

### 8.6 Sampling

Euler integration from $t=0$ to $1$: $x \leftarrow x + v_\theta(x,t)\,\Delta t$. Higher-order solvers (Heun, RK4) trade more network evaluations per step for fewer steps. CFG applies directly to velocities: $v_w = v_\varnothing + w(v_y - v_\varnothing)$.

**How many steps?** The linear path is "straighter" than the VP path in a loose sense, and in practice flow-matching models often sample well with 10–50 steps. But $v^*$ is curved for multimodal data, so very few Euler steps produce averaged samples *[3b §6]*. Notebook 4's 3-step MNIST digits are recognisable, but this does not mean flow matching needs only 3 steps in general. Techniques that actually straighten paths:

- **Rectified flow / reflow** (Liu et al., 2023): retrain on the (noise, sample) pairs produced by the current ODE, which are no longer crossing.
- **Minibatch optimal-transport couplings** (Tong et al., 2023; Pooladian et al., 2023): pair noise and data within a batch by solving a small assignment problem, so conditional paths cross less.
- **Distillation** (consistency models, progressive distillation) for 1–4 step sampling.

### 8.7 DDPM vs. flow matching

| Aspect | DDPM (notebooks 1–3) | Flow matching (notebooks 4–6) |
|---|---|---|
| Path | VP: $\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | Linear: $t x_1 + (1-t)x_0$ |
| Network output | Noise $\hat\epsilon$ | Velocity $v$ |
| Time | Discrete, noise at $t=T$ | Continuous, noise at $t=0$ |
| Usual sampler | Ancestral (stochastic) SDE steps | Euler/ODE (deterministic) |
| Deterministic alternative | DDIM (probability-flow ODE) | — (stochastic versions exist) |
| Ideal model encodes | the score of $p_t$ | the score of $p_t$ (via §8.5) |
| Typical steps | 1000 ancestral; 20–100 with DDIM | 10–100 Euler |

---

## 9. Variational Autoencoders and Latent Diffusion

*References: Kingma & Welling (2014); Rombach et al. (2022). Notebook 5; 0c Parts 1–2.*

### 9.1 The scalability problem

A $512\times512$ RGB image has 786,432 dimensions, and every sampling step processes all of them.

### 9.2 VAEs

- **Encoder** $q_\phi(z\mid x) = \mathcal{N}(\mu_\phi(x), \text{diag}\,\sigma^2_\phi(x))$.
- **Decoder** $p_\psi(x\mid z)$.
- **Reparameterization:** $z = \mu + \sigma\odot\epsilon$, so gradients flow through $\mu,\sigma$.

### 9.3 The ELBO and where the MSE comes from

$$\log p(x) \ge \mathbb{E}_{q_\phi(z|x)}[\log p_\psi(x\mid z)] - D_{KL}\big(q_\phi(z\mid x)\,\|\,\mathcal{N}(0,I)\big)$$

The gap is $D_{KL}(q_\phi(z\mid x)\,\|\,p(z\mid x))$ *[0b §3.2]*. With a Gaussian decoder $p_\psi(x\mid z) = \mathcal{N}(\text{dec}(z), \sigma_{\text{dec}}^2 I)$ the negative ELBO is, up to constants,

$$\mathcal{L}_{\text{VAE}} = \|x - \hat x\|^2 + \beta\,D_{KL}, \qquad \beta = 2\sigma^2_{\text{dec}}$$

So "$\beta$" encodes an assumed pixel noise level. Whether the reconstruction term is a sum or a mean over pixels also changes the effective $\beta$.

**KL between diagonal Gaussians.** For each latent dimension, $D_{KL}(\mathcal{N}(\mu_j,\sigma_j^2)\,\|\,\mathcal{N}(0,1)) = \tfrac12(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2)$. Summing over dimensions:

$$D_{KL} = -\frac{1}{2}\sum_j \Big(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\Big)$$

**Why the KL term matters.** Without it, the encoder can place each image at an isolated point, leaving "holes" in latent space where the decoder produces garbage *[0c §1.4]*. The KL term keeps the latent distribution close to $\mathcal{N}(0, I)$, so latents are smooth, connected and samplable.

### 9.4 The latent diffusion pipeline

1. Train the VAE; freeze it.
2. Encode the dataset: $z = \mu_\phi(x)$ (using the mean is common for stability).
3. Train a diffusion or flow model on $z$. In notebook 5, $z$ is $4\times4\times4$ = **64 numbers vs. 1024 pixels, a 16× reduction** in dimension. (The *spatial* resolution shrinks 64×, $32^2/4^2$, but the latent has 4 channels.)
4. Generate: noise → latent sampler → decoder (run once).

Notebook 5 uses a small ResNet (`LatentDiffuser`) instead of a U-Net because the latents are already tiny. Two practical notes: the latent sampler only sees what the VAE preserves, so VAE blur is a hard ceiling on sharpness. And production systems rescale latents to roughly unit variance before diffusion (Stable Diffusion's 0.18215 factor), because the noise schedule assumes data of unit scale.

---

## 10. Gradient-Based Steering

*Notebook 6; 0c §3.5; 3b §8.*

### 10.1 The setup

You have a trained flow model and a differentiable "energy" $E(x)$ measuring a property you want *less* of. Examples from notebook 6:

- **Thickness/intensity:** $E(x) = -\text{mean}(x)$, so lowering $E$ brightens the image.
- **Symmetry:** $E(x) = \text{mean}\big((x - \text{flip}(x))^2\big)$.

### 10.2 The update

$$x_{t+\Delta t} = x_t + v_\theta(x_t, t)\,\Delta t - \gamma\,\nabla_x E(x_t)$$

The minus sign moves *down* the energy. Gradients come from autograd inside `torch.enable_grad()`. Only the gradient with respect to the *input* is computed, and model weights are untouched.

### 10.3 What this approximately samples

The intended target is the **tilted distribution**

$$\tilde p(x) \propto p_{\text{data}}(x)\,e^{-\lambda E(x)}, \qquad \nabla\log\tilde p = \nabla\log p_{\text{data}} - \lambda\nabla E.$$

For this to be exact, the sampler would need the score of the *noised* tilted distribution at every $t$, which is not $\nabla\log p_t - \lambda\nabla E$ (the same non-commutation as CFG, §6.4). Three practical consequences:

- **Evaluate $E$ on the denoised estimate.** $E$ was designed for clean images, but $x_t$ is noisy. Evaluating $E(\hat x(x_t))$, where for the linear path $\hat x = x_t + (1-t)\,v_\theta$, and differentiating through it, is usually better. This is the idea behind diffusion posterior sampling (Chung et al., 2023) and universal guidance.
- **Scale with $\Delta t$.** In notebook 6 the steering term is not multiplied by $\Delta t$, so the *total* push grows with the number of steps: $\gamma = 100$ at 20 steps and at 80 steps are different strengths. Writing the update as $(v_\theta - \gamma'\nabla E)\,\Delta t$ makes $\gamma'$ step-count independent.
- **Strong $\gamma$ leaves the data manifold.** The energy gradient knows nothing about what images look like.

### 10.4 Limitations

$E$ must be differentiable; each step costs an extra backward pass; and some desirable properties are hard to express as smooth energies.

---

## 11. Energy-Based Diffusion Models

*Inspired by ProteinEBM. Notebook 7; 0c Part 6.*

### 11.1 Why an energy

A standard diffusion model outputs a vector ($\epsilon$ or $v$) and cannot assign a *number* to an input. An EBM outputs a scalar $E_\theta(x,t)$ with

$$s_\theta(x,t) = -\nabla_x E_\theta(x,t),$$

so the same network generates (via its score) and scores inputs (via its value).

### 11.2 The "holy trinity"

$$p_t(x)\propto e^{-E(x,t)}, \qquad \log p_t = -E - \log Z_t, \qquad s = -\nabla_x E$$

### 11.3 Parameterization: $E = \|r_\theta(x,t)\|^2$

Notebook 7 uses a vector network $r_\theta$ and $E = \|r_\theta\|^2$:

- $E \ge 0$. This alone does **not** make $e^{-E}$ normalizable (§1.2); in practice the learned $E$ grows like $\|x\|^2/2$ far from data, as the true $-\log p_t$ does.
- $\nabla_x E = 2J_r^\top r$, one VJP *[0a §2.1]*.
- **Curvature:** the Hessian is $2J_r^\top J_r + 2\sum_i r_i\nabla^2 r_i$. Near the minima ($r\approx0$) the first, Gauss–Newton, term dominates and is positive semi-definite.

### 11.4 Training: denoising score matching

$$\mathcal{L}_{\text{EBM}} = \mathbb{E}\Big[(1-\bar\alpha_t)\,\Big\|\,s_\theta(x_t,t) + \frac{\epsilon}{\sqrt{1-\bar\alpha_t}}\Big\|^2\Big]$$

With $\epsilon_\theta := -\sqrt{1-\bar\alpha_t}\,s_\theta$ this is *exactly* DDPM's $\|\epsilon_\theta - \epsilon\|^2$. The only change from notebook 1 is the constraint that the score be a gradient. The cost is **double backprop**: the loss involves $\nabla_x E_\theta$, so updating $\theta$ differentiates through a gradient (`create_graph=True`). This is about 2× slower per step on small MLPs *[0c §5.5]*.

### 11.5 Sampling

Notebook 7 plugs $s_\theta$ into the DDPM update. Its function is named `langevin_step`, but it is an **ancestral reverse-SDE step** (§7.2), not Langevin MCMC:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\big(x_t + \beta_t\,s_\theta(x_t,t)\big) + \sigma_t z$$

### 11.6 Time-dependent landscapes

| Noise level | Landscape | Use |
|---|---|---|
| High $t$ | Smooth, single broad basin | global structure, early sampling |
| Medium $t$ | Valleys follow the data manifold | **scoring/ranking** works best here |
| Low $t$ | Sharp, narrow valleys on the data | fine corrections; scores are noisy off-manifold |

### 11.7 Scoring, ranking, filtering

Given candidates, compute $E_\theta(x, t_{\text{eval}})$ at a **fixed** $t_{\text{eval}}$ and keep the lowest energies (overgenerate → score → filter). Energies are only meaningful up to a constant at each $t$, so never compare energies across different $t$. The method helps when the energy's ranking correlates with quality, which notebook 7 checks with decoys of increasing distortion. It cannot fix mode coverage: filtering removes bad samples but cannot add missing ones.

### 11.8 Conservative fields

$s_\theta = -\nabla E_\theta$ has a symmetric Jacobian by construction *[0a §5]*. Note that the **true** score is conservative too, so a well-trained $\epsilon$-network is *approximately* conservative already. The EBM enforces exactly, as an architectural constraint, what the other models only learn approximately. In 2D or 3D, that means zero curl.

---

## 12. Looped Reasoning with Flows

*References: Wang et al. (2025, HRM); Jolicoeur-Martineau (2025, TRM); Cameron et al. (2026, "One Step Forward and K Steps Back", DRM, arXiv:2604.18839); Helbling et al. (2026, "Flow Reasoning Models", FRM, arXiv:2606.29150); Suleymanzade et al. (2026, "Thinking with Looped Flows", arXiv:2609.11801). Notebook 8.*

Everything so far *generates* data. The same machinery can also *reason*: given a problem $c$ (a Sudoku grid, a maze, an ARC task) produce its solution $x$, by spending many network evaluations on one answer. That is "thinking with flows".

### 12.1 Looped models and the credit-assignment problem

A **looped** (recurrent-depth) model reuses one network $f$ many times on a hidden state:

$$z_i = f(z_{i-1};\, c), \qquad \hat x_i = g(z_i)$$

More iterations = more serial computation at test time, which is what multi-step problems need. A fixed-depth network can only do a bounded amount of sequential deduction per forward pass; loops remove that bound. HRM and TRM made this work on Sudoku-Extreme, Maze-Hard and ARC-AGI with ~5–30M parameters.

Training is the hard part. Full backpropagation through many iterations is costly and unstable, so in practice each step gets its own loss and the gradient is cut between steps (**truncated BPTT / deep supervision**):

$$\mathcal{L}_{\text{loop}} = \mathbb{E}\Big[\sum_i \text{CE}(\hat x_i, x)\Big], \qquad z_i = f(\text{sg}(z_{i-1}); c)$$

No loss ever tells step $i$ to leave a state that is useful *later*. Every step greedily tries to output the answer right away. The learned recurrence can then fall into **spurious attractors** (it converges to a wrong answer) or **never converge**. Looped Flows measured exactly these two failure modes in TRM on Sudoku-Extreme.

### 12.2 Flows already solve a version of this problem

A flow model is also "a network applied many times", but it is trained **without backpropagating through time** at all: each time $t$ has a *local* denoising loss, and the local losses together define a globally coherent trajectory (§8.4). For categorical solutions (one-hot $x_1 \in \{0,1\}^{L\times|V|}$) the standard recipe is:

$$I_t = (1-t)\,x_0 + t\,x_1,\quad x_0 \sim \mathcal{N}(0,\sigma^2 I), \qquad \mathcal{L}_{\text{flow}} = \mathbb{E}_{t}\big[\text{CE}(\hat D_t(I_t;c),\, x_1)\big]$$

The denoiser $\hat D_t$ outputs a probability vector per position, an estimate of $\mathbb{E}[x_1 \mid I_t, c]$, which is exactly the $\hat x$-prediction of §7.5, and the sampling velocity is

$$b_t(x) = \frac{\hat D_t(x; c) - x}{1-t}.$$

The weakness: a plain flow carries no memory between steps except $x_t$ itself, and is known to struggle with genuinely serial computations. All three papers below add a **recurrent state** to the flow.

### 12.3 Self-conditioning and Flow Reasoning Models (FRM)

**Self-conditioning** (Chen et al., 2022) feeds the model its own previous clean prediction $s$: $\hat D_t(x_t \mid c, s)$. At inference this gives a recurrence over depth $k$, separate from flow time:

$$s^{(k+1)} = \hat D_t\big(x_t \mid c,\, s^{(k)}\big)$$

and a correct solution should be a **fixed point**, $\hat D_t(x_t\mid c, s^*) = s^*$. FRM's two observations:

1. **Exposure bias.** Standard self-conditioning trains on a carry made by *one* pass from the null carry, but at inference the carry comes from the model's own long closed-loop rollout. The model can converge to *confidently wrong* fixed points. **Fixed-Point Forcing** trains on carries produced by actually running the self-conditioned integrator from a random earlier time (detached), while keeping the loss and $x_t$ unchanged.
2. **Self-verification.** With Fixed-Point Forcing, the step-to-step residual $r_k = D_{\text{SKL}}(p_k, p_{k-1})$ becomes informative: stable candidates are usually correct (FRM reports AUROC 1.00 on Sudoku-Extreme, and 99.2% solve rate with 7 forward passes). Convergence is still *not* a proof of correctness; it can certify a wrong attractor.

### 12.4 Denoising Recursion Models (DRM): one step forward, k steps back

DRM (on top of TRM) contrasts two ways to train a recursion:

- **Forward-training** (diffusion-style): corrupt the target to level $\tau$ and train *one* step to undo it. Gives a curriculum, but at test time the model is applied to its *own* outputs for many steps, which it never saw in training.
- **Backward-training** (TRM-style): start from scratch and unroll the recursion with truncated BPTT. Matches test time, but early steps get little useful signal.

DRM does both: take **one step forward** along the corruption process (mask a fraction $r(\tau) = \sin^2(\pi\tau/2)$ of the target cells), initialise the state from the corrupted target, then take **$k$ steps back**, unrolling the recursion $k$ times *with* gradients and applying the loss only at the end:

$$\mathcal{L}_{\text{DRM}} = \mathbb{E}_\tau\big[-\log P_\theta(x \mid H_k)\big],\qquad H_0 = \text{Embed}(\tilde x_\tau)$$

The noise level acts as a difficulty curriculum, and the $k$-step unroll aligns training with test-time recursion. DRM beats TRM on ARC-AGI-2 at matched size (e.g. 9.6% vs 6.3% pass@2 at 7M without pretraining) but is weaker on Sudoku, where long test-time search matters.

### 12.5 Looped Flows

Looped Flows make the flow's denoiser **stateful**, $(\hat x_{t_i}, z_{t_{i+1}}) = \hat D_{t_i}(x_{t_i}, z_{t_i}; c)$, and couple the recurrent state to the flow at inference:

$$x_{t_{i+1}} = x_{t_i} + (t_{i+1}-t_i)\,\frac{\hat x_{t_i} - x_{t_i}}{1-t_i}$$

**Training** runs the recurrence along a *sequence* of local denoising problems, with stop-gradients between steps (so it costs the same as TRM's truncated BPTT):

$$\mathcal{L}_{\text{LF}} = \mathbb{E}\Big[\sum_{i=0}^{k-1}\text{CE}(\hat x_{t_i}, x_1)\Big],\qquad (\hat x_{t_i}, z_{t_{i+1}}) = \hat D_{t_i}\big(I_{t_i}, \text{sg}(z_{t_i}); c\big)$$

The key design choice is **temporal alignment**: times are drawn as $k+1$ sorted uniforms (noise decreases along the rollout), and **one** noise sample $x_0$ and target $x_1$ are shared across all steps. Consecutive problems are then progressively easier versions of the *same* problem. A state that carries partial work forward is directly rewarded at the next step, even though no gradient flows back to the earlier step. Ablations on ARC-AGI-1 (pass@2): full method 58.8%; without decreasing noise 51.6%; without shared noise 56.4%; without time conditioning and interpolant 43.6%.

**Inference** integrates the flow on a grid $0 = t_0 < \dots < t_n = 1$, optionally with a stochastic "noise-backtracking" step that approximates the SDE with the same marginals (§7.3):

$$dx_t = \big[(1+\gamma t)\,b_t(x_t) - \gamma x_t\big]dt + \sigma\sqrt{2\gamma(1-t)}\,dw_t$$

Two knobs for test-time compute follow:
- a **finer time grid** ($n > k$): on Sudoku-Extreme, accuracy rises from 74.5% at 8 steps to 97.9% at 128;
- **multiple noise seeds**: different $x_0$ give different valid solutions on multi-solution problems (N-Queens, graph colouring), and best-of-$N$ ensembling selected by a learned halting/confidence head.

Reported: 97.9% Sudoku-Extreme, 86.7% Maze-Hard, 58.8% ARC-AGI-1, 12.2% ARC-AGI-2 (single trajectory), versus TRM's 87.4 / 85.3 / 44.6 / 7.8 with the same architecture.

**A subtle shortcut.** Because the noise is shared, two interpolants determine the target exactly: $x_1 = \frac{(1-s)I_t - (1-t)I_s}{t-s}$. A model could in principle learn to cancel the noise instead of reasoning. At inference the same trick only reproduces its own first guess, so it cannot help; the authors find that trained models do not rely on it, since improvement with more steps rules this out (notebook 8 §8 shows the identity numerically).

### 12.6 How the three relate

| | Recurrent state | Gradient through time | Noise / corruption | Inference compute knob |
|---|---|---|---|---|
| TRM (looped) | latent $z$ | truncated, loss every step | none | number of loops |
| FRM | previous prediction $s$ | none (detached carry, FPF rollouts) | Gaussian flow on one-hot | recurrence depth + flow steps; stability check |
| DRM | latent + scratchpad | full, through one $k$-step window | masking at level $\tau$ | denoising steps |
| Looped Flows | latent $z$ (hidden features) | truncated, loss every step | shared Gaussian noise, decreasing levels | time-grid resolution, SDE noise, seeds |

Self-conditioning is a special case of a stateful denoiser whose state is its own output. What all three share is the lesson of this whole repository: **local denoising objectives at many noise levels are a remarkably good way to train a long chain of computation without backpropagating through it.**

---

## 13. Unified View: How Everything Connects

### 13.1 The map

```
                         Gaussian path  x_t = a_t x + b_t ε
                                        │
                  optimal predictions = conditional means (§1.5)
            ┌──────────────┬────────────┴───────┬──────────────────┐
         ε̂ = E[ε|x_t]   x̂ = E[x|x_t]     s = ∇ log p_t       v = E[ẋ_t|x_t]
          (DDPM)        (DDIM's x̂₀)    (= −∇E in an EBM)   (flow matching)
            └──────────────┴──── fixed linear maps (§7.5) ───────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
           reverse SDE            probability-flow ODE       scoring (EBM:
       (DDPM ancestral)          (DDIM / FM Euler / RK)      value of E)
                                        │
         ┌──────────────┬───────────────┼────────────────┐
     conditioning    guidance         latent space      steering
     e(y) in net    s_u + w(s_c−s_u)  (VAE: diffuse z)  −γ∇E(x̂ or x_t)
```

### 13.2 The deep equivalences

**DDPM ↔ score:** $\epsilon_\theta = -\sqrt{1-\bar\alpha_t}\;s_\theta$.

**Flow matching ↔ score** (linear path, data at $t=1$): $v_\theta(x,t) = \dfrac{x + (1-t)\,s_\theta(x,t)}{t}$.

**EBM ↔ score:** $s_\theta = -\nabla_x E_\theta$.

**DDIM ↔ flow matching:** the same probability-flow map on different clocks.

**Loss weighting:** written in terms of $\hat x$, every loss in this repo is $\mathbb{E}_t\big[w(t)\,\|\hat x_\theta - x\|^2\big]$ for some weight $w(t)$ (Kingma & Gao, 2023). The $\epsilon$-loss and the velocity loss differ only in $w(t)$ and in the distribution of $t$.

### 13.3 What changes, what stays the same

| Component | Varies across methods | Invariant |
|---|---|---|
| Network output | $\epsilon$, $v$, $\hat x$ or $E$ | all encode the score of $p_t$ |
| Path / schedule | VP, VE, linear | Gaussian corruption $a_t x + b_t\epsilon$ |
| Sampler | SDE (stochastic) or ODE (deterministic) | both integrate a score-driven dynamics |
| Time | discrete or continuous; direction varies | indexes the noise level |
| Conditioning | labels, text, null token | embedding injected into the network |
| Space | pixels or VAE latents | same math, different dimension |
| Control | CFG (score difference), steering ($-\nabla E$) | adds a term to the score/velocity |

### 13.4 Reading path

1. **0a–0c:** calculus, probability, SDEs. Everything later relies on these.
2. **1:** DDPM on a spiral: forward process, $\epsilon$-loss, reverse loop.
3. **2:** the same on images, with a U-Net.
4. **3:** classifier-free guidance.
5. **3b:** the bridge: one path family, four predictions, DDIM, marginal velocity, guidance math.
6. **4:** flow matching.
7. **5:** latent diffusion (VAE + flow matching).
8. **6:** test-time steering.
9. **7:** energy-based diffusion: generation and scoring.
10. **8:** looped reasoning with flows: using many denoising steps to *solve* problems (Sudoku), not just to generate.

---

## 14. Common Pitfalls

- **Time direction.** DDPM: data at $t=0$. Flow matching: data at $t=1$, and $x_0$ is noise.
- **Schedule length.** `linspace(1e-4, 0.02, T)` assumes $T\approx1000$. Shorter $T$ leaves signal at $t=T$ (§2.2).
- **Loss never reaches zero.** It shouldn't (§1.5, §8.4).
- **Sign of steering.** Subtract $\gamma\nabla E$ to *lower* $E$ (§10.2).
- **Steering strength depends on step count** unless scaled by $\Delta t$ (§10.3).
- **CFG ≠ sampling $p(x)p(y|x)^w$** (§6.4).
- **"Straight paths"** describes the training pairs, not the learned trajectories (§8.3).
- **Comparing EBM energies across $t$** is meaningless; compare at a fixed $t$ (§11.7).
- **Data scale.** Schedules assume data of roughly unit variance; normalise inputs (and VAE latents).
- **Embedding continuous $t\in[0,1]$** with frequencies designed for integer $t$ wastes most of the embedding; scale $t$ up (§3).

---

## 15. References

- Sohl-Dickstein et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.*
- Ho, Jain & Abbeel (2020). *Denoising Diffusion Probabilistic Models.* arXiv:2006.11239.
- Song & Ermon (2019). *Generative Modeling by Estimating Gradients of the Data Distribution* (NCSN, annealed Langevin).
- Song et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* (VP/VE-SDE, probability-flow ODE.)
- Song, Meng & Ermon (2021). *Denoising Diffusion Implicit Models* (DDIM).
- Anderson (1982). *Reverse-time diffusion equation models.*
- Vincent (2011). *A Connection Between Score Matching and Denoising Autoencoders.*
- Hyvärinen (2005). *Estimation of Non-Normalized Statistical Models by Score Matching.*
- Efron (2011). *Tweedie's Formula and Selection Bias.*
- Dhariwal & Nichol (2021). *Diffusion Models Beat GANs on Image Synthesis* (classifier guidance).
- Ho & Salimans (2022). *Classifier-Free Diffusion Guidance.* arXiv:2207.12598.
- Karras et al. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM).
- Lipman et al. (2023). *Flow Matching for Generative Modeling.* arXiv:2210.02747.
- Liu, Gong & Liu (2023). *Flow Straight and Fast: Rectified Flow.*
- Albergo & Vanden-Eijnden (2023). *Building Normalizing Flows with Stochastic Interpolants.*
- Tong et al. (2023). *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.*
- Kingma & Gao (2023). *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation.*
- Kingma & Welling (2014). *Auto-Encoding Variational Bayes.*
- Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.*
- Chung et al. (2023). *Diffusion Posterior Sampling for General Noisy Inverse Problems.*
- Grathwohl et al. (2019). *FFJORD* (Hutchinson trace estimator for CNF likelihoods).
- Chen, Zhang & Hinton (2022). *Analog Bits* (self-conditioning).
- Wang et al. (2025). *Hierarchical Reasoning Model* (HRM).
- Jolicoeur-Martineau (2025). *Less is More: Recursive Reasoning with Tiny Networks* (TRM).
- Cameron et al. (2026). *One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion Models.* arXiv:2604.18839.
- Helbling et al. (2026). *Flow Reasoning Models: Scaling Reasoning Through Iterative Self-Refinement.* arXiv:2606.29150.
- Suleymanzade et al. (2026). *Thinking with Looped Flows.* arXiv:2609.11801.
