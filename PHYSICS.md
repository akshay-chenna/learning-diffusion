# The Physics of Diffusion Models

A companion to the [learning-diffusion](.) notebooks that reframes every algorithm as an instance of a physical process. This document develops the connections between the code and the disciplines of **classical thermodynamics**, **statistical mechanics**, **Brownian motion**, **fluid dynamics**, **biophysics**, and **nonequilibrium thermodynamics**. Every section maps equations in the notebooks to their physical counterparts, so that the reader can move fluidly between the ML formulation and the physics that inspired it.

**How to read this document.** Some correspondences here are *exact mathematical identities*: the forward process is an Ornstein–Uhlenbeck process, the probability-flow ODE is a continuity equation, and the score of a Boltzmann distribution is a force. Others are *analogies*: useful for intuition, but not literal. Examples are Maxwell's demon, simulated annealing and phage display. Where the distinction matters, the text says which is which. For the derivations themselves, see [THEORY.md](THEORY.md). Many claims are checked numerically in notebooks 0a–0c and 3b.

---

## Table of Contents

1. [Thermodynamic Prologue: Energy, Entropy, and Probability](#1-thermodynamic-prologue-energy-entropy-and-probability)
2. [Brownian Motion and the Forward Diffusion Process](#2-brownian-motion-and-the-forward-diffusion-process)
3. [The Langevin Equation and Score Functions](#3-the-langevin-equation-and-score-functions)
4. [The Fokker-Planck Equation: From Particles to Probability Densities](#4-the-fokker-planck-equation-from-particles-to-probability-densities)
5. [Time Reversal and the Reverse Process](#5-time-reversal-and-the-reverse-process)
6. [The Noise Schedule as a Thermodynamic Protocol](#6-the-noise-schedule-as-a-thermodynamic-protocol)
7. [Fluid Mechanics and Flow Matching](#7-fluid-mechanics-and-flow-matching)
8. [Free Energy, the ELBO, and Variational Autoencoders](#8-free-energy-the-elbo-and-variational-autoencoders)
9. [Coarse-Graining and Latent Diffusion](#9-coarse-graining-and-latent-diffusion)
10. [External Potentials and Gradient-Based Steering](#10-external-potentials-and-gradient-based-steering)
11. [Energy-Based Models: The Full Statistical-Mechanical Treatment](#11-energy-based-models-the-full-statistical-mechanical-treatment)
12. [The Biophysics Connection: Protein Folding and Energy Landscapes](#12-the-biophysics-connection-protein-folding-and-energy-landscapes)
13. [Unified Physical Picture](#13-unified-physical-picture)
14. [Glossary of Cross-Disciplinary Correspondences](#14-glossary-of-cross-disciplinary-correspondences)

---

## 1. Thermodynamic Prologue: Energy, Entropy, and Probability

### 1.1 The Boltzmann Distribution

The single most important equation bridging physics and generative modeling is the **Boltzmann distribution** (1877):

$$p(x) = \frac{1}{Z} \exp\!\left(-\frac{E(x)}{k_B T}\right)$$

where $E(x)$ is the energy of microstate $x$, $k_B$ is Boltzmann's constant, $T$ is temperature, and $Z = \int \exp(-E(x)/k_BT)\,dx$ is the **partition function** (German: *Zustandssumme*, "sum over states").

In machine learning, we absorb $k_BT$ into the energy definition and write simply:

$$p(x) = \frac{1}{Z} \exp\!\big(-E(x)\big)$$

This is the distribution that appears in Notebook 7's EBM implementation. The `EnergyNetwork.compute_energy()` method outputs $E(x,t)$, and the Boltzmann distribution tells us that low-energy configurations are exponentially more probable than high-energy ones.

**Physical meaning:** Consider a gas of $N$ molecules. Each microstate is a point in $6N$-dimensional phase space (positions and momenta). The Boltzmann distribution assigns probability to each microstate proportional to $\exp(-E/k_BT)$. At low temperature, only the lowest-energy states are populated. At high temperature, many states become accessible. This is exactly the coarse-to-fine hierarchy exploited by diffusion models: at high noise levels (analogous to high temperature), the distribution is broad and smooth; at low noise levels (low temperature), probability concentrates sharply on the data manifold.

### 1.2 Entropy and the Second Law

The **Gibbs entropy** of a distribution $p(x)$ is:

$$S = -k_B \int p(x) \log p(x)\, dx$$

The **Second Law of Thermodynamics** states that the entropy of an isolated system never decreases. This has a direct counterpart in the forward diffusion process: as we progressively add noise to data (Notebooks 1-3), we are increasing the entropy of the sample distribution. Pure data has low entropy (concentrated on a manifold); pure Gaussian noise has maximum entropy (among distributions with fixed variance). The forward process is a thermodynamic relaxation toward the maximum-entropy equilibrium.

Conversely, the reverse process (generation) *decreases* the entropy of the sample distribution. There is no paradox, because the reverse dynamics are not those of an isolated system relaxing to equilibrium. They are driven by a time-dependent, externally supplied force (the learned score), much as a refrigerator lowers entropy locally by doing work. A popular metaphor casts the network as a **Maxwell's demon** that uses learned information about the data to sort signal from noise. That is a helpful picture, but it is an analogy, not a thermodynamic accounting: nothing in the algorithm conserves or dissipates physical energy. (Stochastic thermodynamics does give precise links between information, entropy production and the KL divergence between forward and reverse path measures; see §8.2.)

### 1.3 Free Energy and the Partition Function

The **Helmholtz free energy** is:

$$F = -k_BT \log Z = \langle E \rangle - TS$$

where $\langle E \rangle$ is the average energy and $S$ is entropy. Free energy governs equilibrium: a system at constant temperature minimizes $F$, balancing the energetic tendency toward low $E$ against the entropic tendency toward disorder.

The intractability of $Z$ (and hence $F$) is the central problem of computational statistical mechanics. It is *also* the central problem of generative modeling: if we knew $Z$, we could evaluate $p(x)$ exactly for any $x$ and sample from it directly. Every technique in these notebooks -- score matching, variational bounds, flow-based density estimation -- is ultimately a strategy for circumventing the need to compute $Z$.

### 1.4 The Score Function as a Thermodynamic Force

The **score function** $s(x) = \nabla_x \log p(x)$ has a direct physical interpretation. Using the Boltzmann form:

$$s(x) = \nabla_x \log p(x) = -\nabla_x E(x)$$

This is the **negative gradient of the potential energy** -- i.e., the **force** acting on a particle at position $x$ in the energy landscape. In classical mechanics, $F = -\nabla V$ is Newton's equation relating force to potential. The score function is the statistical-mechanical generalization: it is the mean restoring force that drives a Brownian particle toward regions of high probability.

Every model in these notebooks learns to estimate this force field. When the code in Notebook 1 trains `DenoiserNetwork` to predict noise $\epsilon$, it is implicitly learning the force field of a time-dependent potential. When Notebook 7's `EnergyNetwork.compute_score()` calls `autograd` to differentiate energy, it is explicitly computing the force from the potential.

---

## 2. Brownian Motion and the Forward Diffusion Process

### 2.1 Einstein's Theory of Brownian Motion (1905)

In 1905, Einstein derived the diffusion equation to explain the erratic motion of pollen grains suspended in water, observed by botanist Robert Brown in 1827. A Brownian particle of position $x(t)$ satisfies:

$$\frac{dx}{dt} = \sqrt{2D}\, \xi(t)$$

where $D$ is the diffusion coefficient and $\xi(t)$ is Gaussian white noise ($\langle \xi(t) \rangle = 0$, $\langle \xi(t)\xi(t') \rangle = \delta(t-t')$). Over a time interval $\Delta t$, the displacement is:

$$\Delta x = \sqrt{2D \cdot \Delta t}\, \eta, \quad \eta \sim \mathcal{N}(0, I)$$

The *noise part* of the forward diffusion step in Notebooks 1-3 is exactly such a Brownian kick, but the step also shrinks the signal:

$$x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

So the forward process is Brownian motion **plus a restoring drift**. That makes it an Ornstein–Uhlenbeck process (§2.2), not free diffusion. The correspondence is:
- $\sqrt{1-\beta_t}$ is a **dissipative scaling**, the discrete form of the restoring drift $-\tfrac12\beta x\,dt$. It prevents the variance from growing without bound (variance-preserving property)
- $\sqrt{\beta_t}$ plays the role of $\sqrt{2D \cdot \Delta t}$ -- the amplitude of random kicks
- The noise schedule $\beta_t$ is analogous to a time-dependent temperature or diffusion coefficient

Einstein's key result was the **mean-squared displacement**:

$$\langle |\Delta x|^2 \rangle = 2D \cdot \Delta t$$

which grows linearly with time. In the discrete DDPM setting, the cumulative noise over $t$ steps is captured by $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$, and the total variance of the noise component is $1 - \bar{\alpha}_t$, which grows monotonically from 0 to 1 as $t$ goes from 0 to $T$.

### 2.2 The Ornstein-Uhlenbeck Process

The DDPM forward process is not pure Brownian motion -- it includes the signal-scaling factor $\sqrt{1-\beta_t}$. This makes it an instance of the **Ornstein-Uhlenbeck (OU) process**, the prototypical model of a Brownian particle in a harmonic potential:

$$dx = -\theta\, x\, dt + \sigma\, dW_t$$

where $\theta > 0$ is the friction/restoring coefficient, $\sigma$ is the noise strength, and $W_t$ is a Wiener process. The OU process is the continuous-time limit of the DDPM forward step. The $-\theta x$ drift term pulls toward the origin (causing the signal to decay), while the $\sigma\, dW_t$ term adds random perturbations.

The OU process has a **stationary distribution** $p_\infty(x) = \mathcal{N}(0, \sigma^2 / 2\theta)$. The VP-SDE has $\theta = \beta(t)/2$ and $\sigma^2 = \beta(t)$, so $\sigma^2/2\theta = 1$ at every instant: its stationary distribution is exactly $\mathcal{N}(0, I)$, whatever the schedule. The schedule only controls *how fast* the data relaxes toward it. (Notebooks 1 and 7 use short schedules, $T = 200$–$300$, that stop before full relaxation; see THEORY.md §2.2.)

The closed-form shortcut used throughout the notebooks:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$$

is the **exact solution of the OU process** given the initial condition $x_0$. This is implemented in `sample_t()` across all notebooks and in `get_noisy_image()` for the image case. It replaces iterative simulation with a single analytical formula -- the same mathematical convenience that makes the OU process a workhorse in physics.

### 2.3 The Diffusion Equation

The probability density $p(x, t)$ of an ensemble of Brownian particles evolves according to the **diffusion equation** (also called the heat equation):

$$\frac{\partial p}{\partial t} = D \nabla^2 p$$

This is the macroscopic counterpart of the microscopic random walk. Starting from any initial distribution $p(x, 0)$, the solution at later times is a convolution with a Gaussian kernel of width $\sqrt{2Dt}$:

$$p(x, t) = \int p(y, 0)\, G(x - y, t)\, dy, \quad G(x, t) = \frac{1}{(4\pi Dt)^{d/2}} \exp\!\left(-\frac{|x|^2}{4Dt}\right)$$

This is the **physical explanation** for why the forward process destroys structure: the heat equation smooths it out. Pure diffusion alone spreads the distribution forever and has no equilibrium. The OU drift of the VP process supplies the confining harmonic potential, and with it the unique Gaussian equilibrium. (After rescaling space by $e^{B(t)/2}$, the VP process becomes pure diffusion: this is the variance-exploding form $x_0 + \sigma\epsilon$.) The beta schedule controls the rate of this smoothing.

### 2.4 Variance-Preserving Diffusion as Thermalization

In statistical mechanics, **thermalization** is the process by which a system reaches thermal equilibrium with its heat bath. The forward diffusion process is precisely thermalization: the data (initial microstate) is coupled to a thermal reservoir (Gaussian noise) and relaxes to the equilibrium distribution (standard Gaussian).

The variance-preserving property of DDPM ($\text{Var}(x_t) \approx 1$ for all $t$) corresponds to a system in contact with a thermostat that maintains constant temperature. The VP-SDE used in Notebook 7:

$$\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$$

$$dx = -\frac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dW_t$$

is the **continuous-time OU process** with a time-dependent diffusion coefficient, an annealing protocol that gradually increases the coupling to the heat bath.

---

## 3. The Langevin Equation and Score Functions

### 3.1 The Overdamped Langevin Equation

In 1908, Paul Langevin proposed a stochastic differential equation for a Brownian particle in a potential $V(x)$:

$$m\ddot{x} = -\gamma \dot{x} - \nabla V(x) + \sqrt{2\gamma k_BT}\, \xi(t)$$

In the **overdamped limit** (high friction $\gamma$, negligible inertia $m\ddot{x} \approx 0$), this simplifies to:

$$\gamma \dot{x} = -\nabla V(x) + \sqrt{2\gamma k_BT}\, \xi(t)$$

Dividing by $\gamma$ and using $D = k_BT/\gamma$ (Einstein relation):

$$dx = -\nabla V(x)\, dt / \gamma + \sqrt{2D}\, dW_t$$

Setting $k_BT = 1$ (absorbing temperature into the energy) and $\gamma = 1$:

$$dx = -\nabla E(x)\, dt + \sqrt{2}\, dW_t = s(x)\, dt + \sqrt{2}\, dW_t$$

where $s(x) = -\nabla E(x) = \nabla \log p(x)$ is the score function. This is the **Langevin dynamics** equation that appears in the sampling step of Notebook 7 (`langevin_step()`). The particle follows the score (force) plus thermal noise, and its long-time distribution converges to the Boltzmann distribution $p(x) \propto \exp(-E(x))$.

### 3.2 The Einstein Relation and the Fluctuation-Dissipation Theorem

The Einstein relation $D = k_BT/\gamma$ is a special case of the **Fluctuation-Dissipation Theorem (FDT)**: the strength of random fluctuations (diffusion coefficient $D$) is tied to the strength of dissipation (friction $\gamma$) by the temperature $T$. This is not a coincidence -- it is a fundamental requirement for the system to reach the correct equilibrium.

The cleanest ML instance is **Langevin sampling** (§3.3). In $x \leftarrow x + \tfrac{\eta}{2}\nabla\log p + c\sqrt{\eta}\,z$, only $c = 1$ yields $p$ as the stationary distribution. Notebook 0b §4.4 shows that $c < 1$ gives a "too cold" (over-concentrated) sample, $c > 1$ a "too hot" one, and $c = 0$ collapses onto the modes. This is the fluctuation–dissipation relation in its discrete form. The forward VP-SDE obeys it too (§2.2): drift $\beta/2$ and noise variance $\beta$ are locked together so that $\mathcal{N}(0,I)$ is stationary.

In DDPM's reverse step,

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t) \right) + \sigma_t\, z,$$

the drift and noise are likewise tied together: both come from discretising Anderson's reverse-time SDE (§5.1), where the score drift is $g^2\nabla\log p$ and the noise is $g\,d\bar W$. The two standard variance choices $\sigma_t^2 = \beta_t$ and $\sigma_t^2 = \tilde\beta_t$ are the optimal values for two extreme data distributions (Ho et al., 2020). They differ only slightly and both work. Setting $\sigma_t = 0$ outright breaks the balance unless the drift is halved, which is the probability-flow ODE (§4.3).

### 3.3 Langevin Monte Carlo as a Sampling Algorithm

**Langevin Monte Carlo (LMC)** uses the discretized Langevin equation as a sampling algorithm:

$$x_{k+1} = x_k + \eta\, \nabla_x \log p(x_k) + \sqrt{2\eta}\, z_k, \quad z_k \sim \mathcal{N}(0, I)$$

For small step size $\eta$ and many steps, the iterates $\{x_k\}$ converge to samples from $p(x)$. LMC targets **one fixed** distribution and needs many steps to equilibrate. DDPM's reverse loop is a different algorithm: a single Euler–Maruyama pass through the reverse-time SDE, taking **one** step at each noise level, whose drift contains the learned score $\nabla_x \log p_t(x) \approx -\epsilon_\theta(x_t, t) / \sqrt{1-\bar\alpha_t}$ (0c §4.7). The two are close relatives: Song & Ermon's NCSN used annealed LMC, and Song et al.'s predictor–corrector samplers interleave reverse-SDE steps with LMC "corrector" steps. Notebook 7's function `langevin_step` is, despite its name, a reverse-SDE (ancestral) step.

### 3.4 Annealed Langevin Dynamics

A single Langevin chain at fixed temperature struggles with **multimodal distributions** -- it gets trapped in one mode and takes exponentially long to explore others. The solution, proposed by Song and Ermon (2019), is **annealed Langevin dynamics**: run Langevin sampling at a sequence of decreasing noise levels.

The reverse diffusion process has the same coarse-to-fine structure. At high $t$ (high noise), the landscape is smooth and nearly unimodal, so global structure is decided. At low $t$ (low noise), the landscape sharpens and local detail is refined. This resembles **simulated annealing**, which slowly cools a system toward its ground state, but with two differences. Annealing *convolves* the distribution with Gaussian noise rather than raising it to a power $1/T$, and diffusion aims to sample the full distribution, not to find its minimum.

The reverse loop in all notebooks:

```python
for t in range(TIMESTEPS-1, -1, -1):
    # decreasing t = decreasing noise level
    x = denoise_step(x, t)
```

is a sweep from smooth to rugged landscapes, in the spirit of annealing.

---

## 4. The Fokker-Planck Equation: From Particles to Probability Densities

### 4.1 The Fokker-Planck Equation

The **Fokker-Planck equation** (FPE) describes how the probability density $p(x, t)$ evolves under a stochastic process. For the general Ito SDE:

$$dx = f(x, t)\, dt + g(t)\, dW_t$$

the corresponding FPE is:

$$\frac{\partial p}{\partial t} = -\nabla \cdot \big(f(x,t)\, p\big) + \frac{g(t)^2}{2} \nabla^2 p$$

The two terms have vivid physical meanings:
- $-\nabla \cdot (f\, p)$ is the **advection** (or drift) term: probability is carried along by the deterministic velocity field $f$
- $\frac{g^2}{2} \nabla^2 p$ is the **diffusion** term: probability spreads out due to noise

For the DDPM forward process (VP-SDE):

$$dx = -\frac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dW_t$$

the FPE becomes:

$$\frac{\partial p}{\partial t} = \frac{1}{2}\beta(t) \nabla \cdot (x\, p) + \frac{\beta(t)}{2} \nabla^2 p$$

This is the mathematical statement that the data distribution is being smoothed toward a Gaussian. The advection pulls probability toward the origin ($\nabla \cdot (xp)$), while the diffusion smears it out ($\nabla^2 p$).

### 4.2 The Continuity Equation

When the noise term $g = 0$, the FPE reduces to the **continuity equation** of fluid dynamics:

$$\frac{\partial p}{\partial t} + \nabla \cdot (v\, p) = 0$$

This expresses conservation of probability (the "fluid" of probability mass is neither created nor destroyed). It is the equation governing **flow matching** (Notebook 4), where there is no stochastic term -- probability is transported deterministically along the velocity field $v_\theta(x, t)$.

The continuity equation is the same equation that governs mass conservation in fluid mechanics:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$$

where $\rho$ is fluid density and $\mathbf{u}$ is velocity. In flow matching, $p$ plays the role of $\rho$, and the learned velocity field $v_\theta$ plays the role of $\mathbf{u}$. Sampling is the same as tracing fluid parcels forward in time.

### 4.3 The Probability Flow ODE

Every diffusion process (SDE) has an associated **probability flow ODE**: a deterministic ODE whose trajectories trace out the same time-evolving marginal density $p_t(x)$ as the SDE, but without stochastic noise. It was introduced for generative modelling by Song et al. (2021); see also Maoutsa et al. (2020). (Anderson, 1982, is the source of the reverse-time *SDE* in §5.)

$$\frac{dx}{dt} = f(x,t) - \frac{g(t)^2}{2} \nabla_x \log p_t(x)$$

The derivation is a one-liner in the Fokker–Planck picture. Since $\nabla^2 p = \nabla\cdot(p\,\nabla\log p)$, the diffusion term can be rewritten as advection by the velocity $-\tfrac{g^2}{2}\nabla\log p$. The Fokker–Planck equation then becomes a **continuity equation** with velocity $f - \tfrac{g^2}{2}\nabla\log p_t$. Physically, the "osmotic" drift down the density gradient that diffusion produces on average is replaced by an explicit deterministic flow. Notebook 0c §4.8 checks that SDE and ODE produce the same final distribution.

The probability flow ODE is the theoretical bridge between:
- **DDPM** (stochastic reverse SDE, Notebooks 1-3): uses noise in the reverse step. Its deterministic counterpart, run with the same trained network, is **DDIM**.
- **Flow matching** (deterministic ODE, Notebook 4): no noise, just velocity integration.

Careful: DDPM and flow matching use *different* noising paths (variance-preserving vs. linear interpolation), so their marginals $p_t$ at equal $t$ are not the same. What is true is stronger and more useful. Both paths are Gaussian, so after rescaling $x$ and reparameterising time they become the same path, and their probability-flow ODEs then define the **same deterministic noise→data map** (3b §7). Individual sample paths differ by sampler: reverse-SDE paths are jagged (Brownian), ODE paths are smooth.

---

## 5. Time Reversal and the Reverse Process

### 5.1 Anderson's Time-Reversal Theorem

In 1982, Brian Anderson proved that the time-reversal of a diffusion process:

$$dx = f(x,t)\, dt + g(t)\, dW_t$$

is itself a diffusion process:

$$dx = \Big[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\Big] dt + g(t)\, d\bar{W}_t$$

where $\bar{W}_t$ is a reverse-time Wiener process. The crucial term is the **score** $\nabla_x \log p_t(x)$: to run the diffusion backward, you need to know the gradient of the log-probability at every point and every time.

This is the theoretical foundation of all reverse diffusion in these notebooks. The forward process (Brownian motion toward equilibrium) is easy to simulate -- just add noise. The reverse process (reconstruction from noise) requires knowing the score, which is why we train neural networks to estimate it.

### 5.2 Physical Interpretation: Watching a Drop of Ink in Reverse

Imagine dropping ink into water. The ink diffuses outward, spreading from a concentrated drop to a uniform distribution -- this is the forward process (entropy increases, Second Law satisfied).

Now imagine playing the video backward. You would see uniformly distributed ink spontaneously concentrating into a drop. This violates our intuition about the Second Law, but it is mathematically well-defined: the reverse process exists, and it is described by Anderson's formula.

The "trick" that makes it physically consistent is the score term $\nabla_x \log p_t(x)$. This term encodes complete knowledge of the density at every time -- it is the information that a Maxwell's demon would need to reverse the diffusion. In our notebooks, the neural network is the demon, and training it on data provides the information.

### 5.3 Detailed Balance, and Why Reversal Needs Learning

Each forward kernel $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$ satisfies **detailed balance with respect to $\mathcal{N}(0, I)$**:

$$q(x_t \mid x_{t-1})\,\pi(x_{t-1}) = q(x_{t-1} \mid x_t)\,\pi(x_t), \qquad \pi = \mathcal{N}(0, I)$$

where $q(x_{t-1}\mid x_t)$ on the right is the same Gaussian kernel with the arguments swapped. This is the microscopic reversibility of a system **at equilibrium**: if the particles were already distributed as $\pi$, a movie of the forward process would look statistically identical played backwards.

But the forward process **starts from the data**, far from equilibrium. Bayes' rule always gives a reverse kernel,

$$q(x_{t-1}\mid x_t) = \frac{q(x_t\mid x_{t-1})\,p_{t-1}(x_{t-1})}{p_t(x_t)},$$

but because $p_{t-1}$ is not $\pi$, this reverse kernel is **not** the forward kernel. It depends on the unknown data distribution, through the score of $p_t$. That missing information is exactly what the network must learn. The ink-drop video of §5.2 looks wrong in reverse precisely because the system is out of equilibrium.

The Gaussian reverse posterior used in DDPM,

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}\!\Big(x_{t-1};\; \tilde{\mu}_t(x_t, x_0),\; \tilde{\beta}_t I\Big),$$

sidesteps the problem by *also* conditioning on the starting point $x_0$. Given $x_0$, everything is jointly Gaussian and Bayes' rule is a closed-form calculation (0b §1.4). It is plain Bayesian conditioning, not a consequence of detailed balance.

---

## 6. The Noise Schedule as a Thermodynamic Protocol

### 6.1 The Beta Schedule as a Temperature Program

The linear noise schedule used throughout the notebooks:

$$\beta_t = \text{linspace}(10^{-4}, 0.02, T)$$

is a **thermodynamic protocol** -- a prescription for how to change an external control parameter (temperature, coupling strength) over time. In physical terms:

- Small $\beta_t$ at early times: the system is weakly coupled to the heat bath; perturbations are gentle
- Larger $\beta_t$ at late times: the coupling increases; the system is driven more aggressively toward equilibrium

This is analogous to a **heating schedule** in calorimetry or an **annealing protocol** in metallurgy. The rate at which the system is driven matters: too fast and the system falls out of quasi-static equilibrium; too slow and the process is inefficient.

### 6.2 Quasi-Static Processes and Sampling Quality

In thermodynamics, a **quasi-static process** is one carried out infinitely slowly, so that the system remains in equilibrium at every step. A quasi-static process is reversible and achieves maximum thermodynamic efficiency.

In DDPM, using many small steps ($T$ large, $\beta_t$ small) plays a similar role: each step is small enough that the true reverse transition is close to Gaussian, which is what the sampler assumes. This is why ancestral DDPM sampling typically uses $T \approx 1000$ steps. Note that this concerns the *sampler's discretisation*, not physical reversibility. Deterministic samplers of the same model (DDIM, the probability-flow ODE) get good results with 20–100 steps.

Flow matching (Notebook 4) uses **straight-line conditional paths** between independently paired noise and data samples. Those individual lines are straight, but the learned *marginal* velocity field averages over many crossing lines, and its trajectories are curved (THEORY.md §8.3, 3b §5). Independent pairing is **not** optimal transport. Straight paths in the optimal-transport sense, which minimise a kinetic-energy cost that is a natural analogue of dissipated work in finite-time thermodynamics, require an optimal *coupling*: minibatch OT, or rectified flow's "reflow" procedure.

### 6.3 The VP-SDE Schedule and the Fluctuation-Dissipation Relation

Notebook 7 uses the VP-SDE with:

$$\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min}), \quad \beta_{\min} = 0.1, \quad \beta_{\max} = 20.0$$

The integrated schedule $B(t) = \int_0^t \beta(s)\, ds$ determines the signal-to-noise ratio at time $t$:

$$\text{SNR}(t) = \frac{\text{signal power}}{\text{noise power}} = \frac{\exp(-B(t))}{1 - \exp(-B(t))}$$

At $t=0$: $B(0) = 0$, $\text{SNR} = \infty$ (pure signal). At large $t$: $B(t) \gg 1$, $\text{SNR} \to 0$ (pure noise). The schedule interpolates between these extremes.

The fluctuation-dissipation relation ties the noise amplitude $\sqrt{\beta(t)}$ to the drift coefficient $\beta(t)/2$ for the VP process. This ensures that the process relaxes toward $\mathcal{N}(0, I)$ (§2.2). With a mismatched ratio the forward process would relax toward $\mathcal{N}(0, cI)$ with $c \neq 1$, and sampling from $\mathcal{N}(0, I)$ would then start from the wrong distribution.

These are also exactly DDPM's numbers. With $T = 1000$ and `linspace(1e-4, 0.02, 1000)`, the continuous-time limit has $\beta(t) = 1000\,\beta_{\lfloor 1000t\rfloor}$, i.e. $\beta_{\min} = 0.1$ and $\beta_{\max} = 20$ (0c §4.5).

---

## 7. Fluid Mechanics and Flow Matching

### 7.1 The Euler Equations and Deterministic Transport

Flow matching (Notebook 4) is most naturally understood through the lens of **fluid mechanics**. The core equation:

$$\frac{dx}{dt} = v_\theta(x, t)$$

is the **pathline equation** of fluid mechanics: given a velocity field $\mathbf{u}(\mathbf{x}, t)$, the trajectory of a fluid parcel starting at position $\mathbf{x}_0$ is:

$$\frac{d\mathbf{x}}{dt} = \mathbf{u}(\mathbf{x}, t), \quad \mathbf{x}(0) = \mathbf{x}_0$$

In the notebooks, the "fluid" is the probability distribution, and "fluid parcels" are individual samples. The velocity field $v_\theta$ is learned by the neural network, and sampling is performed by advecting particles from the noise distribution to the data distribution.

### 7.2 The Continuity Equation Revisited

The continuity equation:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$$

states that mass (probability) is conserved. In flow matching, this equation constrains the velocity field: $v_\theta$ must transport the noise density $p_0 = \mathcal{N}(0, I)$ to the data density $p_1 = p_{\text{data}}$ while conserving total probability mass.

For an incompressible fluid, $\nabla \cdot \mathbf{u} = 0$ (divergence-free velocity field), and the density is transported without compression or rarefaction. In flow matching, the velocity field is generally *not* divergence-free -- it must compress the broad Gaussian into the concentrated data distribution. This is analogous to a **compressible flow** where the fluid can change density.

### 7.3 Optimal Transport and Straight-Line Paths

The linear interpolation path used in the notebooks:

$$x_t = (1-t) \cdot x_0 + t \cdot x_1$$

moves each *pair* $(x_0, x_1)$ in a straight line at constant velocity $x_1 - x_0$. Each pair is transported optimally *given that pairing*. But the pairs are formed by drawing $x_0$ and $x_1$ **independently**, which is not the optimal coupling, so the paths of different pairs cross.

**Benamou–Brenier.** Optimal transport between $p_0$ and $p_1$ minimises the kinetic energy

$$\mathcal{E} = \int_0^1 \int |v_t(x)|^2 p_t(x)\, dx\, dt$$

over all velocity fields satisfying the continuity equation. Its optimal flow moves particles along straight lines that **never cross**, and in that case the velocity is the gradient of a potential (Brenier's theorem).

**Flow matching's learned field is not this optimum in general.** The network learns the *average* of the crossing conditional velocities, $v^*(x,t) = \mathbb{E}[x_1 - x_0 \mid x_t = x]$. Its trajectories do not cross (it is a single-valued field), but they are **curved**, and the noise→data map it realises is not the OT map (0a §8.5, 3b §5). In fluid terms: superpose many crossing jets and the averaged flow bends. Making flow matching approach OT requires changing the coupling: minibatch-OT pairing (Tong et al., 2023) or iterated "reflow" (Liu et al., 2023).

Notebook 4 shows recognisable digits with 3 Euler steps. That reflects a smooth velocity field on easy data, not the geometric optimality of OT; with very few steps, samples drift toward averaged, blurry results (3b §6).

### 7.4 Euler Integration as a Numerical Method

The sampling loop in Notebook 4:

```python
for i in range(steps):
    t = i * dt
    v = model(x, t_batch, labels)
    x = x + v * dt
```

is the **forward Euler method** for solving ODEs. In computational fluid dynamics (CFD), this is the simplest time-stepping scheme. Higher-order methods (Runge-Kutta, Adams-Bashforth) can improve accuracy for a given step count, but the smoothness of the flow matching velocity field means Euler's method is surprisingly effective.

The step size $dt = 1/N_{\text{steps}}$ must be small compared with the time scale on which the velocity changes along a trajectory, set roughly by $1/\|\nabla v\|$. This is the familiar stability/accuracy constraint of explicit time-stepping. It is analogous to the Courant–Friedrichs–Lewy (CFL) condition of grid-based CFD, though here there is no spatial grid: the constraint comes from **stiffness**. Diffusion ODEs are stiffest near the data end ($t \to 1$ in flow matching, $t \to 0$ in DDPM), where the field sharpens. That is why good samplers use non-uniform time steps that are finer there.

### 7.5 Lagrangian vs. Eulerian Descriptions

Flow matching uses the **Lagrangian description** of fluid motion: we follow individual particles as they move through space. The velocity field $v_\theta(x, t)$ is evaluated at the current position of each particle.

In contrast, the Fokker-Planck equation (Section 4) uses the **Eulerian description**: we watch the density field $p(x, t)$ evolve at fixed spatial locations. Both descriptions are equivalent (they encode the same physics), but the Lagrangian view is more natural for sampling (we want individual samples, not the full density field).

### 7.6 Vorticity and Conservative Fields

In 3D, a velocity field can be decomposed via the **Helmholtz decomposition**:

$$\mathbf{v} = -\nabla \phi + \nabla \times \mathbf{A}$$

where $\phi$ is a scalar potential (irrotational/curl-free component) and $\mathbf{A}$ is a vector potential (solenoidal/divergence-free component). The **vorticity** $\omega = \nabla \times \mathbf{v}$ measures the "swirling" of the flow. In the high-dimensional spaces of diffusion models the cross product doesn't exist. The general notion of vorticity is the **antisymmetric part of the Jacobian**, $\tfrac12(J - J^\top)$, and a field is a gradient exactly when $J$ is symmetric (0a §5).

What is conservative here:

- The **true** score $\nabla_x\log p_t$ is a gradient, so it is irrotational.
- The **true** flow-matching velocity for a Gaussian path is irrotational too. By THEORY.md §8.5, $v^* = \big(x + (1-t)\nabla\log p_t\big)/t$, and both $x = \nabla\tfrac12\|x\|^2$ and $\nabla\log p_t$ are gradients.
- **Learned** $\epsilon$- or $v$-networks (Notebooks 1–6) are unconstrained, so any rotational component comes from approximation error. Training pushes it toward zero, but nothing forbids it.
- The **EBM** in Notebook 7 is conservative by construction:

$$s_\theta(x, t) = -\nabla_x E_\theta(x, t) \implies J_{s_\theta} = -\nabla^2 E_\theta \text{ is symmetric}$$

So the EBM enforces exactly, as an architectural constraint, a property that the ideal model has anyway. Conservative fields derive from a potential (the energy $E_\theta$), so their deterministic dynamics have well-defined fixed points (energy minima) and no limit cycles. In fluid mechanics, irrotational flow is the simplest type: it is completely determined by a scalar potential, which reduces a vector problem to a scalar one.

---

## 8. Free Energy, the ELBO, and Variational Autoencoders

### 8.1 The Variational Principle in Statistical Mechanics

The **Gibbs-Bogoliubov inequality** states that for any trial distribution $q(x)$ and energy function $E(x)$:

$$F \leq \langle E \rangle_q - T S[q] = \langle E \rangle_q + T \int q \log q\, dx$$

where $F$ is the true free energy and $S[q]$ is the entropy of $q$. Equality holds when $q = p_{\text{Boltzmann}}$. This is the **variational free energy principle**: the Boltzmann distribution minimizes the variational free energy over all possible distributions.

Rearranging:

$$\log Z \geq -\langle E \rangle_q + S[q] = -\int q(x) \log \frac{q(x)}{e^{-E(x)}} dx$$

This is exactly the **Evidence Lower Bound (ELBO)** used to train VAEs (Notebook 5):

$$\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\psi(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))$$

The correspondence is precise once the right "Boltzmann distribution" is identified. It is the **posterior over latents** for a fixed image $x$:

$$p(z \mid x) = \frac{p_\psi(x\mid z)\,p(z)}{p(x)} = \frac{e^{-E_x(z)}}{Z_x}, \qquad E_x(z) = -\log p_\psi(x\mid z) - \log p(z), \qquad Z_x = p(x)$$

- $\log p(x)$ is the **log-partition function** $\log Z_x$ (the evidence), so $-\log p(x)$ is the free energy (with $k_BT = 1$)
- $q_\phi(z|x)$ is the trial/variational distribution (the encoder)
- The energy $E_x(z)$ includes **both** the reconstruction term $-\log p_\psi(x\mid z)$ and the prior term $-\log p(z)$. Grouping the prior term with the entropy of $q$ gives the KL term $D_{\text{KL}}(q \,\|\, p(z))$
- The gap $\log p(x) - \text{ELBO} = D_{\text{KL}}(q_\phi(z\mid x)\,\|\,p(z\mid x))$ is the excess free energy of the trial distribution over the equilibrium one (§8.2)
- Maximizing the ELBO = minimizing the variational free energy

### 8.2 The KL Divergence as Relative Entropy

For a system with energy $E$ at temperature $T$, the non-equilibrium free energy of any distribution $q$, $F[q] = \langle E\rangle_q - TS[q]$, exceeds the equilibrium free energy by exactly a KL divergence:

$$F[q] - F_{\text{eq}} = k_BT\, D_{\text{KL}}(q \,\|\, p_{\text{eq}}) \geq 0$$

So the KL divergence measures the **excess free energy** of a non-equilibrium state. It is the maximum work that could be extracted as the system relaxes to equilibrium, or the minimum work needed to prepare $q$ from $p_{\text{eq}}$. Its non-negativity is a form of the Second Law. (In stochastic thermodynamics, the mean dissipated work of a driven process equals $k_BT$ times the KL divergence between the forward and time-reversed *path* distributions; Kawai, Parrondo & Van den Broeck, 2007.)

In the VAE, the KL term $D_{\text{KL}}(q_\phi(z|x) \| \mathcal{N}(0, I))$ penalizes the encoder for producing a latent distribution far from the prior. Read this way, it is the "free-energy cost" of storing information about $x$ in $z$.

### 8.3 The Reparameterization Trick as a Change of Variables

The reparameterization trick in the VAE:

$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

is a **change of variables** in the integral defining the ELBO. In statistical mechanics, such changes of variables are routine: instead of averaging over the complex distribution $q(z|x)$, we average over the simple distribution $p(\epsilon) = \mathcal{N}(0, I)$ and transform the integrand.

This is equivalent to the physicist's trick of writing a thermal average as:

$$\langle A \rangle = \int A(z) \frac{e^{-\beta E(z)}}{Z} dz = \int A(T(\epsilon)) p(\epsilon)\, d\epsilon$$

where $T$ is a deterministic transformation from a simple distribution to the target. The reparameterization trick makes this transformation differentiable, enabling gradient-based optimization of the ELBO.

### 8.4 The VAE Loss as a Free Energy Functional

The VAE loss function (Notebook 5):

$$\mathcal{L}_{\text{VAE}} = \underbrace{\|x - \hat{x}\|^2}_{\text{reconstruction}} + \underbrace{D_{\text{KL}}(q_\phi(z|x) \| \mathcal{N}(0,I))}_{\text{regularization}}$$

is a **free energy functional** $F[q] = \langle E \rangle_q - TS[q]$:
- The reconstruction term is the **expected energy** (how well the decoder reconstructs from the encoded state)
- The KL term is the **negative entropy relative to the prior** (how disordered/structured the encoding is)

Minimizing this loss finds the encoder-decoder pair that achieves the best balance between accurate reconstruction (low energy) and smooth latent structure (high entropy). This is the same tradeoff that governs all equilibrium thermodynamics: energy minimization vs. entropy maximization.

---

## 9. Coarse-Graining and Latent Diffusion

### 9.1 Coarse-Graining in Statistical Mechanics

**Coarse-graining** is the systematic procedure of reducing the number of degrees of freedom in a physical model while preserving the essential large-scale behavior. Examples:

- **Molecular dynamics to continuum mechanics**: Replace individual atoms with continuous fields (density, velocity, temperature)
- **Renormalization group**: Integrate out short-wavelength fluctuations to derive effective theories at longer length scales
- **Collective coordinates**: Replace $N$-particle coordinates with a few order parameters (center of mass, radius of gyration)

### 9.2 The VAE Encoder as a Coarse-Graining Map

In Notebook 5, the VAE encoder compresses a 32x32 image (1,024 dimensions) to a 4x4x4 latent representation (64 dimensions), a 16:1 compression. This is coarse-graining:

$$\text{Encoder: } x \in \mathbb{R}^{1024} \mapsto z \in \mathbb{R}^{64}$$

The latent space retains the **relevant degrees of freedom** (digit identity, stroke style, orientation) while discarding irrelevant ones (pixel-level noise, exact edge positions). This is the "renormalization group" of the image: the encoder integrates out fine-grained details, and the decoder reconstructs an approximation from the coarse-grained description.

The KL regularization ensures that the latent space is smooth and well-structured -- analogous to the requirement in coarse-grained models that the effective potential is well-defined and smooth, even though the underlying microscopic potential may be rugged.

### 9.3 Diffusion in Latent Space as Effective Dynamics

Running the diffusion model in latent space (Notebook 5) is analogous to studying the dynamics of **collective coordinates** rather than individual particles. Instead of tracking every pixel through the forward and reverse diffusion processes, we track the evolution of the coarse-grained representation $z$.

The `LatentDiffuser` in Notebook 5 is the **effective Hamiltonian** of the coarse-grained system: it captures the dynamics of the relevant degrees of freedom without needing to resolve the microscopic details. This is why latent diffusion is faster -- the number of degrees of freedom is drastically reduced, and the remaining degrees of freedom have smoother, lower-dimensional dynamics.

The two-stage training procedure (train VAE, then train diffusion in latent space) mirrors the standard coarse-graining workflow in physics:
1. **Derive the effective model**: Run molecular dynamics simulations to determine the effective interactions between coarse-grained beads (analogous to training the VAE)
2. **Simulate the effective model**: Run dynamics using the coarse-grained Hamiltonian (analogous to training and sampling the latent diffusion model)

### 9.4 The Information Bottleneck

The VAE's latent space can also be understood through **rate–distortion theory** and the closely related **information bottleneck** (Tishby et al., 1999). The information bottleneck compresses an input $X$ into $Z$ while keeping information about a target $Y$:

$$\max_{q(z|x)}\; I(Z; Y) - \beta\, I(Z; X)$$

For an autoencoder the target is the input itself, and the β-VAE objective is a variational bound on the corresponding **rate–distortion** trade-off (Alemi et al., 2018). The averaged KL term $\mathbb{E}_x D_{\text{KL}}(q(z\mid x)\,\|\,p(z))$ upper-bounds the *rate* $I(Z;X)$, the number of nats the code carries about the input, and the reconstruction error is the *distortion*. Minimising distortion $+\;\beta\cdot$rate has the form of a free energy, $\langle E\rangle - T S$, with $\beta$ playing the role of temperature. A high $\beta$ is "hot": codes are cheap, noisy and close to the prior. A low $\beta$ is "cold": codes are precise and expensive. With a Gaussian decoder, $\beta = 2\sigma^2_{\text{dec}}$ (0c §2.3), so the "temperature" is literally the assumed pixel noise level.

---

## 10. External Potentials and Gradient-Based Steering

### 10.1 Particles in External Fields

Notebook 6 introduces gradient-based optimization: modifying the sampling trajectory by adding an external energy gradient. The modified update rule:

$$x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t - \gamma \cdot \nabla_x E_{\text{ext}}(x_t)$$

has a direct physical interpretation. Consider a charged particle in both an ambient potential $V_{\text{ambient}}$ (the learned data distribution) and an external electric field $\mathbf{E}_{\text{ext}}$. The total force is:

$$\mathbf{F}_{\text{total}} = -\nabla V_{\text{ambient}} + q\mathbf{E}_{\text{ext}} = -\nabla (V_{\text{ambient}} - q\phi_{\text{ext}})$$

The particle moves under the combined influence of both potentials. The steering strength $\gamma$ plays the role of the charge $q$: a larger $\gamma$ means the particle responds more strongly to the external field.

### 10.2 The Thickness Energy as a Uniform External Field

The intensity energy function in Notebook 6:

```python
energy = -torch.mean(x)
```

has gradient $\nabla_x E = -\frac{1}{N}\mathbf{1}$ (a constant vector pointing uniformly in the negative direction across all pixels). Subtracting this gradient ($-\gamma \nabla E = +\frac{\gamma}{N}\mathbf{1}$) shifts all pixel values upward -- making the image brighter/thicker.

In physics, this is equivalent to applying a **uniform gravitational field** or **uniform electric field** to the system. The particles (pixels) experience a constant force in one direction, biasing the equilibrium distribution:

$$\tilde{p}(x) \propto p_{\text{data}}(x) \exp(-\gamma E(x)) = p_{\text{data}}(x) \exp(\gamma \cdot \text{mean}(x))$$

This tilts the Boltzmann distribution to favor configurations with higher mean pixel intensity.

### 10.3 The Symmetry Energy as a Harmonic Constraint

The symmetry energy:

```python
energy = torch.mean((x - torch.flip(x, dims=(3,)))**2)
```

measures the $L^2$ distance between an image and its horizontal reflection. This is a **harmonic (quadratic) potential** in the space of asymmetric deviations: $E_{\text{sym}} = k \|x - R(x)\|^2$, where $R$ is the reflection operator.

In molecular physics, harmonic potentials are used extensively to enforce constraints:
- **Harmonic bonds** between atoms: $E = \frac{k}{2}(r - r_0)^2$
- **Umbrella sampling**: adding harmonic biasing potentials to drive rare events
- **Elastic network models** of proteins: harmonic springs between nearby residues

The symmetry constraint acts like an **elastic spring** that pulls the image toward its mirror image. The steering strength $\gamma$ is the spring constant: larger $\gamma$ means a stiffer spring and stronger enforcement of symmetry.

### 10.4 Tilted Distributions and Importance Sampling

The gradient steering *aims at* the **tilted distribution**:

$$\tilde{p}(x) \propto p_{\text{data}}(x) \cdot \exp(-\lambda E(x))$$

It only reaches it approximately. The bias acts on the noisy intermediate states $x_t$, and "add noise, then tilt" is not the same as "tilt, then add noise". The exact noisy-level bias would be $-\log \mathbb{E}[e^{-\lambda E(x_0)} \mid x_t]$, not $\lambda E(x_t)$. This is the same reason classifier-free guidance does not sample $p(x)\,p(y\mid x)^w$ (3b §8.1). Evaluating $E$ on the denoised estimate $\hat x_0(x_t)$ is a better approximation to the exact bias (THEORY.md §10.3). In notebook 6 the bias is also applied per step without a $dt$ factor, so its effective strength $\lambda$ grows with the number of steps.

In statistical mechanics, this is a standard technique called **reweighting** or **exponential tilting**. It appears in:
- **Biased sampling methods** (umbrella sampling, metadynamics): add a bias potential to enhance sampling of rare states
- **Maximum entropy models**: constrain the distribution to satisfy specific expectation values
- **Large deviations theory**: the exponential tilt generates the rate function for rare events

The tilted distribution satisfies the constraint $\langle E \rangle_{\tilde{p}} < \langle E \rangle_{p_{\text{data}}}$ -- the average energy under the steered distribution is lower than under the natural data distribution. This is the mechanism by which steering achieves its effect: it shifts the distribution toward low-energy (desired-property) configurations.

### 10.5 The Tradeoff: Fidelity vs. Physical Realism

As $\gamma$ increases, the steered distribution concentrates more on low-energy configurations. But if $\gamma$ is too large, the exponential tilt overwhelms $p_{\text{data}}$ and the samples move off the data manifold. This is the physicist's version of the **bias-variance tradeoff**: strong biasing improves convergence to the target property but introduces systematic error (samples that are unphysical/unrealistic).

In the notebooks, this manifests as:
- $\gamma = 0$: standard generation, fully on the data manifold
- $\gamma = 100$: strong steering, noticeable effect but some distortion
- $\gamma \to \infty$: the energy dominates and the model loses contact with the data distribution entirely

---

## 11. Energy-Based Models: The Full Statistical-Mechanical Treatment

### 11.1 The Partition Function Problem

Notebook 7 implements the most directly physical model in the repository. The `EnergyNetwork` outputs a scalar energy $E_\theta(x, t)$, and the Boltzmann distribution:

$$p_\theta(x, t) = \frac{1}{Z_\theta(t)} \exp(-E_\theta(x, t))$$

assigns probability to every configuration. The partition function $Z_\theta(t) = \int \exp(-E_\theta(x, t))\, dx$ remains intractable, but the score:

$$s_\theta(x, t) = -\nabla_x E_\theta(x, t)$$

is computable without $Z$ (since $\nabla_x \log Z = 0$). This is the fundamental observation that makes EBMs tractable.

### 11.2 The Squared-Norm Parameterization and Spectral Theory

The energy is parameterized as:

$$E_\theta(x, t) = \|r_\theta(x, t)\|^2 = \sum_i r_i(x, t)^2$$

This ensures $E \geq 0$ (the energy is bounded below). That is necessary but **not sufficient** for a physical Boltzmann distribution. The partition function converges only if $E(x)$ grows fast enough as $|x| \to \infty$ (a flat $E \equiv 0$ is bounded below but has $Z = \infty$). The squared-norm form can represent the required growth, and a trained model learns roughly $E \approx \|x\|^2/2$ far from the data, matching the Gaussian tails of $p_t$. But the architecture does not guarantee it.

The gradient of this energy is:

$$\nabla_x E = 2 J_r^T r$$

where $J_r = \partial r / \partial x$ is the Jacobian matrix. This is a vector–Jacobian product, which autograd computes in one backward pass. The curvature (Hessian) is

$$\nabla^2 E = 2 J_r^\top J_r + 2\sum_i r_i \nabla^2 r_i .$$

Near a minimum where $r \approx 0$, the second term vanishes and the **Gauss–Newton** term $2J_r^\top J_r$ (positive semi-definite) gives the curvature. Its eigenvalues then play the role of the force constants of **normal modes**, analogous to the vibrational frequencies of a molecule around a stable equilibrium.

### 11.3 Denoising Score Matching and the Fluctuation Theorem

The training loss in Notebook 7:

$$\mathcal{L} = \mathbb{E}\left[(1-\bar{\alpha}_t) \left\|s_\theta(x_t, t) + \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}\right\|^2\right]$$

trains the model so that the learned score matches the score of the noised distribution. Per sample, the target $-\epsilon/\sqrt{1-\bar\alpha_t}$ is only the force of the *conditional* distribution $q(x_t\mid x_0)$. Its average over all $x_0$ consistent with $x_t$ is the true marginal force (Tweedie's formula, THEORY.md §1.6), so regression learns the right thing. The weighting factor $(1-\bar{\alpha}_t)$ makes the loss identical to DDPM's $\|\epsilon_\theta - \epsilon\|^2$.

In physical terms this is **force matching** with noisy reference forces. The instantaneous force on a coarse-grained coordinate fluctuates, and its conditional average is the mean force, the gradient of the potential of mean force. Regressing onto instantaneous forces recovers that mean force; this is exactly how multiscale coarse-graining (MS-CG) fits coarse-grained force fields (Noid et al., 2008). That analogy is tighter than the sometimes-invoked comparison with the Jarzynski equality, $\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$. Both recover equilibrium quantities from averages over fluctuating, cheaply simulated quantities, but denoising score matching involves no work or exponential averaging.

### 11.4 Conservative Dynamics and Gradient Systems

Because the EBM score is the gradient of a scalar potential, the resulting dynamics are a **gradient system**:

$$\frac{dx}{dt} = -\nabla E(x)$$

Gradient systems have several important properties from dynamical systems theory:
1. **No oscillations or limit cycles**: trajectories always flow "downhill" in energy
2. **Lyapunov stability**: the energy $E$ serves as a natural Lyapunov function; $dE/dt = -|\nabla E|^2 \leq 0$
3. **Well-defined fixed points**: equilibria occur at $\nabla E = 0$ (critical points of the energy landscape)
4. **Time-reversal asymmetry**: the dynamics are **dissipative** (energy decreases monotonically), not time-reversible

In contrast, standard diffusion models learn an unconstrained vector field for the score, which may have a (small, approximation-induced) rotational component. The learned dynamics can then exhibit slight "circulation" around data modes, behaviour that has no counterpart in a conservative potential landscape. Since the true score is conservative, a well-trained unconstrained model has little circulation in practice (§7.6).

Note also that the *sampling* dynamics are not a pure gradient flow: the reverse SDE adds noise and the $\tfrac12\beta x$ drift, and the time-dependence of $E_\theta(x,t)$ means the landscape changes under the particle. The properties above describe the score field at a fixed $t$.

### 11.5 The Energy Landscape: From Smooth to Rugged

The time-dependent energy landscape $E_\theta(x, t)$ creates a hierarchy of landscapes at different "resolution scales":

| Time | Landscape Character | Physical Analog |
|------|-------------------|-----------------|
| High $t$ | Smooth, nearly convex, single basin | High-temperature liquid; ergodic exploration |
| Medium $t$ | Multi-basin with moderate barriers | Supercooled liquid; metastable states emerge |
| Low $t$ | Sharp, rugged, deep narrow basins | Crystal at low temperature; frozen into local minima |

This progression is identical to the **energy landscape theory** developed for glasses and proteins by Stillinger and Weber (1983) and refined by Wales and others. The key insight is that **temperature controls the effective ruggedness** of the landscape: at high temperature, thermal fluctuations wash out small barriers, and only the gross features of the landscape matter.

In the notebooks, the animated energy landscape visualizations (`plot_energy_landscapes_over_time()`) directly show this progression: at $t=300$ (high noise), the landscape is a smooth bowl; at $t=1$ (low noise), sharp valleys trace out the spiral data manifold.

### 11.6 Scoring, Ranking, and the Boltzmann Factor

The discriminative capability of the EBM -- scoring arbitrary inputs by energy -- is the ML analog of computing **Boltzmann factors** in statistical mechanics. Given two configurations $A$ and $B$, the ratio of their probabilities is:

$$\frac{p(A)}{p(B)} = \exp(-(E(A) - E(B)))$$

This is how the code in Notebook 7's `score_samples()` and `generate_and_rank()` functions work: they evaluate $E_\theta(x, t)$ for multiple configurations and rank them. Lower energy = higher probability = better quality. The partition function $Z$ cancels in the ratio, so it is never needed.

### 11.7 Rejection Sampling as Experimental Selection

The `generate_with_rejection()` function implements:
1. Generate $N_{\text{over}}$ candidates via reverse diffusion
2. Score each with $E_\theta(x, t_{\text{eval}})$
3. Keep only the top $k\%$ by energy

This is **rejection sampling** biased by the energy function, analogous to experimental selection in biophysics:
- **Phage display**: generate $10^{10}$ random peptides, select the ones that bind a target (low binding energy)
- **Directed evolution**: generate protein variants, screen for desired function (low fitness energy)
- **Crystal growth**: nucleate many crystallites, only the thermodynamically stable ones (lowest free energy) grow

The overgenerate-then-filter strategy leverages the fact that **scoring is cheaper than generation** (one forward and backward pass vs. hundreds of sequential denoising steps). Whether it is also *more reliable* depends on how well the energy's ranking tracks quality, which notebook 7 checks with decoys. Two caveats carry over from physics. Energies are defined only up to a constant at each noise level, so compare at a fixed $t_{\text{eval}}$. And selection cannot create what the generator never produced: it improves precision, not coverage.

---

### 11.8 Reasoning as Annealed Relaxation (Notebook 8)

Looped reasoning models (TRM, Flow Reasoning Models, Looped Flows; THEORY.md §12) solve puzzles by iterating a network until the answer stops changing. Physically, this is **relaxation to a fixed point**. When the iteration is (approximately) a gradient step, the fixed point is a minimum of an energy whose minima should be the valid solutions.

The physics explains both their failure modes and the flow-based fix:

- **Quenching into glassy minima.** A deterministic recurrence applied to a rugged landscape behaves like a rapid quench. It settles into the nearest local minimum, a *spurious attractor* (a confidently wrong Sudoku), or keeps wandering (no convergence). Looped Flows report exactly these two failure modes in TRM.
- **Graduated optimisation = annealing.** A flow model learns a *sequence of progressively smoothed* landscapes: at noise level $1-t$ the effective energy is the data energy convolved with Gaussian noise (§1.1, §11.5). Smoothing erases shallow local minima, so starting at high noise and sharpening slowly is a form of **annealing**. Mobahi & Fisher (2015) call this graduated optimisation. Looped Flows train the recurrence along this sequence (decreasing noise, shared noise sample) instead of on a single rugged landscape.
- **Thermal noise helps escape.** The stochastic sampler of Looped Flows adds noise during integration ($\gamma > 0$). This is the same SDE family as §4.3, run at a finite "temperature", and it improves accuracy and the coverage of multiple valid solutions.
- **Stability as a (fallible) verifier.** Flow Reasoning Models score candidates by how little they change under one more iteration, a residual $\|s^{(k+1)} - s^{(k)}\|$. Physically, this checks that the state is a stable equilibrium. It cannot distinguish the ground state from a deep metastable state, so a converged wrong answer is still possible.

## 12. The Biophysics Connection: Protein Folding and Energy Landscapes

### 12.1 The Protein Folding Problem

Protein folding is the process by which a linear amino acid chain adopts its three-dimensional native structure. The folded structure is the **global free energy minimum** in the astronomically large conformational space ($\sim 3^{300}$ possible conformations for a 300-residue protein).

The mathematical framework is identical to the EBM in Notebook 7:
- **Configuration space**: $x = (\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N)$ (3D coordinates of $N$ atoms)
- **Energy function**: $E(x) = E_{\text{bonds}} + E_{\text{angles}} + E_{\text{vdW}} + E_{\text{electrostatic}} + E_{\text{solvation}}$
- **Boltzmann distribution**: $p(x) \propto \exp(-E(x)/k_BT)$
- **Score function**: $\nabla_x \log p(x) = -\nabla_x E(x) / k_BT$ (the force field)

### 12.2 The Funnel Landscape

Protein folding landscapes are often described as **funnels**: broad at the top (many unfolded conformations, high entropy) and narrow at the bottom (few folded conformations, low energy). This is precisely the coarse-to-fine hierarchy of the diffusion energy landscape:

```
High noise (t=300)        Low noise (t=1)
    ▽ ▽ ▽ ▽                    │
   ▽       ▽                   │
  ▽         ▽                  ▼
 ▽     ○     ▽              ─┐ ┌─
▽             ▽               │○│
  (smooth,                  (sharp,
   broad basin)              narrow valley)
```

At high noise: many configurations are roughly equivalent (high-temperature unfolded ensemble). At low noise: only configurations near the data manifold have low energy (low-temperature native state).

Notebook 7's `create_spiral_decoys()` function directly tests this funnel picture: it creates "decoy" structures at increasing levels of distortion and verifies that energy increases monotonically with distortion -- just as decoy protein structures have higher energies than native structures.

### 12.3 ProteinEBM: The Direct Application

Notebook 7 is explicitly inspired by **ProteinEBM** (Jing et al.). The methodology translates directly:

| Spiral EBM (Notebook 7) | ProteinEBM |
|--------------------------|------------|
| 2D spiral points | 3D protein backbone atoms |
| `EnergyNetwork` | Structure scoring network |
| `reverse_diffusion_ebm()` | Structure generation |
| `score_spiral_structure()` | Structure quality assessment |
| `generate_with_rejection()` | Candidate filtering by energy |
| Decoy ranking experiment | DockQ/RMSD correlation with energy |

The key insight is the same: **generation is hard** (many sequential steps, error accumulation) but **scoring is easy** (single evaluation). Generate many candidates, then use the energy function to select the best ones. This is the computational analog of natural selection operating on a population of protein variants.

### 12.4 Molecular Dynamics and Langevin Thermostats

The reverse diffusion sampling in the DDPM-style notebooks (1–3, 7) is closely analogous to a **Langevin thermostat**, the standard method for running molecular dynamics simulations at constant temperature. It is analogous rather than formally equivalent: the reverse SDE's drift is $\tfrac12\beta x + \beta\, s_\theta$ rather than a pure force, and the "temperature" schedule is set by $\beta(t)$. The flow-matching notebooks (4–6) are deterministic and have no thermostat at all.

$$m\ddot{x} = -\nabla V(x) - \gamma \dot{x} + \sqrt{2\gamma k_BT}\, \xi(t)$$

In the overdamped (high-friction) limit used by the notebooks:
1. The force $-\nabla V(x)$ is the learned score $s_\theta(x, t)$
2. The friction $\gamma$ is implicitly set to 1
3. The noise $\sqrt{2k_BT}\, \xi$ is the $\sigma_t z$ term in the reverse step
4. The temperature decreases with $t$ (annealed dynamics)

Every reverse diffusion step is one step of overdamped molecular dynamics in a time-dependent force field.

---

## 13. Unified Physical Picture

Every notebook in this repository implements a variant of the same physical process. Here is the unified view:

### The Forward Process: Thermalization

Data samples are the initial microstates. The forward process couples them to a heat bath and lets them relax to thermal equilibrium (Gaussian noise):

$$\text{Data} \xrightarrow{\text{Brownian motion}} \text{Thermal equilibrium}$$

The heat equation smooths out all structure. The noise schedule controls the coupling strength and the rate of thermalization.

| Notebook | Forward Process | Physical Analog |
|----------|----------------|-----------------|
| 1-3 (DDPM) | $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$ | Ornstein-Uhlenbeck thermalization |
| 4 (Flow Matching) | $x_t = t\,x_{\text{data}} + (1-t)\,\epsilon$ (noise at $t=0$, data at $t=1$) | Linear interpolation (straight per-pair paths; not OT, §7.3) |
| 5 (Latent) | Same as flow matching, in coarse-grained space | Coarse-grained dynamics |
| 7 (EBM) | Discrete DDPM schedule ($T=300$); VP-SDE $x_t = e^{-B(t)/2} x_0 + \sigma(t) \epsilon$ is its continuous limit | OU thermalization |

### The Reverse Process: Controlled Nucleation

Generation starts from thermal equilibrium and produces ordered structure -- the reverse of thermalization:

$$\text{Noise} \xrightarrow[\text{(score/velocity)}]{\text{reverse SDE or probability-flow ODE}} \text{Data}$$

This is analogous to **nucleation and crystal growth**: starting from a disordered melt (high temperature/noise), progressively cooling allows ordered structures to form. The neural network provides the force field that guides nucleation toward the correct structures.

| Notebook | Reverse Process | Physical Analog |
|----------|----------------|-----------------|
| 1-3 (DDPM) | Discretised reverse-time SDE with learned score | Driven, overdamped Langevin dynamics with a time-dependent force |
| 4 (Flow) | Deterministic ODE integration | Fluid transport (continuity equation) |
| 5 (Latent) | Same as Flow, in latent space | CG dynamics + backmapping |
| 6 (Steering) | ODE + external potential gradient | Biased dynamics (umbrella-like bias potential) |
| 7 (EBM) | Reverse SDE with a conservative (gradient) force | Driven Langevin dynamics on a time-dependent energy landscape |

### The Score Function: The Universal Object

Across all methods, the learned quantity is the **score function** $s(x,t) = \nabla_x \log p_t(x)$, which is the **force** in the Langevin equation. Different notebooks parameterize this force differently:

| Parameterization | Formula | Physical Meaning |
|-----------------|---------|-----------------|
| Noise prediction $\epsilon_\theta$ (DDPM) | $s = -\epsilon_\theta / \sqrt{1-\bar\alpha_t}$ | Force inferred from noise estimate |
| Velocity prediction $v_\theta$ (Flow) | $v = \big(x + (1-t)\,s\big)/t$ (linear path) | Force folded into a transport velocity |
| Energy gradient $-\nabla E_\theta$ (EBM) | $s = -\nabla_x E_\theta$ | Force from a conservative potential |

### The Training Objective: Score Matching as Force-Field Fitting

Training the neural network is equivalent to **fitting a force field** to simulation data. In molecular dynamics, force fields (AMBER, CHARMM, OPLS) are parameterized potential energy functions fitted to quantum-mechanical calculations and experimental data. In diffusion models, the network parameters $\theta$ define a time-dependent force field fitted to the data distribution.

The MSE training loss:

$$\mathcal{L} = \mathbb{E}\left[\|s_\theta(x_t, t) - s_{\text{true}}(x_t, t)\|^2\right]$$

minimizes the discrepancy between the learned force field and the true thermodynamic force. This is the same principle as force-matching in computational chemistry: fit the parameters so that the model forces match the reference forces as closely as possible.

### Conditioning: External Fields and Constraints

Conditional generation (Notebooks 2-3, 5) is the application of **external fields** that bias the dynamics. The label embedding $e(y)$ modifies the force field:

$$s_\theta(x, t, y) \neq s_\theta(x, t)$$

The conditioned model generates samples from a **conditional ensemble** -- like simulating a protein under specific conditions (temperature, pH, ligand concentration) rather than in vacuum.

Classifier-free guidance (Notebook 3) amplifies the conditioning signal:

$$s_{\text{guided}} = s_{\text{uncond}} + w(s_{\text{cond}} - s_{\text{uncond}})$$

This resembles **biased sampling**: the bias force $(w-1)\nabla\log p_t(y\mid x)$ pushes the system harder toward the conditioned ensemble than it would go naturally. Unlike enhanced-sampling methods in molecular simulation, CFG does not reweight afterwards to undo its bias. The ensemble it produces is genuinely different from $p(x\mid y)$, and also different from the sharpened $p(x)\,p(y\mid x)^w$ (3b §8.1). The quality-diversity tradeoff ($w$ too large causes artifacts) reflects the physical reality that overly strong external fields distort the natural dynamics.

---

## 14. Glossary of Cross-Disciplinary Correspondences

| Machine Learning Concept | Physics / Biophysics Analog | First Appears |
|-------------------------|---------------------------|---------------|
| Data distribution $p_{\text{data}}(x)$ | Microstate ensemble / Boltzmann distribution | Notebook 1 |
| Forward diffusion | Brownian motion / thermalization | Notebook 1 |
| Noise schedule $\beta_t$ | Temperature program / coupling schedule | Notebook 1 |
| $\bar\alpha_t \to 0$ (pure noise) | Thermal equilibrium / maximum entropy | Notebook 1 |
| Score function $\nabla_x \log p$ | Thermodynamic force / negative energy gradient | Notebook 1 |
| Noise prediction $\epsilon_\theta$ | Rescaled force field estimate | Notebook 1 |
| Reverse diffusion | Time-reversed OU process (Anderson); annealing-like coarse-to-fine | Notebook 1 |
| Probability-flow ODE / DDIM | Fokker–Planck rewritten as a continuity equation | 0c, 3b |
| MSE training loss | Force-matching / score matching | Notebook 1 |
| Residual connections | Perturbative corrections to identity | Notebook 1 |
| Sinusoidal embedding | Fourier modes / spectral decomposition | Notebook 1 |
| U-Net multi-scale | Renormalization group / multi-scale analysis | Notebook 2 |
| Skip connections | Conservation of fine-scale information | Notebook 2 |
| Conditional generation | Constrained ensemble / external field | Notebook 2 |
| Label embedding | Order parameter for symmetry breaking | Notebook 2 |
| CFG amplification ($w > 1$) | Enhanced sampling / biased dynamics | Notebook 3 |
| Quality-diversity tradeoff | Bias-variance tradeoff in sampling | Notebook 3 |
| Label dropout (null token) | Annealing between constrained and unconstrained ensembles | Notebook 3 |
| Velocity field $v_\theta$ | Fluid velocity / Eulerian field | Notebook 4 |
| ODE integration (Euler) | Lagrangian particle tracking in fluid | Notebook 4 |
| Straight conditional paths | Per-pair straight lines; OT (minimum kinetic energy) only with an optimal coupling | Notebook 4 |
| Marginal velocity $v^*$ | Averaged flow of many crossing jets: curved, non-crossing | 0a, 3b |
| Continuity equation | Mass conservation in fluid dynamics | Notebook 4 |
| $dt = 1/N$ (step size) | Explicit time-stepping / stiffness constraint (CFL-like) | Notebook 4 |
| VAE encoder | Coarse-graining map / renormalization | Notebook 5 |
| Latent space | Collective variable space / order parameters | Notebook 5 |
| KL divergence | Relative entropy / excess free energy over equilibrium | Notebook 5 |
| ELBO | Variational free energy bound | Notebook 5 |
| Reparameterization trick | Change of integration variables | Notebook 5 |
| VAE decoder | Backmapping from CG to all-atom | Notebook 5 |
| Latent diffusion | Dynamics of collective coordinates | Notebook 5 |
| Energy function $E(x)$ | Potential energy surface | Notebook 6 |
| $-\gamma \nabla E$ steering | External force / biasing potential | Notebook 6 |
| Intensity energy | Uniform gravitational field | Notebook 6 |
| Symmetry energy | Harmonic constraint / elastic spring | Notebook 6 |
| Tilted distribution $p \cdot e^{-\lambda E}$ | Umbrella sampling / exponential tilting | Notebook 6 |
| Steering strength $\gamma$ | Spring constant / field amplitude | Notebook 6 |
| EBM energy $E_\theta(x,t)$ | Time-dependent Hamiltonian | Notebook 7 |
| $\|r_\theta\|^2$ parameterization | Non-negative energy; Gauss–Newton curvature $2J_r^\top J_r$ near minima | Notebook 7 |
| Denoising score matching | Force matching onto noisy instantaneous forces (MS-CG) | Notebooks 1, 7 |
| Conservative score field | Irrotational / curl-free flow | Notebook 7 |
| Autograd score computation | Analytical differentiation of potential | Notebook 7 |
| Energy landscape hierarchy | Funnel landscape / glass transition | Notebook 7 |
| Ranking by energy | Boltzmann factor comparison | Notebook 7 |
| Rejection sampling | Directed evolution / phage display | Notebook 7 |
| Overgenerate and filter | Monte Carlo with energy selection | Notebook 7 |
| Decoy ranking | Native vs. non-native structure discrimination | Notebook 7 |
| Scoring at medium $t$ | Optimal temperature for discrimination | Notebook 7 |
| Looped reasoning / fixed-point iteration | Relaxation to equilibrium; spurious attractors = glassy minima | Notebook 8 |
| Decreasing noise levels in looped flows | Annealing / graduated optimisation on smoothed landscapes | Notebook 8 |

---

## Coda: Why Physics Matters for Understanding Diffusion Models

The mathematical formalism of diffusion models was not developed in a vacuum: it was inherited directly from a century of work in statistical physics, stochastic processes, and thermodynamics. The forward process is an Ornstein–Uhlenbeck process (Brownian motion in a harmonic trap). The reverse process is its time reversal, a driven Langevin-type dynamics, or equivalently a deterministic flow obeying a continuity equation. The training objective is score matching, which is force-field fitting. The noise schedule is a thermodynamic protocol. The ELBO is a free energy bound. The energy-based model is the Boltzmann distribution.

Understanding these physical roots provides more than historical context. It provides **intuition for design choices**: why the noise and drift of the forward process must be balanced (fluctuation-dissipation), why the reverse process needs a learned score (the forward process starts far from equilibrium, so it is not self-reversing), why straighter paths need fewer steps (smoother, less stiff transport), why EBMs have no curl (conservative forces), and why latent diffusion works (coarse-graining preserves the essential degrees of freedom). The analogies also mark where intuition fails. Guidance and steering are biases without reweighting, and independent coupling is not optimal transport.

Every equation in the code has a physical meaning. Every hyperparameter has a physical interpretation. The reader who understands both the ML and the physics can navigate between them freely, using whichever perspective is more illuminating for the problem at hand.
