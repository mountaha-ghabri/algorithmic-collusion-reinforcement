# The Q-Tables Remember: Persistence of Learned Collusion After Regulatory Disruption

> **Research Paper** | MBA 501: Advanced Decision and Game Theory | Tunis Business School  
> **Author:** Montaha GHABRI  
> **Supervisor:** Dr. Sonia Rebai | **Institution:** Tunis Business School (TBS), University of Tunis

---

## 📄 Abstract

Independent reinforcement learning agents deployed in competitive markets can converge to tacitly collusive pricing without explicit coordination. While the emergence of algorithmic collusion is well-documented, its **stability once learned** remains an open question of direct relevance to antitrust enforcement.

This paper tests four regulatory-style interventions applied after Q-learning agents have converged to collusive pricing in a repeated Bertrand duopoly. Replicating Calvano et al. (2020), the baseline yields an equilibrium price of **1.887** and a Profit Gain Index of **Δ = 0.991**. All four interventions reduce equilibrium prices but do not restore competitive outcomes — recovery rates range from **54.5%** (forced pricing) to **81.8%** (memory reset).

---

## 📚 Table of Contents

1. [Research Questions](#research-questions)
2. [Background](#background)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Discussion & Policy Implications](#discussion--policy-implications)
6. [Repository Structure](#repository-structure)
7. [Replication Guide](#replication-guide)
8. [References](#references)

---

## Research Questions

| # | Question |
|---|---|
| **RQ1** | How stable is algorithmic collusion against market disruptions of different types? |
| **RQ2** | Does intervention duration affect the degree of disruption? |
| **RQ3** | Do symmetric and asymmetric interventions differ in effectiveness? |

### Key Contributions

- First systematic comparison of **four regulatory intervention types** applied to converged algorithmic collusion, using full re-convergence as the measurement standard
- Evidence of **duration insensitivity**, consistent with a disruption-threshold interpretation for forced pricing
- Evidence that **asymmetric interventions** are less disruptive than symmetric ones, with implications for algorithm-replacement remedies
- A compact **robustness check** across a 3×3 (α, β) parameter grid showing qualitative patterns are not confined to a single parameterisation

---

## Background

### Repeated Bertrand Competition

Two symmetric firms compete in an infinitely repeated duopoly. Consumers purchase according to multinomial logit demand:

$$q_i = \frac{\exp((a - p_i)/\mu)}{\sum_j \exp((a - p_j)/\mu) + \exp(a_0/\mu)}$$

**Parameters** (following Calvano et al.): `c = 1`, `a−c = 1`, `a₀ = 0`, `μ = 0.25`, `δ = 0.95`  
This yields **Nash price pᴺ ≈ 1.473** and **monopoly price pᴹ ≈ 1.925**.

### Q-Learning Update Rule

$$Q_{i,t+1}(s_t, a_{i,t}) = (1-\alpha)\,Q_{i,t}(s_t, a_{i,t}) + \alpha\left[\pi_{i,t} + \delta \max_{a'} Q_{i,t}(s_{t+1}, a')\right]$$

### Collusion Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Profit Gain Index (Δ)** | `(π̄ − πᴺ) / (πᴹ − πᴺ)` | 0 = competitive, 1 = full monopoly |
| **Recovery Rate (R)** | `(p̄post − pᴺ) / (p̄base − pᴺ) × 100%` | 0% = collapsed to Nash, 100% = full return to collusion |

---

## Methodology

### Experimental Setup

- **State space:** Previous period prices `sₜ = (p₁,ₜ₋₁, p₂,ₜ₋₁)` → |S| = 225
- **Price grid:** m = 15 points on [1.2, 2.0]
- **Learning rate:** α = 0.15
- **Exploration decay:** εₜ = exp(−βt), β = 4×10⁻⁶
- **Convergence:** Greedy policy unchanged across all 225 states for 10⁵ consecutive periods

### Intervention Designs

| Intervention | Mechanism | Policy Analogue |
|---|---|---|
| **Forced Pricing (k=50)** | Firm 0 locked to pᴺ for 50 periods; Firm 1 updates freely | Consent decree |
| **Forced Pricing (k=100)** | Same mechanism, 100 periods | Extended decree |
| **Exploration Shock** | Both agents set ε=0.5 for 100 periods | Regulatory audit |
| **Memory Reset** | Firm 0 Q-table reset to Q₀; Firm 1 retains memory | Algorithm replacement |

### Robustness Analysis

Sensitivity grid across **9 parameter combinations**:
- α ∈ {0.10, 0.15, 0.25}
- β ∈ {4×10⁻⁶, 10⁻⁵, 10⁻⁴}
- 3 independent sessions per cell

---

## Results

### Baseline Replication

| Metric | Value |
|---|---|
| Nash price (pᴺ) | 1.473 |
| Monopoly price (pᴹ) | 1.925 |
| Equilibrium price (p*) | **1.887** |
| Profit Gain Index (Δ) | **0.991** |
| Convergence (iterations) | 1.4 × 10⁶ |

The impulse response replicates Calvano et al. Figure 3 — prices drop below Nash for ~3 periods (punishment phase) before returning to the collusive equilibrium by period 6.

### Post-Intervention Results

| Intervention | p̄post | R (%) | Δp |
|---|---|---|---|
| Forced (k=50) | 1.699 | **54.5** | −0.188 |
| Forced (k=100) | 1.699 | **54.5** | −0.188 |
| Exploration shock | 1.756 | 68.2 | −0.131 |
| Memory reset | 1.812 | 81.8 | −0.075 |
| Nash (R=0%) | 1.473 | 0.0 | — |
| Baseline (R=100%) | 1.887 | 100.0 | — |

> **Key finding:** All four interventions reduce collusion; none restore competitive pricing. The most disruptive intervention (forced pricing) still leaves agents at R = 54.5% — more than halfway back to collusion.

### Sensitivity Analysis

All 9 cells in the (α, β) grid yield **Δ ≥ 0.63**, confirming substantial collusion is not confined to a single parameterisation. The minimum is 0.63 ± 0.10 at (α=0.15, β=10⁻⁴); the maximum is 0.88 ± 0.12 at the paper baseline.

---

## Discussion & Policy Implications

### Why Interventions Partially Fail

Once converged, Q-tables encode a reinforced collusive strategy. Interventions perturb this encoding but do not erase it. Even mechanistically distinct disruptions share the same partial-recovery pattern, suggesting this is a structural feature of Q-learning in repeated Bertrand games.

### Duration Insensitivity

Forcing Nash pricing for 50 vs. 100 periods produces **identical outcomes** in this run. This suggests a possible threshold mechanism — once Q-value corruption crosses a critical level, extending enforcement adds little benefit.

### Symmetric vs. Asymmetric Interventions

The memory reset (asymmetric) achieves the **least disruption** (R = 81.8%). The intact firm anchors re-convergence toward collusion, effectively re-teaching the reset competitor. This challenges the policy intuition that replacing one firm's algorithm suffices to break coordination.

### Three Policy Conclusions

1. **Reactive enforcement is not futile** — a decline from Δ = 0.991 to 0.55–0.82 represents a real consumer welfare gain
2. **Symmetric enforcement outperforms targeted** — industry-wide audits or coordinated resets should produce larger disruptions than firm-specific remedies
3. **Preventive approaches may dominate reactive ones** — algorithmic design requirements or ex-ante approval regimes could outperform post-detection enforcement

---

## Repository Structure

```
├── LaTeX/                          # Paper source files
├── plots/                          # Generated figures
├── replication/
│   ├── input/
│   │   ├── main.jl                 # Julia implementation
│   │   ├── main.m                  # MATLAB implementation
│   │   └── main.py                 # Python implementation
│   └── scripts/
│       ├── script_01_baseline.py   # Train baseline, save converged game
│       ├── script_02_interventions.py  # Apply & measure all 4 interventions
│       └── script_03_sensitivity.py    # 3×3 (α, β) robustness grid
├── baseline_game.pkl               # Saved converged game object
├── calvano_replication.png         # Impulse response figure
├── intervention_results.png        # Recovery rates figure
├── Paper.pdf                       # Full paper
└── draft.pdf
```

---

## Replication Guide

### Requirements

```bash
pip install numpy matplotlib seaborn pickle
```

### Run Order

```bash
# Step 1 — Train baseline once and save converged game
python replication/scripts/script_01_baseline.py

# Step 2 — Apply all four interventions and measure recovery rates
python replication/scripts/script_02_interventions.py

# Step 3 — Run sensitivity analysis across 3×3 parameter grid
python replication/scripts/script_03_sensitivity.py
```

> **Note:** `script_01_baseline.py` saves `baseline_game.pkl`. All subsequent scripts load this same object, ensuring interventions are applied to an identical baseline. The baseline is trained exactly once.

### Key Parameters

```python
# Economic environment (Calvano et al. defaults)
c = 1           # Marginal cost
mu = 0.25       # Horizontal differentiation
delta = 0.95    # Discount factor
m = 15          # Price grid points on [1.2, 2.0]

# Q-learning
alpha = 0.15    # Learning rate
beta = 4e-6     # Exploration decay
```

---

## Limitations

- Headline intervention results are based on **one baseline training realisation** — interpret as run-specific, not population-level estimates
- Interventions applied once in isolation; repeated or coordinated interventions may produce different dynamics
- Sensitivity analysis uses only 3 sessions per cell — indicative rather than definitive
- Model abstracts from entry, asymmetric costs, demand uncertainty, and multi-product competition

---

## References

| # | Citation |
|---|---|
| [1] | Abada, Lambin & Tóth (2023). Artificial Intelligence, Algorithmic Pricing, and Tacit Collusion. *Journal of Industrial Economics* |
| [2] | Assad et al. (2024). Algorithmic Pricing and Competition: Evidence from the German Retail Gasoline Market. *Management Science* |
| [3] | Baker (2021). Algorithms and Tacit Collusion: An Antitrust Analysis. *Antitrust Law Journal* |
| [4] | **Calvano et al. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. *American Economic Review*** ← baseline replicated |
| [5] | Courthoud (2021). Algorithmic Collusion Replication. GitHub repository |
| [6] | Dafoe et al. (2020). Open Problems in Cooperative AI. arXiv:2012.08630 |
| [7] | Ezrachi & Stucke (2016). *Virtual Competition*. Harvard University Press |
| [8] | Klein (2021). Autonomous Algorithmic Collusion: Q-Learning under Sequential Pricing. *RAND Journal of Economics* |
| [9] | Leibo et al. (2017). Multi-Agent Reinforcement Learning in Sequential Social Dilemmas. *AAMAS* |
| [10] | Mas-Colell, Whinston & Green (1995). *Microeconomic Theory*. Oxford University Press |
| [11] | Mehra (2016). Antitrust and the Roboseller. *Minnesota Law Review* |
| [12] | Singh et al. (2000). Convergence Results for Single-Step On-Policy RL Algorithms. *Machine Learning* |
| [13] | Sutton & Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press |
| [14] | Watkins & Dayan (1992). Q-Learning. *Machine Learning* |

---

<p align="center">
  <em>Tunis Business School · University of Tunis · MBA 501: Advanced Decision and Game Theory</em>
</p>
