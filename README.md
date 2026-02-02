# The Collusion Reinforcement Paradox

> When Regulatory Interventions Strengthen Algorithmic Tacit Collusion

**Author:** Montaha Ghabri (moontahaghabry@gmail.com)  
**Institution:** Tunis Business School  
**Course:** Advanced Decision and Game Theory (2025–2026)  
**Status:** Unpublished academic work - Not licensed

---

## Overview

This research investigates a troubling phenomenon in algorithmic pricing: regulatory interventions designed to disrupt algorithmic collusion may paradoxically **strengthen** it rather than weaken it. Using reinforcement learning agents in simulated Bertrand competition, we demonstrate that forced competitive pricing, exploration shocks, and memory resets all result in recovery rates exceeding 110%, indicating collusion reinforcement rather than disruption.

### Key Finding

Independent Q-learning agents converge to collusive pricing (30% toward monopoly levels). After regulatory-style interventions:
- **Forced competitive pricing:** +3.6-3.7% price increase
- **Exploration shocks:** +2.2% price increase  
- **Memory resets:** +3.4% price increase
- **Average recovery rate:** 117.2% (all interventions strengthen collusion)

---

## Repository Structure

```
.
├── CODE/
│   ├── agent.py              # Q-learning agent implementation
│   ├── analysis.py           # Statistical analysis tools
│   ├── environment.py        # Bertrand competition environment
│   ├── plots.py              # Visualization utilities
│   ├── run_all.py            # Main experiment orchestrator
│   ├── simulation.py         # Core simulation logic
│   ├── tables.py             # Results table generation
│   └── test.py               # Unit tests and validation
│
├── figures/                  # Generated plots and visualizations
│   ├── baseline_convergence.pdf
│   ├── intervention_comparison.pdf
│   ├── recovery_timelines.pdf
│   └── ...
│
├── paper/
│   ├── sections/            # LaTeX manuscript sections
│   │   ├── main.aux
│   │   └── references.bib   # Bibliography
│   ├── main.tex             # Main LaTeX document
│   ├── main.bib             # Reference management
│   ├── main.blb            
│   ├── main.pdf             # Compiled manuscript
│   └── *.npy, *.pkl         # Saved results and data
│
└── README.md                # This file
```

---

## Research Questions

1. **RQ1:** How stable is algorithmic collusion learned by independent Q-learning agents to various market disruptions?

2. **RQ2:** What are the effects of different intervention types on collusive equilibria?

3. **RQ3:** Do interventions successfully disrupt coordination, or do they paradoxically reinforce it?

---

## Methodology

### Economic Environment
- **Market structure:** Duopoly with differentiated products
- **Competition type:** Repeated Bertrand price competition
- **Demand specification:** Logit demand model
- **Theoretical benchmarks:**
  - Nash equilibrium price: p^N ≈ 1.268
  - Monopoly price: p^M ≈ 2.250
  - Observed collusive price: p̄ ≈ 1.561

### Q-Learning Agents
- **Algorithm:** Independent Q-learning with ε-greedy exploration
- **State representation:** 1-period memory (previous prices)
- **Learning rate:** α = 0.15
- **Discount factor:** δ = 0.95
- **Exploration:** Time-declining ε from 1.0 → 0.001
- **Price discretization:** 15 equally-spaced points in [1.2, 2.0]

### Intervention Types
1. **Forced Competitive Pricing:** Mandate Nash-level pricing for 50/100 periods
2. **Exploration Shocks:** Temporarily increase ε to 0.5 for 100 periods
3. **Memory Resets:** Reset one agent's Q-table while other retains learning

### Statistical Analysis
- 50 independent training sessions per experiment
- Convergence criterion: Stable greedy policy for 100,000 periods
- Paired t-tests for price comparisons
- Robustness checks across parameter variations

---

## Key Results

| Intervention | Final Price | % Change | Recovery Rate | Status |
|--------------|-------------|----------|---------------|---------|
| Baseline | 1.561 | — | 100.0% | — |
| Forced Competitive (50) | 1.617 | +3.59% | 119.1% | Reinforced |
| Forced Competitive (100) | 1.618 | +3.67% | 119.6% | Reinforced |
| Exploration Shock | 1.596 | +2.22% | 111.8% | Reinforced |
| Memory Reset | 1.615 | +3.44% | 118.3% | Reinforced |

**Recovery Rate Formula:**  
Recovery Rate = (p̄_post - p^N) / (p̄_baseline - p^N) × 100%

Values >100% indicate collusion **reinforcement** (stronger coordination after intervention).

---

## Installation & Usage

### Requirements
```bash
Python 3.11+
numpy
matplotlib
scipy
```

### Running Experiments

1. **Clone repository:**
```bash
git clone [repository-url]
cd algorithmic-collusion-paradox
```

2. **Run baseline experiment:**
```bash
cd CODE
python simulation.py
```

3. **Run all intervention experiments:**
```bash
python run_all.py
```

4. **Generate figures:**
```bash
python plots.py
```

5. **Statistical analysis:**
```bash
python analysis.py
```

### Reproducing Paper Results

All results reported in the manuscript can be reproduced by running:
```bash
python run_all.py --seed 42 --sessions 50
```

Results are saved to `paper/*.npy` and figures to `figures/*.pdf`.

---

## Policy Implications

### Traditional Tools Are Inadequate
One-time fines, mandatory audits, and software updates all failed to disrupt coordination and often strengthened it.

### Recommended Approaches
1. **Prevention over cure:** Focus on preventing collusion emergence rather than disrupting established coordination
2. **Algorithmic diversity requirements:** Mandate heterogeneous pricing algorithms
3. **Enhanced monitoring:** Real-time access to algorithmic decision-making
4. **Novel intervention strategies:** Coordinated resets, permanent exploration parameter changes

---

## Limitations

- **Market structure:** Limited to duopoly; real markets have 3+ firms
- **Algorithmic homogeneity:** All agents use identical Q-learning
- **Simplified interventions:** Real regulatory actions are more complex
- **Simulation-only:** Requires empirical validation in real markets

---

## Future Research Directions

1. Test intervention effectiveness in oligopolistic markets (n ≥ 3)
2. Examine heterogeneous algorithm combinations (Q-learning vs. DQN vs. policy gradients)
3. Design adaptive interventions that respond to agent behavior
4. Empirical validation using real-world pricing data
5. Investigate deep reinforcement learning architectures

---

## Citation

If you reference this work, please cite:

```bibtex
@unpublished{ghabri2025collusion,
  author = {Ghabri, Montaha},
  title = {The Collusion Reinforcement Paradox: When Regulatory Interventions 
           Strengthen Algorithmic Tacit Collusion},
  year = {2025},
  note = {Unpublished manuscript, Tunis Business School},
  email = {moontahaghabry@gmail.com}
}
```

---

## Contact

**Montaha Ghabri**  
Email: moontahaghabry@gmail.com  
Institution: Tunis Business School  
Location: Tunis, Tunisia

---

## License

**This work is currently unlicensed and unpublished.**  
All rights reserved. Do not distribute, reproduce, or build upon this work without explicit permission from the author.

---

## Acknowledgments

This paper was prepared for the Advanced Decision and Game Theory course taught by Dr. Sonia Rebai at Tunis Business School (2025–2026).

---

## Disclaimer

This is academic research conducted for educational purposes. The findings represent preliminary results from simulation studies and have not undergone peer review. The work should not be interpreted as policy recommendations without further empirical validation.

---

*Last updated: February 2026*
