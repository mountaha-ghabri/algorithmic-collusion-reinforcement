# Stability of Algorithmic Collusion in Independent Q-Learning Agents

## Overview

This project investigates the stability and disruptability of tacit collusion learned by independent Q-learning agents in repeated Bertrand price competition. While Calvano et al. (2020) demonstrated that such collusion can emerge, this work analyzes whether it persists under realistic market disruptions and identifies minimum intervention thresholds for permanent disruption.

## Key Research Questions

1. How stable is learned collusion to temporary competitive pricing?
2. What intervention intensity is required to permanently disrupt collusion?
3. Do agents exhibit "collusion memory" (faster re-learning after disruption)?

## Methodology

- Repeated Bertrand duopoly with differentiated products
- Independent Q-learning with epsilon-greedy exploration
- Four intervention types tested post-convergence:
  * Forced competitive pricing (simulating regulatory fines)
  * Exploration shocks (simulating management changes)
  * Asymmetric learning rates (heterogeneous algorithms)
  * Memory resets (software updates)

## Novel ContributionFirst systematic analysis of algorithmic collusion stability and intervention effectiveness, identifying the "collusion basin of attraction" concept.

## File Structure

- **`simulation.py`** - Main simulation code
- `environment.py` - Bertrand competition environment
- `agent.py` - Q-learning agent implementation
- `analysis.py` - Results analysis and plotting
- `paper/` - LaTeX paper with all sections
