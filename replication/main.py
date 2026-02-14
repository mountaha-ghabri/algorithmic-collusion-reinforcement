import numpy as np
import matplotlib.pyplot as plt
from input.init import model
from input.qlearning import simulate_game

# Init and simulate
game = model()
game = simulate_game(game)

# ── 1. Extract equilibrium prices ──────────────────────────────────────────
# The equilibrium state: each firm plays argmax of its Q given the state
# In the converged game, state = last period's actions (s = a)
# Find the Nash equilibrium of the converged Q-matrices

def get_eq_actions(game, n_attempts=20):
    """
    Find equilibrium actions robustly.
    Tries multiple starting states, returns most common converged action.
    Also detects cycles and returns cycle midpoint.
    """
    results = []
    
    for _ in range(n_attempts):
        # Random starting state
        s = np.random.randint(0, game.k, size=game.n)
        history = []
        
        for _ in range(500):
            a = np.array([np.argmax(game.Q[(n,) + tuple(s)]) for n in range(game.n)])
            history.append(tuple(a))
            
            # Check for fixed point
            if np.all(a == s):
                results.append(a)
                break
            
            # Check for cycle (last 20 steps)
            if len(history) > 20:
                recent = history[-20:]
                if tuple(a) in recent[:-1]:
                    # Cycle detected — use most common action in cycle
                    from collections import Counter
                    cycle_actions = np.array(recent)
                    modal = np.array([Counter(cycle_actions[:, n]).most_common(1)[0][0] 
                                     for n in range(game.n)])
                    results.append(modal)
                    break
            s = a
    
    # Return most common result across attempts
    if results:
        results_array = np.array(results)
        from scipy import stats
        modal_actions = stats.mode(results_array, axis=0).mode.flatten()
        return modal_actions.astype(int)
    else:
        return np.array([game.k // 2] * game.n)  # fallback to center

eq_actions = get_eq_actions(game)
eq_prices = game.A[eq_actions]
print(f"Equilibrium prices: {eq_prices}")
print(f"Nash prices:        {game.p_minmax[0]}")
print(f"Monopoly prices:    {game.p_minmax[1]}")

# ── 2. Compute profits at equilibrium ──────────────────────────────────────

# Nash: both firms play competitive price simultaneously
p_nash = game.p_minmax[0]  # shape (n,) — already a vector, this is fine
nash_profits = game.compute_profits(p_nash)

# Monopoly: both firms play the joint monopoly price simultaneously
# p_minmax[1] gives symmetric monopoly prices, use them as a vector
p_mono = game.p_minmax[1]  # shape (n,)
mono_profits = game.compute_profits(p_mono)

# Equilibrium profits (already correct)
eq_profits = game.PI[tuple(eq_actions)]

# Delta
delta_index = (eq_profits - nash_profits) / (mono_profits - nash_profits)
print(f"Profit gain index (Delta): {delta_index}")
print(f"Mean Delta: {np.mean(delta_index):.4f}")

# ── 3. Profit gain index (Delta) ───────────────────────────────────────────
delta_index = (eq_profits - nash_profits) / (mono_profits - nash_profits)
print(f"\nProfit gain index (Delta): {delta_index}")
print(f"  0 = Nash/competitive, 1 = full monopoly/collusion")

# ── 4. Impulse response (Figure 3 in paper) ────────────────────────────────
def impulse_response(game, eq_actions, deviate_firm=0, n_periods=30):
    """
    Simulate: firms play equilibrium, then firm 0 deviates by
    undercutting by one price step. Track prices for n_periods after.
    """
    s = eq_actions.copy()
    prices = [game.A[s].copy()]

    # Deviation: firm 0 drops price by one grid step
    s_dev = s.copy()
    s_dev[deviate_firm] = max(0, s[deviate_firm] - 1)
    prices.append(game.A[s_dev].copy())
    s = s_dev

    for _ in range(n_periods - 1):
        a = np.array([np.argmax(game.Q[(n,) + tuple(s)]) for n in range(game.n)])
        prices.append(game.A[a].copy())
        s = a

    return np.array(prices)

irf = impulse_response(game, eq_actions)

# ── 5. Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Price paths after deviation
ax = axes[0]
ax.axhline(game.p_minmax[1][0], color='gray', linestyle='--', label='Monopoly price')
ax.axhline(game.p_minmax[0][0], color='black', linestyle='--', label='Nash price')
ax.plot(irf[:, 0], 'b-o', markersize=4, label='Firm 1 (deviator)')
ax.plot(irf[:, 1], 'r-s', markersize=4, label='Firm 2 (punisher)')
ax.axvline(1, color='orange', linestyle=':', label='Deviation period')
ax.set_xlabel('Period')
ax.set_ylabel('Price')
ax.set_title('Impulse Response to Deviation')
ax.legend()

# Plot 2: Full profit matrix heatmap (optional diagnostic)
ax = axes[1]
im = ax.imshow(game.PI[:, :, 0], origin='lower',
               extent=[game.A[0], game.A[-1], game.A[0], game.A[-1]],
               aspect='auto', cmap='viridis')
ax.scatter(game.A[eq_actions[1]], game.A[eq_actions[0]],
           color='red', s=100, zorder=5, label='Equilibrium')
ax.set_xlabel('Firm 2 price')
ax.set_ylabel('Firm 1 price')
ax.set_title('Firm 1 Profit Matrix')
plt.colorbar(im, ax=ax)
ax.legend()

plt.tight_layout()
plt.savefig('calvano_replication.png', dpi=150)
plt.show()

print("\nDone. Figure saved to calvano_replication.png")