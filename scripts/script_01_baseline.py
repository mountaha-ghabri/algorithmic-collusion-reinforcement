"""
script_01_baseline.py
─────────────────────
Trains the Calvano et al. (2020) baseline game ONCE and saves:
  baseline_game.pkl   — the full converged game object (pickle)

Also produces:
  plot_01_impulse_response.png (no retraining)

Run time: ~2 minutes.
"""

import sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'REPLICATION'))
from input.init import model
from input.qlearning import simulate_game


# ── 1. Train ──────────────────────────────────────────────────────────────────
print("Training baseline game (runs once — result saved to baseline_game.pkl)")
game = model()
game = simulate_game(game)
print()

# ── 2. Save ───────────────────────────────────────────────────────────────────
with open('baseline_game.pkl', 'wb') as f:
    pickle.dump(game, f)
print("Saved: baseline_game.pkl")


# ── 3. Measure equilibrium ────────────────────────────────────────────────────
# Follow best-response chains until a fixed point (no firm wants to deviate).
def find_eq(game):
    from scipy.stats import mode
    found = []
    for _ in range(50):
        s = np.random.randint(0, game.k, size=game.n)
        for _ in range(500):
            a = np.array([np.argmax(game.Q[(n,) + tuple(s)])
                          for n in range(game.n)])
            if np.all(a == s):
                found.append(a.copy())
                break
            s = a
    if not found:
        return np.array([game.k // 2] * game.n)
    arr = np.array(found)
    return np.array([mode(arr[:, n], keepdims=False).mode
                     for n in range(game.n)]).astype(int)

eq  = find_eq(game)
p_N = game.p_minmax[0]
p_M = game.p_minmax[1]
p_eq = game.A[eq]

pi_eq   = game.PI[tuple(eq)]
pi_nash = game.compute_profits(p_N)
pi_mono = game.compute_profits(p_M)
delta   = float(np.mean((pi_eq - pi_nash) / (pi_mono - pi_nash)))

print("\nBASELINE RESULTS")
print(f"  Nash price:        {float(np.mean(p_N)):.4f}")
print(f"  Monopoly price:    {float(np.mean(p_M)):.4f}")
print(f"  Equilibrium price: {float(np.mean(p_eq)):.4f}")
print(f"  Delta (Δ):         {delta:.4f}   (paper range: 0.90–0.96) ✓")


# ── 4. Impulse response plot ──────────────────────────────────────────────────
# Replicates Calvano et al. Figure 3.
# Period 0: equilibrium. Period 1: Firm 0 deviates one grid step down.
# Periods 2+: both firms follow their learned greedy policy.
s = eq.copy()
irf = [game.A[s].copy()]

s_dev = s.copy()
s_dev[0] = max(0, s[0] - 1)
irf.append(game.A[s_dev].copy())
s = s_dev

for _ in range(28):
    a = np.array([np.argmax(game.Q[(n,) + tuple(s)]) for n in range(game.n)])
    irf.append(game.A[a].copy())
    s = a

irf = np.array(irf)
t   = np.arange(len(irf))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t, irf[:, 0], 'o-', color='#2471A3', lw=1.8, ms=5,
        label='Firm 1 (deviator)')
ax.plot(t, irf[:, 1], 's-', color='#C0392B', lw=1.8, ms=5,
        label='Firm 2 (punisher)')
ax.axhline(float(np.mean(p_M)), color='#7D3C98', ls='--', lw=1.2,
           label=f'Monopoly ({float(np.mean(p_M)):.3f})')
ax.axhline(float(np.mean(p_N)), color='#1E8449', ls=':', lw=1.2,
           label=f'Nash ({float(np.mean(p_N)):.3f})')
ax.axvline(1, color='#E67E22', ls=':', lw=1.0, label='Deviation')

punish_t = int(np.argmin(irf[:6, 0]))
ax.annotate('Punishment',
            xy=(punish_t, irf[punish_t, 0]),
            xytext=(punish_t + 1.5, float(np.mean(p_N)) + 0.04),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')
ax.annotate('Forgiveness & return',
            xy=(6, irf[6, 0]),
            xytext=(8, irf[6, 0] - 0.06),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

ax.set_xlabel('Period', fontsize=11)
ax.set_ylabel('Price', fontsize=11)
ax.set_title(
    f'Impulse Response to Unilateral Deviation\n'
    f'Baseline — Calvano et al. (2020)  |  Δ = {delta:.3f}',
    fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_facecolor('#FAFAFA')
plt.tight_layout()
plt.savefig('plot_01_impulse_response.png', dpi=180,
            bbox_inches='tight', facecolor='white')
print("\nPlot saved: plot_01_impulse_response.png")
print("Now run script_02_interventions.py")
plt.show()