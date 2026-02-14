"""
script_02_interventions.py
──────────────────────────
Loads the trained game saved by script_01_baseline.py.
Applies four interventions, then lets the game RE-CONVERGE using the
same simulate_game() call used in training. Measures the new equilibrium
price and computes recovery rates.

KEY DESIGN: after each intervention, simulate_game() is called on the
modified game. This is the only valid way to measure where the system
ends up — running the exact same convergence procedure, not a custom
recovery loop with an arbitrary exploration schedule.

Recovery Rate R = (p_post - p_Nash) / (p_baseline - p_Nash) × 100%
  R = 100%  →  full return to pre-intervention collusion
  R > 100%  →  collusion STRONGER than before (reinforcement paradox)
  R = 0%    →  prices collapsed to Nash

Saves:
  plot_02_recovery_rates.png   — bar chart of R for each intervention
  plot_02_price_comparison.png — before/after price comparison

Run time: ~10 minutes (4 × ~2 min convergence runs).
"""

import sys, os, pickle, io
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'REPLICATION'))
from input.init import model
from input.qlearning import simulate_game


# ── Load saved baseline ───────────────────────────────────────────────────────
if not os.path.exists('baseline_game.pkl'):
    sys.exit("baseline_game.pkl not found. Run script_01_baseline.py first.")

with open('baseline_game.pkl', 'rb') as f:
    baseline_game = pickle.load(f)

p_N = float(np.mean(baseline_game.p_minmax[0]))
p_M = float(np.mean(baseline_game.p_minmax[1]))

print(f"Loaded baseline game.  Nash={p_N:.4f}  Monopoly={p_M:.4f}")


# ── Shared helpers ────────────────────────────────────────────────────────────

def find_eq(game):
    """Find equilibrium actions via best-response iteration."""
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


def eq_price(game):
    """Return mean equilibrium price of a converged game."""
    eq = find_eq(game)
    return float(np.mean(game.A[eq]))


def reconverge(game):
    """
    Re-run simulate_game() on a modified game object.
    simulate_game() prints to stdout — suppress it so terminal stays clean.
    Returns the reconverged game.
    """
    old, sys.stdout = sys.stdout, io.StringIO()
    try:
        game = simulate_game(game)
    finally:
        sys.stdout = old
    return game


def recovery_rate(p_post, p_base, p_nash):
    return (p_post - p_nash) / (p_base - p_nash) * 100.0


def classify(R):
    if R >= 105:  return "Reinforced (R>100%)"
    if R >= 95:   return "Full Recovery"
    if R >= 10:   return "Partial Recovery"
    return "Permanent Disruption"


# ── Baseline equilibrium price ────────────────────────────────────────────────
p_base = eq_price(baseline_game)
print(f"Baseline equilibrium price: {p_base:.4f}\n")
print("=" * 55)
print("Running interventions (each re-converges via simulate_game)")
print("Expected: ~2 min per intervention")
print("=" * 55)


# ── Intervention 1: Forced pricing k=50 ──────────────────────────────────────
# Firm 0 is locked to the Nash grid index for 50 periods.
# Firm 1 continues updating its Q-table normally.
# After 50 periods both firms resume free learning until re-convergence.
print("\n[1/4] Forced Pricing (k=50)...")

game1 = deepcopy(baseline_game)
nash_idx = int(np.argmin(np.abs(game1.A - p_N)))
s = find_eq(game1)

for _ in range(50):
    a    = np.array([nash_idx,
                     np.argmax(game1.Q[(1,) + tuple(s)])])
    pi   = game1.PI[tuple(a)]
    # Only update Firm 1 — Firm 0 is under forced pricing
    idx  = (1,) + tuple(s) + (a[1],)
    game1.Q[idx] = ((1 - game1.alpha) * game1.Q[idx]
                    + game1.alpha * (pi[1] + game1.delta
                                     * np.max(game1.Q[(1,) + tuple(a)])))
    s = a

# Reset epsilon to allow re-convergence
game1.t = 0
game1   = reconverge(game1)
p1      = eq_price(game1)
R1      = recovery_rate(p1, p_base, p_N)
print(f"  Post-int. price: {p1:.4f}  |  R = {R1:.1f}%  |  {classify(R1)}")


# ── Intervention 2: Forced pricing k=100 ─────────────────────────────────────
print("\n[2/4] Forced Pricing (k=100)...")

game2 = deepcopy(baseline_game)
s = find_eq(game2)

for _ in range(100):
    a   = np.array([nash_idx,
                    np.argmax(game2.Q[(1,) + tuple(s)])])
    pi  = game2.PI[tuple(a)]
    idx = (1,) + tuple(s) + (a[1],)
    game2.Q[idx] = ((1 - game2.alpha) * game2.Q[idx]
                    + game2.alpha * (pi[1] + game2.delta
                                     * np.max(game2.Q[(1,) + tuple(a)])))
    s = a

game2.t = 0
game2   = reconverge(game2)
p2      = eq_price(game2)
R2      = recovery_rate(p2, p_base, p_N)
print(f"  Post-int. price: {p2:.4f}  |  R = {R2:.1f}%  |  {classify(R2)}")


# ── Intervention 3: Exploration shock ────────────────────────────────────────
# Both agents explore with epsilon=0.5 for 100 periods.
# Q-tables update normally throughout the shock.
print("\n[3/4] Exploration Shock (ε=0.5, 100 periods)...")

game3 = deepcopy(baseline_game)
s = find_eq(game3)

for _ in range(100):
    a = np.array([
        np.random.randint(0, game3.k) if 0.5 > np.random.rand()
        else np.argmax(game3.Q[(n,) + tuple(s)])
        for n in range(game3.n)
    ])
    pi = game3.PI[tuple(a)]
    for n in range(game3.n):
        idx = (n,) + tuple(s) + (a[n],)
        game3.Q[idx] = ((1 - game3.alpha) * game3.Q[idx]
                        + game3.alpha * (pi[n] + game3.delta
                                          * np.max(game3.Q[(n,) + tuple(a)])))
    s = a

game3.t = 0
game3   = reconverge(game3)
p3      = eq_price(game3)
R3      = recovery_rate(p3, p_base, p_N)
print(f"  Post-int. price: {p3:.4f}  |  R = {R3:.1f}%  |  {classify(R3)}")


# ── Intervention 4: Memory reset ─────────────────────────────────────────────
# Firm 0's Q-table is reset to its initial values (as if never trained).
# Firm 1 retains full memory. Both re-converge freely.
# Tests whether an experienced firm re-teaches the naive one to collude.
print("\n[4/4] Memory Reset (Firm 0 wiped)...")

game4 = deepcopy(baseline_game)

# Re-initialise Firm 0 using equation (5) from Calvano et al.:
# Q(s, a0) = mean profit across opponent actions / (1 - delta)
pi_avg = np.mean(game4.PI[:, :, 0], axis=1)   # mean over Firm 1's actions
game4.Q[0] = (np.ones(game4.Q[0].shape)
              * pi_avg.mean() / (1 - game4.delta))

game4.t = 0
game4   = reconverge(game4)
p4      = eq_price(game4)
R4      = recovery_rate(p4, p_base, p_N)
print(f"  Post-int. price: {p4:.4f}  |  R = {R4:.1f}%  |  {classify(R4)}")


# ── Summary table ─────────────────────────────────────────────────────────────
results = [
    ("Forced Pricing (k=50)",    p1, R1),
    ("Forced Pricing (k=100)",   p2, R2),
    ("Exploration Shock (ε=0.5)",p3, R3),
    ("Memory Reset (Firm 0)",    p4, R4),
]

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  Baseline price:  {p_base:.4f}")
print(f"  Nash price:      {p_N:.4f}")
print(f"  Monopoly price:  {p_M:.4f}")
print()
print(f"{'Intervention':<28} {'Post price':>10} {'R (%)':>8}  Status")
print("-" * 60)
for name, p, R in results:
    print(f"{name:<28} {p:>10.4f} {R:>8.1f}%  {classify(R)}")
print("=" * 60)

# LaTeX table
print("\n% LaTeX table — paste into paper:")
print(r"\begin{table}[h]\centering")
print(r"\caption{Post-Intervention Equilibrium Prices and Recovery Rates}")
print(r"\begin{tabular}{lccc}\hline\hline")
print(r"Intervention & $\bar{p}_{\text{post}}$ & $R$ (\%) & Status \\\hline")
for name, p, R in results:
    print(f"{name} & {p:.3f} & {R:.1f}\\% & {classify(R)} \\\\")
print(r"\hline\hline\end{tabular}\end{table}")


# ── Plot 1: Recovery rates bar chart ─────────────────────────────────────────
STATUS_COLOR = {
    "Reinforced (R>100%)":  "#922B21",
    "Full Recovery":        "#1E8449",
    "Partial Recovery":     "#E67E22",
    "Permanent Disruption": "#2471A3",
}

labels = [r[0] for r in results]
rates  = [r[2] for r in results]
colors = [STATUS_COLOR[classify(r)] for r in rates]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(range(len(results)), rates, color=colors,
              alpha=0.82, edgecolor='black', linewidth=0.6, width=0.55)

ax.axhline(100, color='black', ls='--', lw=1.3, label='Full recovery (R=100%)')
ax.axhline(0,   color='#1E8449', ls=':', lw=1.0, label='Nash (R=0%)')
if max(rates) > 100:
    ax.axhspan(100, max(rates) + 15, alpha=0.07, color='#922B21')
    ax.text(len(results) - 0.4, max(rates) + 5,
            'Reinforcement\nzone', fontsize=8, color='#922B21', ha='right')

for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, rate + 1.5,
            f'{rate:.1f}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

ax.set_xticks(range(len(results)))
ax.set_xticklabels(labels, rotation=12, ha='right', fontsize=9)
ax.set_ylabel('Recovery Rate (%)', fontsize=11)
ax.set_ylim(0, max(max(rates) + 20, 120))
ax.set_title('Recovery Rate by Intervention\n'
             'R > 100% = collusion reinforcement paradox',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor('#FAFAFA')
plt.tight_layout()
plt.savefig('plot_02_recovery_rates.png', dpi=180,
            bbox_inches='tight', facecolor='white')
print("\nPlot saved: plot_02_recovery_rates.png")


# ── Plot 2: Before / after price comparison ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

x       = np.arange(len(results))
p_posts = [r[1] for r in results]

ax.bar(x - 0.2, [p_base] * len(results), 0.35,
       label=f'Baseline ({p_base:.3f})',
       color='#2C3E50', alpha=0.75, edgecolor='black', lw=0.5)
ax.bar(x + 0.2, p_posts, 0.35,
       label='Post-intervention',
       color=colors, alpha=0.82, edgecolor='black', lw=0.5)

ax.axhline(p_N, color='#1E8449', ls=':', lw=1.2,
           label=f'Nash ({p_N:.3f})')
ax.axhline(p_M, color='#7D3C98', ls='--', lw=1.2,
           label=f'Monopoly ({p_M:.3f})')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=12, ha='right', fontsize=9)
ax.set_ylabel('Equilibrium Price', fontsize=11)
ax.set_ylim(p_N - 0.05, p_M + 0.05)
ax.set_title('Equilibrium Price Before and After Intervention',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor('#FAFAFA')
plt.tight_layout()
plt.savefig('plot_02_price_comparison.png', dpi=180,
            bbox_inches='tight', facecolor='white')
print("Plot saved: plot_02_price_comparison.png")
plt.show()