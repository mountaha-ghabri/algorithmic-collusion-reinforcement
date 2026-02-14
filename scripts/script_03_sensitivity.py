"""
script_03_sensitivity.py
────────────────────────
Loads the baseline game from script_01 for reference Delta, then tests
whether the collusion result is robust across different learning parameters.

For each (alpha, beta) pair: trains a fresh game and measures Delta.
Result: a heatmap where dark = high collusion, light = low collusion.
The baseline cell (alpha=0.15, beta=4e-6) should match script_01's Delta.

Grid: 3 alpha × 3 beta = 9 combinations, 3 sessions each.
Run time: ~20–30 minutes (avoids the slow alpha=0.05 + low-beta combos).

Saves:
  plot_03_sensitivity_heatmap.png
"""

import sys, os, pickle, io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'REPLICATION'))
from input.init import model
from input.qlearning import simulate_game

# ── Load baseline Delta for reference ────────────────────────────────────────
if not os.path.exists('baseline_game.pkl'):
    sys.exit("baseline_game.pkl not found. Run script_01_baseline.py first.")

with open('baseline_game.pkl', 'rb') as f:
    baseline_game = pickle.load(f)


def find_eq(game):
    found = []
    for _ in range(30):
        s = np.random.randint(0, game.k, size=game.n)
        for _ in range(300):
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


def compute_delta(game):
    eq      = find_eq(game)
    pi_eq   = game.PI[tuple(eq)]
    pi_nash = game.compute_profits(game.p_minmax[0])
    pi_mono = game.compute_profits(game.p_minmax[1])
    return float(np.mean((pi_eq - pi_nash) / (pi_mono - pi_nash)))


def train_silent(alpha, beta):
    """Train one game with given parameters; suppress stdout."""
    old, sys.stdout = sys.stdout, io.StringIO()
    try:
        g = model(alpha=alpha, beta=beta)
        g = simulate_game(g)
    except Exception:
        sys.stdout = old
        return None
    finally:
        sys.stdout = old
    return g


# ── Parameter grid ────────────────────────────────────────────────────────────
# Baseline: alpha=0.15, beta=4e-6
# Avoid alpha=0.05 with beta<=1e-6 (extremely slow — 10M+ iterations)
ALPHAS     = [0.10, 0.15, 0.25]
BETAS      = [4e-6, 1e-5, 1e-4]
N_SESSIONS = 3

n_combos = len(ALPHAS) * len(BETAS)
print(f"Sensitivity: {len(ALPHAS)} × {len(BETAS)} = {n_combos} combos, "
      f"{N_SESSIONS} sessions each")
print(f"Baseline: alpha=0.15, beta=4e-6  (marked with * in output)\n")

mean_grid = np.full((len(ALPHAS), len(BETAS)), np.nan)
std_grid  = np.full((len(ALPHAS), len(BETAS)), np.nan)

combo = 0
for i, alpha in enumerate(ALPHAS):
    for j, beta in enumerate(BETAS):
        combo += 1
        tag = " *" if (alpha == 0.15 and beta == 4e-6) else ""
        print(f"  [{combo}/{n_combos}] alpha={alpha}, beta={beta:.0e}{tag}")
        deltas = []
        for s in range(N_SESSIONS):
            sys.stdout.write(f"\r    session {s+1}/{N_SESSIONS}...")
            sys.stdout.flush()
            g = train_silent(alpha, beta)
            if g is not None:
                deltas.append(compute_delta(g))
        print()
        if deltas:
            mean_grid[i, j] = np.mean(deltas)
            std_grid[i, j]  = np.std(deltas)
            print(f"    Δ = {mean_grid[i,j]:.3f} ± {std_grid[i,j]:.3f}")


# ── Heatmap ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

im = ax.imshow(mean_grid, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')

ax.set_xticks(range(len(BETAS)))
ax.set_yticks(range(len(ALPHAS)))
ax.set_xticklabels([f'{b:.0e}' for b in BETAS], fontsize=10)
ax.set_yticklabels([f'{a:.2f}' for a in ALPHAS], fontsize=10)
ax.set_xlabel('Exploration decay rate (β)', fontsize=11)
ax.set_ylabel('Learning rate (α)', fontsize=11)
ax.set_title('Sensitivity Analysis — Profit Gain Index (Δ)\n'
             'Darker = more collusion  |  * = paper baseline',
             fontsize=11, fontweight='bold')

for i in range(len(ALPHAS)):
    for j in range(len(BETAS)):
        v  = mean_grid[i, j]
        sd = std_grid[i, j]
        if not np.isnan(v):
            tag   = '\n*' if (ALPHAS[i] == 0.15 and BETAS[j] == 4e-6) else ''
            txt   = f'{v:.2f}\n±{sd:.2f}{tag}'
            color = 'white' if v > 0.6 else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='Mean Δ')
plt.tight_layout()
plt.savefig('plot_03_sensitivity_heatmap.png', dpi=180,
            bbox_inches='tight', facecolor='white')
print("\nPlot saved: plot_03_sensitivity_heatmap.png")
plt.show()

# LaTeX table
print("\n% LaTeX sensitivity table:")
print(r"\begin{table}[h]\centering")
print(r"\caption{Sensitivity of $\Delta$ to Learning Parameters}")
print(r"\begin{tabular}{l" + "c" * len(BETAS) + r"}\hline\hline")
header = r"$\alpha$ \textbackslash\ $\beta$ & " + \
         " & ".join(f"${b:.0e}$" for b in BETAS) + r" \\"
print(header)
print(r"\hline")
for i, alpha in enumerate(ALPHAS):
    row = f"${alpha:.2f}$"
    for j in range(len(BETAS)):
        v, sd = mean_grid[i, j], std_grid[i, j]
        row  += f" & ${v:.3f} \\pm {sd:.3f}$" if not np.isnan(v) else " & ---"
    row += r" \\"
    print(row)
print(r"\hline\hline\end{tabular}\end{table}")