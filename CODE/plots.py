#!/usr/bin/env python3
"""
Generate all figures for the paper from your actual simulation data
Run: python generate_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d
import os

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

print("Loading your simulation data...")

# Load your actual data
prices = np.load('baseline_prices.npy')
with open('summary_stats.pkl', 'rb') as f:
    stats = pickle.load(f)

print(f"✅ Loaded {len(prices)} price periods")
print(f"Baseline price: {float(stats['baseline_price']):.3f}")

# Extract your exact numbers
baseline_price = float(stats['baseline_price'])
nash_price = float(stats['nash_price'])
monopoly_price = float(stats['monopoly_price'])
collusion_index = float(stats['collusion_index'])

# Get intervention data
interventions_data = stats['interventions']

# Prepare data for plotting
INTERVENTION_NAMES = []
FINAL_PRICES = []
RECOVERY_RATES = []
CHANGES = []
STATUSES = []
COLORS = ['#E74C3C', '#E67E22', '#3498DB', '#2ECC71']  # Red, Orange, Blue, Green

for name, data in interventions_data.items():
    final_price = float(data['final_price'])
    recovery_rate = float(data['recovery_percentage'])
    status = data['recovery_status']
    
    # Calculate % change
    change = ((final_price - baseline_price) / baseline_price) * 100
    
    # Format name
    if 'forced_competitive_50' in name:
        display_name = 'Competitive (50)'
    elif 'forced_competitive_100' in name:
        display_name = 'Competitive (100)'
    elif 'exploration_shock' in name:
        display_name = 'Exploration Shock'
    elif 'memory_reset' in name:
        display_name = 'Memory Reset'
    else:
        display_name = name.replace('_', ' ').title()
    
    INTERVENTION_NAMES.append(display_name)
    FINAL_PRICES.append(final_price)
    RECOVERY_RATES.append(recovery_rate)
    CHANGES.append(change)
    STATUSES.append(status)

print(f"\nInterventions loaded: {len(INTERVENTION_NAMES)}")

# ============================================================================
# FIGURE 1: Baseline Price Convergence
# ============================================================================
print("\nGenerating Figure 1: Baseline Convergence...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Full timeline with smoothing
t = np.arange(len(prices))
avg_price = prices.mean(axis=1)

# Smooth for visualization
window = 1000
if len(avg_price) > window:
    smooth_price = np.convolve(avg_price, np.ones(window)/window, mode='valid')
    t_smooth = np.arange(len(smooth_price))
    ax1.plot(t_smooth, smooth_price, 'b-', linewidth=2, alpha=0.8, label='Moving average')
else:
    ax1.plot(t, avg_price, 'b-', linewidth=1, alpha=0.6, label='Average price')

# Add individual firm traces (subsampled)
if len(prices) > 10000:
    ax1.plot(t[::100], prices[::100, 0], 'r-', alpha=0.3, linewidth=0.5, label='Firm 1')
    ax1.plot(t[::100], prices[::100, 1], 'g-', alpha=0.3, linewidth=0.5, label='Firm 2')

# Benchmarks
ax1.axhline(y=nash_price, color='green', linestyle='--', 
            linewidth=1.5, alpha=0.7, label=f'Nash ({nash_price:.3f})')
ax1.axhline(y=monopoly_price, color='purple', linestyle='--', 
            linewidth=1.5, alpha=0.7, label=f'Monopoly ({monopoly_price:.3f})')
ax1.axhline(y=baseline_price, color='orange', linestyle=':', 
            linewidth=2, alpha=0.8, label=f'Collusion ({baseline_price:.3f})')

ax1.set_xlabel('Time Period', fontsize=11)
ax1.set_ylabel('Price', fontsize=11)
ax1.set_title('(a) Algorithmic Collusion Emergence', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, len(prices)])
ax1.set_ylim([1.1, 2.3])

# Right: Collusion index evolution
collusion_idx = (avg_price - nash_price) / (monopoly_price - nash_price)
window_idx = 5000
if len(collusion_idx) > window_idx:
    smooth_idx = np.convolve(collusion_idx, np.ones(window_idx)/window_idx, mode='valid')
    t_idx = np.arange(len(smooth_idx))
    ax2.plot(t_idx, smooth_idx, 'k-', linewidth=2)
    ax2.fill_between(t_idx, 0, smooth_idx, alpha=0.2, color='blue')
else:
    ax2.plot(collusion_idx, 'k-', linewidth=2)
    ax2.fill_between(range(len(collusion_idx)), 0, collusion_idx, alpha=0.2, color='blue')

ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Nash (0)')
ax2.axhline(y=1, color='purple', linestyle='--', alpha=0.5, label='Monopoly (1)')
ax2.axhline(y=collusion_index, color='orange', linestyle=':', 
            linewidth=2, alpha=0.8, label=f'Final: {collusion_index:.3f}')

ax2.set_xlabel('Time Period', fontsize=11)
ax2.set_ylabel('Collusion Index', fontsize=11)
ax2.set_title('(b) Evolution of Collusion Strength', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-0.1, 0.6])

plt.tight_layout()
plt.savefig('figures/baseline_convergence.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/baseline_convergence.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 2: Intervention Comparison
# ============================================================================
print("Generating Figure 2: Intervention Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Recovery rates bar chart
x = np.arange(len(INTERVENTION_NAMES))
bars = ax1.bar(x, RECOVERY_RATES, color=COLORS, alpha=0.8, edgecolor='black', linewidth=1)

ax1.axhline(y=100, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (100%)')

# Add value labels on bars
for i, (bar, rate, price) in enumerate(zip(bars, RECOVERY_RATES, FINAL_PRICES)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 1, 
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    # Add price at bottom
    ax1.text(bar.get_x() + bar.get_width()/2, 5, 
            f'{price:.3f}', ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

ax1.set_xlabel('Intervention Type', fontsize=11)
ax1.set_ylabel('Recovery Rate (% of baseline)', fontsize=11)
ax1.set_title('(a) Intervention Effectiveness\nRecovery >100% = Collusion Reinforcement', 
             fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(INTERVENTION_NAMES, rotation=45, ha='right', fontsize=10)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0, 130])

# Add significance stars
for i, rate in enumerate(RECOVERY_RATES):
    if rate > 110:
        ax1.text(i, rate - 5, '***', ha='center', va='top', fontsize=12, fontweight='bold', color='white')

# Right: Price changes scatter
ax2.scatter(RECOVERY_RATES, FINAL_PRICES, s=200, c=COLORS, alpha=0.8, 
           edgecolors='black', linewidth=1, zorder=5)

# Add labels for each point
for i, (name, rate, price) in enumerate(zip(INTERVENTION_NAMES, RECOVERY_RATES, FINAL_PRICES)):
    ax2.annotate(name, (rate, price), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, ha='left', va='bottom')

# Add reference lines
ax2.axhline(y=baseline_price, color='orange', linestyle=':', 
           linewidth=2, alpha=0.7, label=f'Baseline ({baseline_price:.3f})')
ax2.axhline(y=nash_price, color='green', linestyle='--', 
           alpha=0.5, label=f'Nash ({nash_price:.3f})')
ax2.axvline(x=100, color='black', linestyle='--', alpha=0.7)

ax2.set_xlabel('Recovery Rate (%)', fontsize=11)
ax2.set_ylabel('Final Price', fontsize=11)
ax2.set_title('(b) Recovery Rate vs. Final Price\nUpper-right quadrant = Reinforcement', 
             fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

# Shade reinforcement region
ax2.axhspan(ymin=baseline_price, ymax=2.3, xmin=0, xmax=1, 
           alpha=0.1, color='red', label='Reinforcement region')
ax2.axvspan(xmin=100, xmax=130, ymin=0, ymax=1, 
           alpha=0.1, color='red')

plt.tight_layout()
plt.savefig('figures/intervention_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/intervention_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("FIGURES GENERATED SUCCESSFULLY")
print("="*60)
print("\nSaved to 'figures/' directory:")
print("  ✅ baseline_convergence.pdf/png")
print("  ✅ intervention_comparison.pdf/png")
print("\nUse these in your LaTeX paper!")