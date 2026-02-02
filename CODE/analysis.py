import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import pickle  # ADDED

def load_simulation_data():
    """Load saved simulation data"""
    try:
        # Load baseline prices
        prices = np.load('baseline_prices.npy')
        print(f"✅ Loaded baseline data: {prices.shape} periods")
        
        # Load stability results
        with open('stability_results.pkl', 'rb') as f:
            stability_results = pickle.load(f)
        print(f"✅ Loaded {len(stability_results)} intervention results")
        
        # Load summary stats
        with open('summary_stats.pkl', 'rb') as f:
            summary_stats = pickle.load(f)
        
        return prices, stability_results, summary_stats
    
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("Please run simulation.py first to generate data.")
        return None, None, None

def plot_collusion_dynamics(price_history, summary_stats, intervention_points=None):
    """
    Create professional-quality plots for paper
    Uses YOUR actual data
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    price_array = np.array(price_history)
    
    # Get YOUR actual values from summary_stats
    nash_price = summary_stats['nash_price']
    monopoly_price = summary_stats['monopoly_price']
    baseline_price = summary_stats['baseline_price']
    collusion_index_val = summary_stats['collusion_index']
    
    # 1. Price Trajectory (smoothed)
    ax = axes[0, 0]
    t = np.arange(len(price_array))
    
    # Smooth for cleaner visualization
    smooth_p1 = gaussian_filter1d(price_array[:, 0], sigma=100)
    smooth_p2 = gaussian_filter1d(price_array[:, 1], sigma=100)
    
    ax.plot(t, smooth_p1, 'b-', linewidth=1.5, alpha=0.8, label='Firm 1')
    ax.plot(t, smooth_p2, 'r-', linewidth=1.5, alpha=0.8, label='Firm 2')
    
    # Add YOUR actual benchmarks
    ax.axhline(y=nash_price, color='green', linestyle='--', 
               alpha=0.7, linewidth=1, label=f'Nash ({nash_price:.3f})')
    ax.axhline(y=monopoly_price, color='purple', linestyle='--', 
               alpha=0.7, linewidth=1, label=f'Monopoly ({monopoly_price:.3f})')
    ax.axhline(y=baseline_price, color='orange', linestyle=':', 
               alpha=0.7, linewidth=1.5, label=f'Your collusion ({baseline_price:.3f})')
    
    # Mark intervention points
    if intervention_points:
        colors = ['orange', 'red', 'brown', 'gray']
        for i, (name, t_point) in enumerate(intervention_points):
            ax.axvline(x=t_point, color=colors[i % len(colors)], 
                      linestyle=':', alpha=0.7, linewidth=1)
            ax.text(t_point, ax.get_ylim()[1]*0.95, name, 
                   rotation=90, fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Time Period', fontsize=11)
    ax.set_ylabel('Price', fontsize=11)
    ax.set_title('(a) Price Dynamics: Algorithmic Collusion Emergence', 
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim([0, len(price_array)])
    ax.set_ylim([1.1, 2.3])
    ax.grid(True, alpha=0.3)
    
    # 2. Collusion Index Evolution
    ax = axes[0, 1]
    
    # Compute rolling collusion index with YOUR actual values
    window = 5000
    rolling_avg = np.convolve(price_array.mean(axis=1), 
                             np.ones(window)/window, mode='valid')
    
    collusion_index = (rolling_avg - nash_price) / (monopoly_price - nash_price)
    t_roll = np.arange(len(collusion_index))
    
    ax.plot(t_roll, collusion_index, 'k-', linewidth=2, alpha=0.8)
    ax.fill_between(t_roll, 0, collusion_index, alpha=0.2, color='blue')
    
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='purple', linestyle='--', alpha=0.5)
    ax.axhline(y=collusion_index_val, color='orange', linestyle=':', 
               alpha=0.7, linewidth=2, label=f'Your level ({collusion_index_val:.3f})')
    
    ax.set_xlabel('Time Period', fontsize=11)
    ax.set_ylabel('Collusion Index', fontsize=11)
    ax.set_title('(b) Evolution of Collusion Strength', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.1, 0.6])  # Adjusted for your range (0.0-0.4)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 3. Phase Diagram (last 50k periods)
    ax = axes[1, 0]
    
    last_50k = price_array[-50000:] if len(price_array) > 50000 else price_array
    
    # Create 2D histogram
    hb = ax.hexbin(last_50k[:, 0], last_50k[:, 1], 
                   gridsize=30, cmap='Blues', bins='log', 
                   mincnt=1, edgecolors='none')
    
    # Add 45-degree line (perfect coordination)
    min_p = min(last_50k[:, 0].min(), last_50k[:, 1].min())
    max_p = max(last_50k[:, 0].max(), last_50k[:, 1].max())
    ax.plot([min_p, max_p], [min_p, max_p], 
           'r--', alpha=0.7, linewidth=1.5, label='Perfect coordination')
    
    # Add YOUR actual Nash and Monopoly points
    ax.scatter([nash_price], [nash_price], color='green', s=100, 
              edgecolor='black', zorder=5, label=f'Nash ({nash_price:.3f})')
    ax.scatter([monopoly_price], [monopoly_price], color='purple', s=100, 
              edgecolor='black', zorder=5, label=f'Monopoly ({monopoly_price:.3f})')
    ax.scatter([baseline_price], [baseline_price], color='orange', s=150, 
              marker='*', edgecolor='black', zorder=5, 
              label=f'Your collusion ({baseline_price:.3f})')
    
    ax.set_xlabel('Firm 1 Price', fontsize=11)
    ax.set_ylabel('Firm 2 Price', fontsize=11)
    correlation = np.corrcoef(last_50k[:, 0], last_50k[:, 1])[0,1]
    ax.set_title(f'(c) Price Coordination Phase Diagram\nCorrelation: {correlation:.3f}', 
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    # Add colorbar
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Frequency (log scale)', fontsize=9)
    
    # 4. Intervention Recovery Analysis - USING YOUR ACTUAL DATA
    ax = axes[1, 1]
    
    # Get YOUR actual intervention results
    interventions_data = summary_stats['interventions']
    
    # Prepare data for plotting
    names = []
    recovery_rates = []
    final_prices = []
    status_colors = []
    
    for inter_name, inter_data in interventions_data.items():
        names.append(inter_name.replace('_', ' ').title())
        recovery_rates.append(inter_data['recovery_percentage'] / 100.0)  # Convert % to fraction
        final_prices.append(inter_data['final_price'])
        
        # Color coding based on recovery status
        if inter_data['recovery_status'] == 'full':
            status_colors.append('green')
        elif inter_data['recovery_status'] == 'partial':
            status_colors.append('orange')
        else:  # disrupted
            status_colors.append('red')
    
    # Sort by recovery rate (for better visualization)
    sorted_indices = np.argsort(recovery_rates)
    names = [names[i] for i in sorted_indices]
    recovery_rates = [recovery_rates[i] for i in sorted_indices]
    final_prices = [final_prices[i] for i in sorted_indices]
    status_colors = [status_colors[i] for i in sorted_indices]
    
    bars = ax.barh(names, recovery_rates, color=status_colors, alpha=0.7)
    
    # Add value labels (final prices)
    for i, (bar, price) in enumerate(zip(bars, final_prices)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{price:.3f}', va='center', fontsize=9)
    
    # Add reference lines
    ax.axvline(x=1.0, color='blue', linestyle='--', alpha=0.5, 
               label='Baseline (100%)')
    ax.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Nash (0%)')
    
    # Add percentage labels
    for i, (bar, rate) in enumerate(zip(bars, recovery_rates)):
        percentage = rate * 100
        if bar.get_width() > 0.05:  # If there's space inside the bar
            ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                   f'{percentage:.0f}%', va='center', ha='center', 
                   color='white', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Recovery Rate (% of baseline collusion)', fontsize=11)
    ax.set_title('(d) Intervention Effectiveness\n>100% = Stronger collusion after intervention', 
                fontsize=12, fontweight='bold')
    ax.set_xlim([0, max(1.5, max(recovery_rates) * 1.1)])  # Dynamic limit
    ax.legend(loc='lower right', fontsize=9)
    
    # Add annotations
    if max(recovery_rates) > 1.1:
        ax.text(1.25, 0.5, 'Super-collusion\n(>100%)', transform=ax.transAxes,
               fontsize=10, ha='center', alpha=0.7, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig('collusion_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('collusion_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig('collusion_analysis.eps', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print YOUR actual statistics
    print("\n" + "="*60)
    print("YOUR SIMULATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Statistics:")
    print(f"Average price: {price_array.mean():.3f}")
    print(f"Standard deviation: {price_array.std():.3f}")
    print(f"Price correlation: {correlation:.3f}")
    
    print(f"\nCollusion Analysis:")
    print(f"Nash price: {nash_price:.3f}")
    print(f"Monopoly price: {monopoly_price:.3f}")
    print(f"Your collusive price: {baseline_price:.3f}")
    print(f"Collusion index: {collusion_index_val:.3f} (0 = Nash, 1 = Monopoly)")
    
    if len(price_array) > 100000:
        early = price_array[:50000]
        late = price_array[-50000:]
        print(f"\nEarly phase (first 50k periods): {early.mean():.3f}")
        print(f"Late phase (last 50k periods): {late.mean():.3f}")
        print(f"Collusion increase: {(late.mean() - early.mean()):.3f}")
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(early.flatten(), late.flatten(), 
                                         equal_var=False)
        print(f"Statistical significance: t={t_stat:.2f}, p={p_value:.4f}")
        if p_value < 0.05:
            print(f"✓ Statistically significant collusion (p < 0.05)")
    
    print(f"\nIntervention Effectiveness:")
    for inter_name, inter_data in interventions_data.items():
        change_pct = ((inter_data['final_price'] - baseline_price) / baseline_price * 100)
        arrow = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
        print(f"  {inter_name}: {inter_data['final_price']:.3f} "
              f"({arrow}{abs(change_pct):.1f}%) - {inter_data['recovery_status']}")
    
    return fig

def create_intervention_timeline(baseline_prices, stability_results, summary_stats):
    """Create timeline showing interventions and recovery - USING YOUR DATA"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot baseline
    t_baseline = np.arange(len(baseline_prices))
    ax.plot(t_baseline, baseline_prices.mean(axis=1), 
           'b-', alpha=0.7, linewidth=1, label='Baseline collusion')
    
    # Plot each intervention recovery
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    
    # Use your actual intervention points
    intervention_points = []
    start_t = len(baseline_prices)
    
    for i, (name, result) in enumerate(stability_results.items()):
        if 'post_prices' in result:
            post_prices = result['post_prices']
            if len(post_prices) > 0:
                t_post = start_t + np.arange(len(post_prices))
                ax.plot(t_post, post_prices.mean(axis=1), 
                       color=colors[i % len(colors)], 
                       linewidth=1.5, alpha=0.8, label=f'{name}: {result["final_avg_price"]:.3f}')
                intervention_points.append((name, start_t))
                start_t += len(post_prices)
    
    # Add YOUR actual benchmarks
    nash_price = summary_stats['nash_price']
    monopoly_price = summary_stats['monopoly_price']
    baseline_price = summary_stats['baseline_price']
    
    ax.axhline(y=nash_price, color='green', linestyle='--', 
               alpha=0.5, label=f'Nash ({nash_price:.3f})')
    ax.axhline(y=monopoly_price, color='purple', linestyle='--', 
               alpha=0.5, label=f'Monopoly ({monopoly_price:.3f})')
    ax.axhline(y=baseline_price, color='orange', linestyle=':', 
               alpha=0.7, linewidth=2, label=f'Baseline collusion ({baseline_price:.3f})')
    
    # Mark intervention starts
    for i, (name, t_point) in enumerate(intervention_points):
        ax.axvline(x=t_point, color=colors[i % len(colors)], 
                  linestyle=':', alpha=0.5, linewidth=1)
        ax.text(t_point, ax.get_ylim()[1]*0.97, name.split('_')[0], 
               rotation=90, fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Time Period', fontsize=11)
    ax.set_ylabel('Average Price', fontsize=11)
    ax.set_title('Intervention Timeline and Recovery Dynamics', 
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('intervention_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('intervention_timeline.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

# Main execution - LOADS YOUR DATA
if __name__ == "__main__":
    print("=" * 60)
    print("ANALYSIS OF YOUR SIMULATION RESULTS")
    print("=" * 60)
    
    # Load your actual data
    prices, stability_results, summary_stats = load_simulation_data()
    
    if prices is not None:
        # Create intervention points (for illustration)
        # In your simulation, interventions happen after baseline
        intervention_points = [
            ('Intervention Start', len(prices) - 100000)  # Example
        ]
        
        # Create main analysis plots
        print("\nCreating analysis plots...")
        fig1 = plot_collusion_dynamics(prices, summary_stats, intervention_points)
        
        # Create intervention timeline if we have stability results
        if stability_results:
            print("\nCreating intervention timeline...")
            fig2 = create_intervention_timeline(prices, stability_results, summary_stats)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - collusion_analysis.pdf/png/eps")
        print("  - intervention_timeline.pdf/png")
        print("\nUse these figures in your paper!")
        
    else:
        print("\nGenerating example data for demonstration...")
        # Keep your existing example data generation as fallback
        np.random.seed(42)
        n_periods = 300000
        prices = np.zeros((n_periods, 2))
        
        # Phase 1: Learning (0-100k)
        for t in range(100000):
            learning = 1 - np.exp(-t/20000)
            base = 1.47 + 0.4 * learning
            prices[t, 0] = base + np.random.normal(0, 0.1)
            prices[t, 1] = base + np.random.normal(0, 0.1)
        
        # Phase 2: Collusive (100k-250k)
        prices[100000:250000] = 1.65 + np.random.normal(0, 0.03, (150000, 2))
        
        # Phase 3: Intervention at 250k, then recovery
        prices[250000:250050] = 1.47
        
        for t in range(250050, 300000):
            recovery = 1 - np.exp(-(t-250050)/10000)
            prices[t] = 1.47 + 0.18 * recovery + np.random.normal(0, 0.02)
        
        # Create example summary stats
        example_stats = {
            'nash_price': 1.268,
            'monopoly_price': 2.250,
            'baseline_price': 1.566,
            'collusion_index': 0.303
        }
        
        intervention_points = [('Competitive Pricing', 250000)]
        fig = plot_collusion_dynamics(prices, example_stats, intervention_points)