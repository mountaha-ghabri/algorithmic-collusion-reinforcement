import numpy as np
import time
import pickle  # ADDED FOR SAVING
from environment import BertrandEnvironment
from agent import QLearningAgent

def get_state_index(my_price_idx, other_price_idx, n_prices):
    """Convert price pair to state index (0 to n_prices^2 - 1)"""
    return my_price_idx * n_prices + other_price_idx

def epsilon_decay(t, epsilon_start=1.0, epsilon_end=0.001, decay_rate=5e-6):
    """Exponential epsilon decay satisfying GLIE conditions"""
    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-decay_rate * t)

def run_baseline_simulation(n_periods=500000):
    """
    Run baseline simulation as in Calvano
    Returns: price_history, profit_history, agent1, agent2, env
    """
    print("=" * 60)
    print("BASELINE SIMULATION: Algorithmic Collusion Emergence")
    print("=" * 60)
    
    # Price grid as in Calvano
    p_min, p_max = 1.2, 2.0  # Slightly wider than [p^N, p^M]
    n_prices = 15
    prices = np.linspace(p_min, p_max, n_prices)
    
    # Create environment
    env = BertrandEnvironment(prices)
    print(f"Price grid: {prices}")
    print(f"Nash price: {env.p_nash:.3f}, Monopoly price: {env.p_monopoly:.3f}")
    
    # State and action spaces
    n_states = n_prices * n_prices  # 225 states
    n_actions = n_prices
    
    # Create agents with Calvano parameters
    agent1 = QLearningAgent(n_states, n_actions, alpha=0.15, gamma=0.95)
    agent2 = QLearningAgent(n_states, n_actions, alpha=0.15, gamma=0.95)
    
    # Track history
    price_history = []
    profit_history = []
    
    # Initial state (start at medium prices)
    my_price_idx = n_prices // 2  # Index 7 (price ~1.6)
    other_price_idx = n_prices // 2
    state = get_state_index(my_price_idx, other_price_idx, n_prices)
    
    print("\nTraining progress:")
    start_time = time.time()
    
    for t in range(n_periods):
        # Decaying exploration
        epsilon = epsilon_decay(t)
        
        # Select actions (epsilon-greedy)
        a1 = agent1.select_action(state, epsilon)
        a2 = agent2.select_action(state, epsilon)
        
        # Get profits and next state
        profit1, profit2, (p1, p2) = env.profit(a1, a2)
        next_state = get_state_index(a1, a2, n_prices)
        
        # Update Q-values
        agent1.update(state, a1, profit1, next_state)
        agent2.update(state, a2, profit2, next_state)
        
        # Record history
        price_history.append([p1, p2])
        profit_history.append([profit1, profit2])
        
        # Update state
        state = next_state
        
        # Print progress every 50k periods
        if t % 50000 == 0 and t > 0:
            recent_prices = np.array(price_history[-10000:])
            avg_price = np.mean(recent_prices)
            collusion_index = (avg_price - env.p_nash) / (env.p_monopoly - env.p_nash)
            print(f"  Period {t:,}: Avg price = {avg_price:.3f}, "
                  f"Collusion index = {collusion_index:.3f}, "
                  f"Epsilon = {epsilon:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    
    # Final analysis
    price_array = np.array(price_history)
    profit_array = np.array(profit_history)
    
    # Last 50k periods (stable phase)
    stable_prices = price_array[-50000:]
    stable_profits = profit_array[-50000:]
    
    print("\n" + "=" * 60)
    print("RESULTS - Last 50,000 periods:")
    print("=" * 60)
    print(f"Average price: Firm 1 = {stable_prices[:, 0].mean():.3f}, "
          f"Firm 2 = {stable_prices[:, 1].mean():.3f}")
    print(f"Std deviation: {stable_prices.std():.3f}")
    print(f"Price correlation: {np.corrcoef(stable_prices[:, 0], stable_prices[:, 1])[0,1]:.3f}")
    
    collusion_index = (stable_prices.mean() - env.p_nash) / (env.p_monopoly - env.p_nash)
    print(f"Collusion index: {collusion_index:.3f} (0 = Nash, 1 = Monopoly)")
    
    if collusion_index > 0.3:
        print("✓ Collusion detected!")
    else:
        print("✗ No significant collusion")
    
    return {
        'price_history': price_array,
        'profit_history': profit_array,
        'agent1': agent1,
        'agent2': agent2,
        'env': env,
        'prices': prices,
        'n_prices': n_prices
    }

def test_stability_interventions(results, intervention_type='forced_competitive', duration=50):
    """
    Test collusion stability after various interventions
    
    intervention_type: 
        'forced_competitive' - Force Nash pricing
        'exploration_shock' - High exploration phase
        'asymmetric_learning' - Different learning rates
        'memory_reset' - Reset one agent's Q-table
    """
    print(f"\n" + "=" * 60)
    print(f"STABILITY TEST: {intervention_type.upper()}")
    print("=" * 60)
    
    agent1 = results['agent1']
    agent2 = results['agent2']
    env = results['env']
    n_prices = results['n_prices']
    prices = results['prices']
    
    # Find competitive price index
    nash_idx = np.argmin(np.abs(prices - env.p_nash))
    
    # Start from last state of training
    price_history = results['price_history']
    last_prices = price_history[-1]
    last_price_idx1 = np.argmin(np.abs(prices - last_prices[0]))
    last_price_idx2 = np.argmin(np.abs(prices - last_prices[1]))
    state = get_state_index(last_price_idx1, last_price_idx2, n_prices)
    
    post_intervention_prices = []
    post_intervention_profits = []
    
    # Intervention phase
    print(f"Intervention: {duration} periods of {intervention_type}")
    
    if intervention_type == 'forced_competitive':
        # Force both firms to price at Nash level
        for t in range(duration):
            a1 = nash_idx
            a2 = nash_idx
            profit1, profit2, (p1, p2) = env.profit(a1, a2)
            next_state = get_state_index(a1, a2, n_prices)
            
            # Still update (learning during intervention)
            agent1.update(state, a1, profit1, next_state)
            agent2.update(state, a2, profit2, next_state)
            
            state = next_state
    
    elif intervention_type == 'exploration_shock':
        # Temporarily increase exploration
        for t in range(duration):
            epsilon = 0.5  # High exploration
            
            a1 = agent1.select_action(state, epsilon)
            a2 = agent2.select_action(state, epsilon)
            
            profit1, profit2, (p1, p2) = env.profit(a1, a2)
            next_state = get_state_index(a1, a2, n_prices)
            
            agent1.update(state, a1, profit1, next_state)
            agent2.update(state, a2, profit2, next_state)
            
            state = next_state
    
    elif intervention_type == 'asymmetric_learning':
        # Change learning rates
        original_alpha1 = agent1.alpha
        original_alpha2 = agent2.alpha
        
        agent1.alpha = 0.05   # Slow learner
        agent2.alpha = 0.25   # Fast learner
        
        # Run with new learning rates
        duration = 10000  # Longer for learning rate effect
    
    elif intervention_type == 'memory_reset':
        # Reset one agent's Q-table
        n_states = agent1.n_states
        n_actions = agent1.n_actions
        agent1.Q = np.ones((n_states, n_actions)) * 2.0  # Reset to initial
    
    # Post-intervention learning (observe recovery)
    print(f"Post-intervention learning (100,000 periods)...")
    
    recovery_periods = 100000
    for t in range(recovery_periods):
        epsilon = epsilon_decay(t + 500000)  # Continue decay from where we left off
        
        a1 = agent1.select_action(state, epsilon)
        a2 = agent2.select_action(state, epsilon)
        
        profit1, profit2, (p1, p2) = env.profit(a1, a2)
        next_state = get_state_index(a1, a2, n_prices)
        
        agent1.update(state, a1, profit1, next_state)
        agent2.update(state, a2, profit2, next_state)
        
        post_intervention_prices.append([p1, p2])
        post_intervention_profits.append([profit1, profit2])
        state = next_state
        
        # Print progress
        if t % 20000 == 0 and t > 0:
            recent = np.array(post_intervention_prices[-5000:])
            avg_price = recent.mean()
            print(f"  Recovery period {t:,}: Avg price = {avg_price:.3f}")
    
    # Analyze recovery
    post_prices = np.array(post_intervention_prices)
    post_profits = np.array(post_intervention_profits)
    
    # Last 20k periods of recovery
    stable_post = post_prices[-20000:] if len(post_prices) > 20000 else post_prices
    
    print(f"\nRecovery analysis:")
    print(f"Average price after recovery: {stable_post.mean():.3f}")
    print(f"Collusion index: {(stable_post.mean() - env.p_nash)/(env.p_monopoly - env.p_nash):.3f}")
    
    # Determine if collusion re-formed
    baseline_avg = results['price_history'][-50000:].mean()
    recovery_avg = stable_post.mean()
    
    if abs(recovery_avg - baseline_avg) < 0.05:  # Within 5 cents
        print("✓ Collusion fully recovered")
        recovery_status = "full"
    elif recovery_avg > env.p_nash + 0.1:  # At least 10 cents above Nash
        print("✓ Partial recovery")
        recovery_status = "partial"
    else:
        print("✗ Collusion disrupted permanently")
        recovery_status = "disrupted"
    
    return {
        'post_prices': post_prices,
        'post_profits': post_profits,
        'recovery_status': recovery_status,
        'final_avg_price': stable_post.mean()
    }

def main():
    """Main simulation pipeline"""
    
    # Run baseline to establish collusion
    print("\n" + "=" * 60)
    print("PHASE 1: Establishing Baseline Collusion")
    print("=" * 60)
    baseline_results = run_baseline_simulation(n_periods=300000)
    
    print("\n💾 Saving baseline data...")
    np.save('baseline_prices.npy', baseline_results['price_history'])
    np.save('baseline_profits.npy', baseline_results['profit_history'])
    print(f"✅ Saved baseline data: {len(baseline_results['price_history'])} periods")
    # =======================================================================
    
    # Only continue if collusion emerged
    stable_prices = baseline_results['price_history'][-50000:]
    collusion_index = (stable_prices.mean() - baseline_results['env'].p_nash) / \
                      (baseline_results['env'].p_monopoly - baseline_results['env'].p_nash)
    
    if collusion_index < 0.2:
        print("\nWARNING: Insufficient collusion in baseline. Exiting stability tests.")
        return baseline_results, {}
    
    # Run stability tests
    print("\n" + "=" * 60)
    print("PHASE 2: Stability Tests")
    print("=" * 60)
    
    interventions = [
        ('forced_competitive', 50),
        ('forced_competitive', 100),
        ('exploration_shock', 100),
        ('memory_reset', 0),
    ]
    
    stability_results = {}
    
    for intervention, duration in interventions:
        print(f"\nTesting: {intervention} (duration={duration})")
        result = test_stability_interventions(
            baseline_results, 
            intervention_type=intervention,
            duration=duration
        )
        stability_results[f"{intervention}_{duration}"] = result
        
        # Brief pause between tests
        time.sleep(1)
    
    print("\n💾 Saving stability results...")
    with open('stability_results.pkl', 'wb') as f:
        pickle.dump(stability_results, f)
    print(f"✅ Saved stability results for {len(stability_results)} interventions")
    # ==========================================================================
    
    # Summary
    print("\n" + "=" * 60)
    print("STABILITY TEST SUMMARY")
    print("=" * 60)
    
    baseline_avg = baseline_results['price_history'][-50000:].mean()
    print(f"Baseline collusive price: {baseline_avg:.3f}")
    print(f"Nash price: {baseline_results['env'].p_nash:.3f}")
    print(f"Monopoly price: {baseline_results['env'].p_monopoly:.3f}")
    print()
    
    for test_name, result in stability_results.items():
        status_symbol = "✓" if result['recovery_status'] != "disrupted" else "✗"
        print(f"{status_symbol} {test_name}: "
              f"Final price = {result['final_avg_price']:.3f} "
              f"({result['recovery_status']})")
    
        print("\n💾 Saving summary statistics...")
    summary_stats = {
        'baseline_price': baseline_avg,
        'nash_price': baseline_results['env'].p_nash,
        'monopoly_price': baseline_results['env'].p_monopoly,
        'collusion_index': collusion_index,
        'interventions': {}
    }
    
    for test_name, result in stability_results.items():
        summary_stats['interventions'][test_name] = {
            'final_price': result['final_avg_price'],
            'recovery_status': result['recovery_status'],
            'recovery_percentage': (result['final_avg_price'] - baseline_results['env'].p_nash) / 
                                   (baseline_avg - baseline_results['env'].p_nash) * 100
        }
    
    with open('summary_stats.pkl', 'wb') as f:
        pickle.dump(summary_stats, f)
    print("✅ Saved summary statistics")
    # =======================================================================
    
    return baseline_results, stability_results

if __name__ == "__main__":
    baseline, stability = main()
    print("\n" + "=" * 60)
    print("DATA SAVED SUCCESSFULLY")
    print("=" * 60)
    print("Files created:")
    print("  - baseline_prices.npy        # Price history (NumPy array)")
    print("  - baseline_profits.npy       # Profit history (NumPy array)")
    print("  - stability_results.pkl      # Intervention results (pickle)")
    print("  - summary_stats.pkl          # Summary statistics (pickle)")
    print("\nRun 'python analysis.py' to generate plots!")
    # ==================================================================