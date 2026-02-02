import numpy as np

class QLearningAgent:
    """
    Independent Q-learning agent with Calvano's initialization
    Follows Watkins & Dayan (1992) with epsilon-greedy exploration
    """
    
    def __init__(self, n_states, n_actions, alpha=0.15, gamma=0.95):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha      # Learning rate (Calvano: 0.15)
        self.gamma = gamma      # Discount factor (Calvano: 0.95)
        
        # Initialize Q-values as in Calvano: expected discounted sum 
        # of profits when opponent randomizes uniformly
        self.Q = self.initialize_q_values()
        
    def initialize_q_values(self):
        """Initialize Q-values following Calvano's approach"""
        # They use: Q_0(s,a) = E[π(s,a,a_-i)] / (1-δ|A|^(n-1))
        # For simplicity, we initialize to small positive values
        # This encourages exploration while being realistic
        return np.ones((self.n_states, self.n_actions)) * 2.0
    
    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Break ties randomly
            max_q = np.max(self.Q[state, :])
            max_indices = np.where(self.Q[state, :] == max_q)[0]
            return np.random.choice(max_indices)
    
    def update(self, state, action, reward, next_state):
        """Standard Q-learning update"""
        best_next_q = np.max(self.Q[next_state, :])
        target = reward + self.gamma * best_next_q
        self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * target
    
    def get_greedy_action(self, state):
        """Get greedy action for given state (no exploration)"""
        return np.argmax(self.Q[state, :])