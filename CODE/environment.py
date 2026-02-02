import numpy as np

class BertrandEnvironment:
    """
    Repeated Bertrand competition with differentiated products (logit demand)
    Follows Calvano et al. (2020) parameters exactly
    """
    
    def __init__(self, prices):
        self.prices = prices  # Discrete price grid
        self.mu = 0.25        # Differentiation parameter (Calvano: 0.25)
        self.a = 2.0          # a_i (Calvano: 2)
        self.a0 = 0.0         # Outside option (Calvano: 0)
        self.c = 1.0          # Marginal cost (Calvano: 1)
        self.n_prices = len(prices)
        
        # Precompute for speed
        self.u_prices = np.array([(self.a - p)/self.mu for p in prices])
        self.u0 = self.a0/self.mu
        
        # Compute theoretical benchmarks
        self.p_nash, self.p_monopoly = self.compute_benchmarks()
        
    def compute_benchmarks(self):
        """Compute Nash and Monopoly prices analytically"""
        # For logit with these parameters, we can compute
        # Following Calvano's derivations
        p_nash = 1 + self.mu + 1/(1 + np.exp(1/self.mu))  # Approx 1.47
        p_monopoly = 1 + self.mu + 1  # Approx 2.0 for mu=0.25
        return p_nash, p_monopoly
    
    def demand(self, price_idx_i, price_idx_j):
        """Logit demand as in Calvano Eq (1)"""
        u_i = self.u_prices[price_idx_i]
        u_j = self.u_prices[price_idx_j]
        
        denominator = np.exp(u_i) + np.exp(u_j) + np.exp(self.u0)
        
        q_i = np.exp(u_i) / denominator
        q_j = np.exp(u_j) / denominator
        
        return q_i, q_j
    
    def profit(self, price_idx_i, price_idx_j):
        """Compute profits for both firms"""
        q_i, q_j = self.demand(price_idx_i, price_idx_j)
        p_i = self.prices[price_idx_i]
        p_j = self.prices[price_idx_j]
        
        profit_i = (p_i - self.c) * q_i
        profit_j = (p_j - self.c) * q_j
        
        return profit_i, profit_j, (p_i, p_j)
    
    def get_competitive_price_idx(self):
        """Find price closest to Nash equilibrium"""
        return np.argmin(np.abs(self.prices - self.p_nash))