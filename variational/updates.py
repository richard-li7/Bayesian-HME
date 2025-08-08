import numpy as np
from scipy.special import digamma, gammaln, expit
from scipy.linalg import inv, det
import logging

class VariationalUpdates:
    """
    Implements the variational update equations from Bishop & Svensén (2003)
    Section 3.2: Factorized Distributions
    """
    
    def __init__(self, hme_model, bounds, var_dists, max_iterations=100, tol=1e-6):
        self.hme_model = hme_model
        self.bounds = bounds
        self.var_dists = var_dists
        self.max_iterations = max_iterations
        self.tol = tol
        self.logger = logging.getLogger(__name__)

        #Cache to save avoid recomputation
        self.zeta_expectations = None
        
    def variational_em_step(self, X, T):
        """
        Perform one complete variational EM step
        
        Args:
            X: Input data (N, input_dim)
            T: Target data (N, target_dim)
            
        Returns:
            lower_bound: Current value of variational lower bound
        """
        N, input_dim = X.shape
        N_t, target_dim = T.shape
        
        # Augment inputs with bias term (last column = 1)
        X_aug = np.column_stack([X, np.ones(N)])
        
        # Initialize xi parameters if not present
        if not hasattr(self, 'xi_params'):
            self.xi_params = np.random.normal(0, 1, (N, self.hme_model.num_gating_nodes))
        
        converged = False
        iteration = 0
        prev_bound = -np.inf
        
        while not converged and iteration < self.max_iterations:
            # Update each variational factor
            self.update_q_Z(X_aug, T)
            self.update_q_W(X_aug, T)
            self.update_q_v(X_aug, T)
            self.update_q_alpha()
            self.update_q_beta()  
            self.update_q_tau(X_aug, T)
            self.update_xi_parameters(X_aug)
            
            # Compute lower bound
            current_bound = self.compute_lower_bound(X_aug, T)
            
            # Check convergence
            if abs(current_bound - prev_bound) < self.tol:
                converged = True
                
            self.logger.debug(f"Iteration {iteration}: Lower bound = {current_bound}")
            prev_bound = current_bound
            iteration += 1
            
        return current_bound
    
    def update_q_Z(self, X_aug, T):
        """
        Update q*(Z) - posterior over latent gating variables
        """
        pass
    
    def _compute_gating_log_prob(self, z_i, v_i, x, xi_ni, gating_idx):
        """Compute log probability contribution from gating node using variational bound"""
        activation = np.dot(v_i, x)
        
        # From the bound: p(z_i|v_i,x) ≥ exp(z_i * v_i^T x) * F(-v_i^T x, xi)
        linear_term = z_i * activation
        
        # Compute F(-activation, xi_ni) bound term
        bound_term = self.bounds.sigmoid_bound(-activation, xi_ni)
        
        return linear_term + np.log(bound_term + 1e-8)  # Add small epsilon for numerical stability
    
    def _compute_gating_log_prob(self, z_i, v_i, x, xi_ni, gating_idx):
        """Compute log probability contribution from gating node using variational bound"""
        activation = np.dot(v_i, x)
        
        # From the bound: p(z_i|v_i,x) ≥ exp(z_i * v_i^T x) * F(-v_i^T x, xi)
        linear_term = z_i * activation
        
        # Compute F(-activation, xi_ni) bound term
        bound_term = self.bounds.sigmoid_bound(-activation, xi_ni)
        
        return linear_term + np.log(bound_term + 1e-8)  # Add small epsilon for numerical stability
    
    def _compute_zeta_expectations(self, X_aug):
        """
        Compute ⟨ζ_jn⟩ for all experts j and data points n
        
        ζ_jn represents the probability that expert j is selected for data point n
        From equation (3): ζ_j = ∏_i z̃_i where z̃_i depends on the path to expert j
        
        Returns:
            zeta_exp: (num_experts, N) array of ⟨ζ_jn⟩ values
        """
        N = X_aug.shape[0]
        num_experts = self.hme_model.num_experts
        
        # Get current expectations of v parameters
        expectations = self.var_dists.get_expectations()
        v_means = {}
        for i, gating_id in enumerate(self.hme_model.gating_nodes):
            v_means[gating_id] = expectations['v'][i]
        
        zeta_exp = np.zeros((num_experts, N))
        
        # For each data point
        for n in range(N):
            x_n = X_aug[n]
            
            # For each expert, compute the path probability
            for j, expert_id in enumerate(self.hme_model.experts):
                # Get path from root to this expert
                path = self.hme_model.get_path_to_expert(expert_id)
                
                # Compute product over all gating nodes on path
                path_prob = 1.0
                for gating_id, direction in path:
                    v_i = v_means[gating_id]
                    
                    # Compute gating probability: σ(v_i^T x_n)
                    gate_activation = np.dot(v_i, x_n)
                    gate_prob = expit(gate_activation)
                    
                    # Apply direction: left (z_i=1) uses gate_prob, right (z_i=0) uses 1-gate_prob
                    if direction == 'left':
                        path_prob *= gate_prob
                    else:  # direction == 'right'
                        path_prob *= (1.0 - gate_prob)
                
                zeta_exp[j, n] = path_prob
        
        return zeta_exp
    
    def update_q_W(self, X_aug, T):
        """
        Update q*(W_j) - posterior over expert weight matrices
        Result: q_W*(W_j) = N(W_j | μ_W_j, Σ_W_j)
        """
        N, input_dim_aug = X_aug.shape
        target_dim = T.shape[1]
        
        # Get current expectations
        expectations = self.var_dists.get_expectations()
        alpha_exp = expectations['alpha']  # ⟨α_j⟩
        tau_exp = expectations['tau']      # ⟨τ_j⟩
        
        # Compute ζ expectations
        zeta_exp = self._compute_zeta_expectations(X_aug)  # (num_experts, N)
        
        # Update each expert's weight distribution
        for j, expert_id in enumerate(self.hme_model.experts):
            # Compute precision matrix: Σ_j^{-1} = ⟨α_j⟩I + ⟨τ_j⟩∑_n ⟨ζ_jn⟩x_n x_n^T
            precision_matrix = alpha_exp[j] * np.eye(input_dim_aug)
            
            # Add weighted outer products of inputs
            for n in range(N):
                x_n = X_aug[n:n+1].T  # (input_dim_aug, 1)
                zeta_jn = zeta_exp[j, n]
                precision_matrix += tau_exp[j] * zeta_jn * np.outer(x_n.flatten(), x_n.flatten())
            
            # Compute covariance matrix: Σ_j = (Σ_j^{-1})^{-1}
            try:
                covariance_matrix = inv(precision_matrix)
            except np.linalg.LinAlgError:
                # Handle singular matrix by adding small regularization
                self.logger.warning(f"Singular precision matrix for expert {j}, adding regularization")
                precision_matrix += 1e-6 * np.eye(input_dim_aug)
                covariance_matrix = inv(precision_matrix)
            
            # Compute mean: μ_j = Σ_j ⟨τ_j⟩∑_n ⟨ζ_jn⟩x_n t_n^T
            # Note: This gives us a (input_dim_aug, target_dim) matrix
            weighted_sum = np.zeros((input_dim_aug, target_dim))
            
            for n in range(N):
                x_n = X_aug[n:n+1].T  # (input_dim_aug, 1)
                t_n = T[n:n+1].T      # (target_dim, 1)
                zeta_jn = zeta_exp[j, n]
                
                # Add weighted contribution: x_n t_n^T
                weighted_sum += zeta_jn * np.outer(x_n.flatten(), t_n.flatten())
            
            mean_matrix = tau_exp[j] * np.dot(covariance_matrix, weighted_sum)
            
            # Update variational distribution
            # Note: We store the precision for efficiency, not covariance
            self.var_dists.update_q_W(j, mean_matrix, precision_matrix)
            
            self.logger.debug(f"Updated q_W for expert {j}: "
                            f"mean shape {mean_matrix.shape}, "
                            f"precision condition number {np.linalg.cond(precision_matrix):.2e}")
    
            
    def update_q_v(self, X_aug, T):
        """
        Update q*(v_i) - posterior over gating weight vectors
        Result: q_v*(v_i) = N(v_i | μ_v_i, Σ_v_i)
        """
        pass
        
    
    def update_q_alpha(self):
        """
        Update q*(alpha_j) - posterior over expert precision hyperparameters
        """
        pass
    
    def update_q_beta(self):
        """
        Update q*(beta_i) - posterior over gating precision hyperparameters
        """
        pass
    
    def update_q_tau(self, X_aug, T):
        """
        Update q*(tau_j) - posterior over noise precisions
        """
        pass
    
    def update_xi_parameters(self, X_aug):
        """
        Update variational parameters ξ for sigmoid bounds
        """
        pass
    
    def compute_lower_bound(self, X_aug, T):
        """
        Compute the variational lower bound L(q)
        This is equation (5) from the paper
        """
        
        pass