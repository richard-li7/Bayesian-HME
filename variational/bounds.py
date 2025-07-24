import numpy as np
from scipy.special import expit

class SigmoidBound:
    """Implements the local convex bound from Eq. (7)"""
    
    def __init__(self, eps=1e-8):
        self.xi_params = {}  # Variational parameters  
        self.eps = eps
    
    def gating_bound(self, zi, vi, x, xi):
        """p(z_i | v_i, x) ≥ exp(z_i v_i^T x) F(-v_i^T x, \xi_i)
        
        Parameter zi: zi \in {0,1} where 0 signifies going left, and 1 signifies going right in the expert tree

        Parameter vi: Vector of weight associated with gate i

        Parameter x: Input

        Parameter xi: Variational parameter as an input to F, the sigmoid bound \in R
        """
        vi = np.asarray(vi)
        x = np.asarray(x)
        
        activation = np.dot(vi, x)  # v_i^T x
        exp_term = np.exp(zi * activation)
        bound_term = self.sigmoid_bound(-activation, xi)  # F(-v_i^T x, \xi_i)
        
        return exp_term * bound_term
    
    def sigmoid_bound(self, x, xi):
        """F(x,\xi) = \sigma(\xi)exp{(x-\xi)/2 - \lambda(\xi)(x^2-\xi^2)}
        
        Parameter x: Input

        Parameter xi: Variational parameter \in R
        """
        x = np.asarray(x)
        xi = np.asarray(xi)
        
        sigma_xi = expit(xi)
        lambda_xi = self.lambda_function(xi)
        
        exponent = (x - xi)/2.0 - lambda_xi * (x**2 - xi**2)
        exponent = np.clip(exponent, -700, 700)  # Numerical stability
        
        return sigma_xi * np.exp(exponent)
    
    def lambda_function(self, xi):
        """\lambda(\xi) = tanh(\xi/2)/(4\xi)
        
        Parameter xi: Variational parameter \in R
        """
        xi = np.asarray(xi)
        xi_abs = np.abs(xi)
        
        # Handle xi ≈ 0 using limit: lim_{\xi→0} tanh(\xi/2)/(4\xi) = 1/8
        result = np.where(xi_abs < self.eps,
                         1.0/8.0,
                         np.tanh(xi/2.0) / (4.0 * xi))
        
        return result
    
    def update_xi(self, data):
        """Optimize \xi parameters for tightest bound"""
        pass
