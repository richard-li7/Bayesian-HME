import numpy as np
from scipy.special import expit
from scipy.stats import gamma
import matplotlib.pyplot as plt
import math

class LinearExpert:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(output_dim, input_dim + 1) * 0.1  # +1 for bias
        self.tau = 1.0  # precision (1/variance)
    
    def predict_mean(self, X):
        # Add bias column
        X_bias = np.column_stack([np.ones(X.shape[0]), X])
        return X_bias @ self.W.T
    
    def log_likelihood(self, X, y):
        y_pred = self.predict_mean(X)
        return -0.5 * self.tau * np.sum((y - y_pred)**2)
