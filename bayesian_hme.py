# variational/hme_model.py
import numpy as np
from scipy.special import expit

class HMEModel:
    """Hierarchical Mixture of Experts model"""
    
    def __init__(self, expert_paths):
        """
        expert_paths: dict mapping expert_id -> list of (gating_node, direction) tuples
        
        Example for Figure 1 tree:
        expert_paths = {
            0: [(0, 'left')],                    # expert 0: left at gating node 0
            1: [(0, 'right'), (1, 'left')],     # expert 1: right at 0, left at 1  
            2: [(0, 'right'), (1, 'right')]     # expert 2: right at 0, right at 1
        }
        """
        self.expert_paths = expert_paths
        self.num_experts = len(expert_paths)
        
        # Extract all gating nodes used
        all_gating_nodes = set()
        for path in expert_paths.values():
            for gating_node, _ in path:
                all_gating_nodes.add(gating_node)
        
        self.gating_nodes = sorted(list(all_gating_nodes))
        self.num_gating_nodes = len(self.gating_nodes)
    
    def compute_mixing_coefficients(self, x, v_means):
        """
        Compute π_j(x) for each expert j
        
        From paper: π_j(x) = ∏_{i on path to j} [σ(v_i^T x) if left, 1-σ(v_i^T x) if right]
        
        Args:
            x: input vector (shape: input_dim,)
            v_means: dict mapping gating_node -> weight vector (from variational dist)
        
        Returns:
            mixing_coeffs: array of shape (num_experts,) with π_j(x) values
        """
        x = np.asarray(x)
        mixing_coeffs = np.zeros(self.num_experts)
        
        # For each expert, compute product of gating probabilities along its path
        for expert_j, path in self.expert_paths.items():
            path_prob = 1.0
            
            for gating_node_i, direction in path:
                # Get gating weight vector for this node
                v_i = v_means[gating_node_i]  # shape: (input_dim,)
                
                # Compute gating probability σ(v_i^T x)
                gate_activation = np.dot(v_i, x)
                gate_prob = expit(gate_activation)  # σ(v_i^T x)
                
                # Multiply by appropriate probability based on direction
                if direction == 'left':
                    path_prob *= gate_prob        # z_i = 1 case
                else:  # direction == 'right' 
                    path_prob *= (1 - gate_prob)  # z_i = 0 case
            
            mixing_coeffs[expert_j] = path_prob
            
        return mixing_coeffs
    
    def expert_prediction(self, x, W_j):
        """
        Compute mean prediction for expert j: y_j(x) = W_j x
        
        Args:
            x: input vector (shape: input_dim,)
            W_j: weight matrix for expert j (shape: target_dim, input_dim)
        
        Returns:
            prediction: vector of shape (target_dim,)
        """
        x = np.asarray(x)
        W_j = np.asarray(W_j)
        
        return np.dot(W_j, x)  # Matrix-vector multiply
    
    def forward_pass(self, x, v_means, W_means):
        """
        Complete forward pass: compute mixture prediction
        
        Args:
            x: input vector  
            v_means: dict mapping gating_node -> weight vector
            W_means: dict mapping expert -> weight matrix
            
        Returns:
            prediction: weighted average of expert predictions
            mixing_coeffs: individual mixing coefficients
        """
        # Get mixing coefficients
        mixing_coeffs = self.compute_mixing_coefficients(x, v_means)
        
        # Get expert predictions  
        expert_preds = []
        for expert_j in range(self.num_experts):
            pred_j = self.expert_prediction(x, W_means[expert_j])
            expert_preds.append(pred_j)
        
        expert_preds = np.array(expert_preds)  # shape: (num_experts, target_dim)
        
        # Weighted mixture
        prediction = np.sum(mixing_coeffs[:, None] * expert_preds, axis=0)
        
        return prediction, mixing_coeffs

# Helper function to create common tree structures
def create_binary_tree(depth):
    """
    Create a balanced binary tree of given depth
    
    Args:
        depth: tree depth (depth=1 means 2 experts, depth=2 means 4 experts, etc.)
    
    Returns:
        expert_paths: dict suitable for HMEModel
    """
    num_experts = 2 ** depth
    expert_paths = {}
    
    for expert_id in range(num_experts):
        path = []
        
        # Convert expert_id to binary path
        binary = format(expert_id, f'0{depth}b')
        
        for level, bit in enumerate(binary):
            gating_node = level
            direction = 'left' if bit == '0' else 'right'
            path.append((gating_node, direction))
            
        expert_paths[expert_id] = path
    
    return expert_paths

# Example usage:
if __name__ == "__main__":
    # Create the tree from Figure 1 in the paper
    expert_paths = {
        0: [(0, 'left')],                    
        1: [(0, 'right'), (1, 'left')],     
        2: [(0, 'right'), (1, 'right')]     
    }
    
    model = HMEModel(expert_paths)
    print(f"Number of experts: {model.num_experts}")
    print(f"Number of gating nodes: {model.num_gating_nodes}")
    print(f"Gating nodes: {model.gating_nodes}")
    
    # Or create a balanced binary tree
    balanced_paths = create_binary_tree(depth=2)  # 4 experts
    balanced_model = HMEModel(balanced_paths)