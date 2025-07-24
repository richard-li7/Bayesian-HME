# variational/hme_model.py
import numpy as np
from scipy.special import expit

class TreeNode:
    """Base class for tree nodes"""
    def __init__(self):
        pass

class GatingNode(TreeNode):
    """Internal node that makes binary decisions"""
    def __init__(self, node_id, left_child=None, right_child=None):
        super().__init__()
        self.node_id = node_id
        self.left_child = left_child    # Go here if z_i = 1 
        self.right_child = right_child  # Go here if z_i = 0
        
    def is_leaf(self):
        return False

class ExpertNode(TreeNode):
    """Leaf node that makes predictions"""
    def __init__(self, expert_id):
        super().__init__()
        self.expert_id = expert_id
        
    def is_leaf(self):
        return True

class HMEModel:
    """Hierarchical Mixture of Experts with proper binary tree structure"""
    
    def __init__(self, root_node):
        """
        Args:
            root_node: TreeNode that is the root of the HME tree
        """
        self.root = root_node
        self.experts = self._collect_experts()
        self.gating_nodes = self._collect_gating_nodes()
        self.num_experts = len(self.experts)
        self.num_gating_nodes = len(self.gating_nodes)
    
    def _collect_experts(self):
        """Collect all expert nodes via tree traversal"""
        experts = []
        
        def traverse(node):
            if node is None:
                return
            if node.is_leaf():
                experts.append(node.expert_id)
            else:
                traverse(node.left_child)
                traverse(node.right_child)
        
        traverse(self.root)
        return sorted(experts)
    
    def _collect_gating_nodes(self):
        """Collect all gating node IDs via tree traversal"""
        gating_nodes = []
        
        def traverse(node):
            if node is None:
                return
            if not node.is_leaf():
                gating_nodes.append(node.node_id)
                traverse(node.left_child)
                traverse(node.right_child)
        
        traverse(self.root)
        return sorted(gating_nodes)
    
    def compute_mixing_coefficients(self, x, v_means):
        """
        Compute pi_j(x) for each expert j by traversing tree paths
        
    
            Parameter x: input vector (shape: input_dim,)
            Parameter v_means: dict mapping gating_node_id -> weight vector
        
        Returns dict mapping expert_id -> pi_j(x) (mixing coefficients)
        """
        x = np.asarray(x)
        mixing_coeffs = {}
        
        def traverse_to_experts(node, current_prob):
            """Recursively compute path probabilities"""
            if node is None:
                return
                
            if node.is_leaf():
                # We've reached an expert - store the accumulated probability
                mixing_coeffs[node.expert_id] = current_prob
            else:
                # This is a gating node - compute left/right probabilities
                v_i = v_means[node.node_id]
                gate_activation = np.dot(v_i, x)
                gate_prob = expit(gate_activation)  # σ(v_i^T x)
                
                # Traverse left (z_i = 1) and right (z_i = 0)
                traverse_to_experts(node.left_child, current_prob * gate_prob)
                traverse_to_experts(node.right_child, current_prob * (1 - gate_prob))
        
        # Start traversal from root with probability 1.0
        traverse_to_experts(self.root, 1.0)
        
        return mixing_coeffs
    
    def expert_prediction(self, x, W_j):
        """Compute prediction for expert j: y_j(x) = W_j x"""
        return np.dot(W_j, x)
    
    def forward_pass(self, x, v_means, W_means):
        """
        Complete forward pass through the HME
        
            Parameter x: input vector
            Parameter v_means: dict mapping gating_node_id -> weight vector  
            Parameter W_means: dict mapping expert_id -> weight matrix
            
        Returns
            prediction: final mixture prediction
            mixing_coeffs: dict of mixing coefficients
        """
        # Get mixing coefficients
        mixing_coeffs = self.compute_mixing_coefficients(x, v_means)
        
        # Compute weighted mixture of expert predictions
        prediction = 0
        for expert_id, coeff in mixing_coeffs.items():
            expert_pred = self.expert_prediction(x, W_means[expert_id])
            prediction += coeff * expert_pred
        
        return prediction, mixing_coeffs
    
    def get_path_to_expert(self, target_expert_id):
        """
        Get the path from root to a specific expert
        
        Returns path: list of (gating_node_id, direction) tuples
        """
        path = []
        
        def find_path(node, current_path):
            if node is None:
                return False
                
            if node.is_leaf():
                if node.expert_id == target_expert_id:
                    path.extend(current_path)
                    return True
                return False
            else:
                # Try left branch
                if find_path(node.left_child, current_path + [(node.node_id, 'left')]):
                    return True
                # Try right branch  
                if find_path(node.right_child, current_path + [(node.node_id, 'right')]):
                    return True
                return False
        
        find_path(self.root, [])
        return path
    
    def print_tree(self):
        """Print tree structure for debugging"""
        def print_node(node, level=0):
            indent = "  " * level
            if node is None:
                return
            if node.is_leaf():
                print(f"{indent}Expert {node.expert_id}")
            else:
                print(f"{indent}Gate {node.node_id}")
                print(f"{indent}├─ Left:")
                print_node(node.left_child, level + 1)
                print(f"{indent}└─ Right:")
                print_node(node.right_child, level + 1)
        
        print_node(self.root)

# Helper functions to build common tree structures
def build_figure1_tree():
    """Build the tree from Figure 1 in the paper"""
    expert_0 = ExpertNode(0)
    expert_1 = ExpertNode(1) 
    expert_2 = ExpertNode(2)
    
    gating_1 = GatingNode(1, left_child=expert_1, right_child=expert_2)
    gating_0 = GatingNode(0, left_child=expert_0, right_child=gating_1)
    
    return gating_0

def build_balanced_binary_tree(depth):
    """Build a balanced binary tree of given depth"""
    expert_counter = [0]  # Use list for mutable counter
    gating_counter = [0]
    
    def build_subtree(current_depth):
        if current_depth == depth:
            # Create leaf (expert)
            expert_id = expert_counter[0]
            expert_counter[0] += 1
            return ExpertNode(expert_id)
        else:
            # Create internal node (gating)
            gating_id = gating_counter[0]
            gating_counter[0] += 1
            
            left_child = build_subtree(current_depth + 1)
            right_child = build_subtree(current_depth + 1)
            
            return GatingNode(gating_id, left_child, right_child)
    
    return build_subtree(0)

# Example usage:
if __name__ == "__main__":
    # Build tree from Figure 1
    root = build_figure1_tree()
    model = HMEModel(root)
    
    print("Tree structure:")
    model.print_tree()
    
    print(f"\nExperts: {model.experts}")
    print(f"Gating nodes: {model.gating_nodes}")
    
    # Show path to each expert
    for expert_id in model.experts:
        path = model.get_path_to_expert(expert_id)
        print(f"Path to expert {expert_id}: {path}")