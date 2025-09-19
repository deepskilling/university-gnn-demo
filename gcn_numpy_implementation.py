"""
Graph Convolutional Network (GCN) Implementation in Pure NumPy

This module implements a GCN from scratch using only NumPy, with detailed
step-by-step explanations of each operation.

Key Concepts:
1. Graph Convolution: Aggregates information from neighbors
2. Message Passing: How information flows through the graph
3. Node Embeddings: Learned representations of nodes
"""

import numpy as np
from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt

class GCNLayer:
    """A single Graph Convolutional Network layer."""
    
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
        """
        Initialize a GCN layer.
        
        Args:
            input_dim: Dimension of input node features
            output_dim: Dimension of output node features
            use_bias: Whether to use bias term
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        # Initialize weights using Xavier/Glorot initialization
        # This helps with gradient flow during training
        std = np.sqrt(2.0 / (input_dim + output_dim))
        self.W = np.random.normal(0, std, (input_dim, output_dim)).astype(np.float32)
        
        if self.use_bias:
            self.b = np.zeros(output_dim, dtype=np.float32)
        else:
            self.b = None
            
        # Store intermediate values for analysis
        self.last_input = None
        self.last_normalized_adj = None
        self.last_output = None
    
    def forward(self, X: np.ndarray, A_norm: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Forward pass through the GCN layer.
        
        The GCN operation is: H^(l+1) = σ(A_norm @ H^(l) @ W^(l) + b^(l))
        
        Args:
            X: Node feature matrix of shape (N, input_dim)
            A_norm: Normalized adjacency matrix of shape (N, N)
            verbose: Whether to print detailed step-by-step information
            
        Returns:
            Output node features of shape (N, output_dim)
        """
        if verbose:
            print(f"\n🔧 GCN Layer Forward Pass:")
            print(f"   Input shape: {X.shape}")
            print(f"   Adjacency matrix shape: {A_norm.shape}")
            print(f"   Weight matrix shape: {self.W.shape}")
        
        # Store for analysis
        self.last_input = X.copy()
        self.last_normalized_adj = A_norm.copy()
        
        # Step 1: Linear transformation H @ W
        # This is like a regular neural network layer
        XW = X @ self.W  # Shape: (N, output_dim)
        
        if verbose:
            print(f"   After linear transform (X @ W): {XW.shape}")
            print(f"   Sample values: {XW[0, :3]}")
        
        # Step 2: Graph convolution A_norm @ (X @ W)
        # This aggregates information from neighbors
        AXW = A_norm @ XW  # Shape: (N, output_dim)
        
        if verbose:
            print(f"   After graph convolution (A @ X @ W): {AXW.shape}")
            print(f"   Sample values: {AXW[0, :3]}")
        
        # Step 3: Add bias if using
        if self.use_bias:
            output = AXW + self.b
        else:
            output = AXW
            
        self.last_output = output.copy()
        
        if verbose:
            print(f"   Final output shape: {output.shape}")
            print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
            
        return output


class GCN:
    """Multi-layer Graph Convolutional Network."""
    
    def __init__(self, layer_dims: List[int], activation: str = 'relu'):
        """
        Initialize a multi-layer GCN.
        
        Args:
            layer_dims: List of layer dimensions, e.g., [4, 16, 8, 3] for 3 layers
            activation: Activation function ('relu', 'sigmoid', 'tanh', or 'none')
        """
        self.layer_dims = layer_dims
        self.activation = activation
        self.layers = []
        
        # Create GCN layers
        for i in range(len(layer_dims) - 1):
            layer = GCNLayer(layer_dims[i], layer_dims[i + 1])
            self.layers.append(layer)
        
        print(f"🧠 Created GCN with {len(self.layers)} layers: {layer_dims}")
    
    def _apply_activation(self, x: np.ndarray, name: str = None) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            result = np.maximum(0, x)
        elif self.activation == 'sigmoid':
            result = 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        elif self.activation == 'tanh':
            result = np.tanh(x)
        else:  # 'none'
            result = x
            
        if name:
            print(f"   Applied {self.activation} activation to {name}")
            print(f"   Output range: [{result.min():.4f}, {result.max():.4f}]")
            
        return result
    
    def forward(self, X: np.ndarray, A_norm: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through the entire GCN.
        
        Args:
            X: Input node features
            A_norm: Normalized adjacency matrix
            verbose: Whether to print detailed information
            
        Returns:
            Final node embeddings and intermediate representations
        """
        if verbose:
            print(f"\n🚀 GCN Forward Pass (Full Network):")
            print(f"   Input: {X.shape}")
            print(f"   Normalized adjacency: {A_norm.shape}")
        
        h = X.copy()
        intermediate_outputs = []
        
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f"\n--- Layer {i + 1}/{len(self.layers)} ---")
                
            # Forward pass through layer
            h = layer.forward(h, A_norm, verbose=verbose)
            
            # Apply activation (except for the last layer typically)
            if i < len(self.layers) - 1:  # Not the last layer
                h = self._apply_activation(h, f"Layer {i + 1}")
            
            intermediate_outputs.append(h.copy())
            
            if verbose:
                print(f"   Layer {i + 1} output shape: {h.shape}")
        
        if verbose:
            print(f"\n✅ GCN forward pass complete!")
            print(f"   Final embedding shape: {h.shape}")
            
        return h, intermediate_outputs


def normalize_adjacency(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Normalize adjacency matrix for GCN.
    
    The normalization is: A_norm = D^(-1/2) @ A @ D^(-1/2)
    where D is the degree matrix.
    
    Args:
        A: Adjacency matrix
        add_self_loops: Whether to add self-loops (identity matrix)
        
    Returns:
        Normalized adjacency matrix
    """
    print(f"\n🔧 Normalizing Adjacency Matrix:")
    print(f"   Original adjacency shape: {A.shape}")
    print(f"   Number of edges: {np.sum(A) // 2}")  # Divide by 2 for undirected
    
    if add_self_loops:
        # Add self-loops: A = A + I
        A_self = A + np.eye(A.shape[0])
        print(f"   Added self-loops: {np.sum(np.diag(A_self))} self-loops")
    else:
        A_self = A
    
    # Compute degree matrix D
    # Degree of each node = sum of its row/column in adjacency matrix
    degrees = np.sum(A_self, axis=1)  # Shape: (N,)
    print(f"   Node degrees - min: {degrees.min():.1f}, max: {degrees.max():.1f}, mean: {degrees.mean():.1f}")
    
    # Compute D^(-1/2)
    # Add small epsilon to avoid division by zero
    degrees_inv_sqrt = np.power(degrees + 1e-8, -0.5)
    D_inv_sqrt = np.diag(degrees_inv_sqrt)
    
    # Symmetric normalization: D^(-1/2) @ A @ D^(-1/2)
    A_norm = D_inv_sqrt @ A_self @ D_inv_sqrt
    
    print(f"   Normalized adjacency range: [{A_norm.min():.4f}, {A_norm.max():.4f}]")
    print(f"   Row sums (should be ≈1.0): min={np.sum(A_norm, axis=1).min():.3f}, max={np.sum(A_norm, axis=1).max():.3f}")
    
    return A_norm.astype(np.float32)


def analyze_embeddings(embeddings: np.ndarray, node_types: Dict[int, str], 
                      idx_to_node: Dict[int, int], top_k: int = 5) -> None:
    """Analyze the learned node embeddings."""
    print(f"\n📊 Embedding Analysis:")
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    
    # Compute pairwise similarities (cosine similarity)
    # Normalize embeddings to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-8)
    
    # Cosine similarity matrix
    similarity_matrix = embeddings_normalized @ embeddings_normalized.T
    
    print(f"\n🔍 Most Similar Node Pairs:")
    # Find most similar pairs (excluding self-similarity)
    np.fill_diagonal(similarity_matrix, -1)  # Exclude diagonal
    
    # Get top similar pairs
    flat_indices = np.argsort(similarity_matrix.flatten())[-top_k:]
    for idx in reversed(flat_indices):
        i, j = np.unravel_index(idx, similarity_matrix.shape)
        similarity = similarity_matrix[i, j]
        
        node_i_id = idx_to_node[i]
        node_j_id = idx_to_node[j]
        type_i = node_types[node_i_id]
        type_j = node_types[node_j_id]
        
        print(f"   {type_i} {node_i_id} ↔ {type_j} {node_j_id}: similarity = {similarity:.4f}")


def step_by_step_walkthrough(university_data: Dict[str, Any]) -> None:
    """
    Detailed step-by-step walkthrough of GCN computation.
    """
    print("="*80)
    print("🎓 STEP-BY-STEP GCN WALKTHROUGH ON UNIVERSITY GRAPH")
    print("="*80)
    
    # Extract data
    A = university_data['adjacency_matrix']
    X = university_data['feature_matrix']
    node_types = university_data['node_types']
    idx_to_node = university_data['idx_to_node']
    
    print(f"\n📋 Dataset Overview:")
    print(f"   Nodes: {A.shape[0]} ({university_data['num_students']} students, "
          f"{university_data['num_courses']} courses, {university_data['num_professors']} professors)")
    print(f"   Node features: {X.shape[1]} dimensions")
    print(f"   Edges: {np.sum(A) // 2} (undirected)")
    
    # Step 1: Normalize adjacency matrix
    print(f"\n" + "="*60)
    print("STEP 1: ADJACENCY MATRIX NORMALIZATION")
    print("="*60)
    A_norm = normalize_adjacency(A, add_self_loops=True)
    
    # Step 2: Initialize GCN
    print(f"\n" + "="*60)
    print("STEP 2: GCN INITIALIZATION")
    print("="*60)
    
    # Create a 3-layer GCN: 4 -> 16 -> 8 -> 3
    gcn = GCN(layer_dims=[4, 16, 8, 3], activation='relu')
    
    # Step 3: Forward pass with detailed explanation
    print(f"\n" + "="*60)
    print("STEP 3: FORWARD PASS")
    print("="*60)
    
    final_embeddings, intermediate_outputs = gcn.forward(X, A_norm, verbose=True)
    
    # Step 4: Analyze results
    print(f"\n" + "="*60)
    print("STEP 4: RESULTS ANALYSIS")
    print("="*60)
    
    analyze_embeddings(final_embeddings, node_types, idx_to_node)
    
    # Step 5: Visualize embedding space
    print(f"\n" + "="*60)
    print("STEP 5: VISUALIZATION")
    print("="*60)
    
    # Use PCA to reduce to 2D for visualization if embedding dim > 2
    if final_embeddings.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(final_embeddings)
        print(f"   Reduced embeddings to 2D using PCA")
        print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")
    else:
        embeddings_2d = final_embeddings
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Create colors for different node types
    colors = {'student': 'blue', 'course': 'red', 'professor': 'green'}
    
    for i, (matrix_idx, node_id) in enumerate(idx_to_node.items()):
        node_type = node_types[node_id]
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                   c=colors[node_type], alpha=0.6, s=50)
    
    # Create legend
    for node_type, color in colors.items():
        plt.scatter([], [], c=color, alpha=0.6, s=50, label=node_type.capitalize())
    
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title('University Graph Node Embeddings (2D Visualization)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = '/Users/rchandran/Library/CloudStorage/OneDrive-DiligentCorporation/RESEARCH/GNN/embedding_visualization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   Saved visualization to: {plot_path}")
    
    return {
        'final_embeddings': final_embeddings,
        'intermediate_outputs': intermediate_outputs,
        'normalized_adjacency': A_norm,
        'gcn': gcn
    }


if __name__ == "__main__":
    # Load the university dataset
    print("📚 Loading University Dataset...")
    university_data = np.load('/Users/rchandran/Library/CloudStorage/OneDrive-DiligentCorporation/RESEARCH/GNN/university_dataset.npy', allow_pickle=True).item()
    
    # Run the step-by-step walkthrough
    results = step_by_step_walkthrough(university_data)
    
    print(f"\n🎉 GCN Walkthrough Complete!")
    print(f"   Final embedding shape: {results['final_embeddings'].shape}")
    print(f"   Number of intermediate layers: {len(results['intermediate_outputs'])}")
