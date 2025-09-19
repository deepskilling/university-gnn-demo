"""
Quick GCN Demo - Essential Highlights

A concise demonstration of the key GCN concepts in just a few lines.
Perfect for a quick understanding of what's happening.
"""

import numpy as np
from gcn_numpy_implementation import GCN, normalize_adjacency

def quick_gcn_demo():
    """Quick demo showing the essential GCN workflow."""
    
    print("🚀 QUICK GCN DEMO - University Graph")
    print("="*50)
    
    # Load pre-generated university dataset
    print("\n1️⃣ Loading dataset...")
    university_data = np.load('/Users/rchandran/Library/CloudStorage/OneDrive-DiligentCorporation/RESEARCH/GNN/university_dataset.npy', 
                             allow_pickle=True).item()
    
    A = university_data['adjacency_matrix']  # Graph structure
    X = university_data['feature_matrix']    # Node features
    
    print(f"   📊 Graph: {A.shape[0]} nodes, {int(np.sum(A)/2)} edges")
    print(f"   📋 Features: {X.shape[1]} dimensions per node")
    
    # Normalize adjacency matrix
    print("\n2️⃣ Normalizing graph...")
    A_norm = normalize_adjacency(A, add_self_loops=True)
    
    # Create and run GCN
    print("\n3️⃣ Running GCN...")
    gcn = GCN(layer_dims=[4, 16, 8, 3], activation='relu')
    final_embeddings, _ = gcn.forward(X, A_norm, verbose=False)
    
    print(f"   🧠 Final embeddings shape: {final_embeddings.shape}")
    print(f"   📈 Embedding range: [{final_embeddings.min():.3f}, {final_embeddings.max():.3f}]")
    
    # Show sample results
    print("\n4️⃣ Sample results...")
    nodes = university_data['nodes']
    idx_to_node = university_data['idx_to_node']
    
    # Show first few student embeddings
    student_count = 0
    for i in range(min(5, len(final_embeddings))):
        node_id = idx_to_node[i]
        if nodes[node_id]['type'] == 'student':
            student_name = nodes[node_id]['name']
            original_features = X[i]
            learned_embedding = final_embeddings[i]
            
            print(f"   🎓 {student_name}:")
            print(f"      Original: {original_features}")
            print(f"      Learned:  {learned_embedding}")
            
            student_count += 1
            if student_count >= 3:
                break
    
    print(f"\n✅ Demo complete! The GCN learned {final_embeddings.shape[1]}D embeddings")
    print(f"   that capture both node features AND graph structure.")

if __name__ == "__main__":
    quick_gcn_demo()
