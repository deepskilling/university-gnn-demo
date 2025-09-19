"""
Educational GNN Demo: Understanding Graph Neural Networks Step by Step

This demo provides an intuitive understanding of how GNNs work by walking through
specific examples with the university graph dataset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from university_graph_dataset import UniversityGraphDataset
from gcn_numpy_implementation import GCN, normalize_adjacency
import matplotlib.pyplot as plt

def explain_graph_structure(university_data):
    """Explain the graph structure with concrete examples."""
    print("="*80)
    print("🔍 UNDERSTANDING THE GRAPH STRUCTURE")
    print("="*80)
    
    A = university_data['adjacency_matrix']
    nodes = university_data['nodes']
    edges = university_data['edges']
    node_mapping = university_data['node_mapping']
    
    print(f"\n📊 Graph Statistics:")
    print(f"   Total nodes: {A.shape[0]}")
    print(f"   Total edges: {len(edges)}")
    
    # Find a student with multiple courses to demonstrate
    student_examples = []
    for edge in edges:
        source, target, edge_type = edge
        if edge_type == 'enrolls':
            source_name = nodes[source].get('name', f'Node_{source}')
            target_name = nodes[target].get('name', f'Node_{target}')
            student_examples.append((source, source_name, target, target_name))
    
    # Group by student
    student_courses = {}
    for source, source_name, target, target_name in student_examples:
        if source not in student_courses:
            student_courses[source] = []
        student_courses[source].append((target, target_name))
    
    # Show example student with their courses
    example_student = list(student_courses.keys())[0]
    student_name = nodes[example_student]['name']
    student_courses_list = student_courses[example_student]
    
    print(f"\n🎓 Example: {student_name}")
    print(f"   Features: {nodes[example_student]['features']}")
    print(f"   Year: {nodes[example_student]['year']}, GPA: {nodes[example_student]['gpa']:.2f}")
    print(f"   Major: {nodes[example_student]['major']}")
    print(f"   Enrolled in {len(student_courses_list)} courses:")
    
    for i, (course_id, course_name) in enumerate(student_courses_list[:5]):  # Show first 5
        print(f"     {i+1}. {course_name} (Features: {nodes[course_id]['features']})")
    
    return example_student, student_courses_list

def demonstrate_message_passing(university_data, example_student, student_courses):
    """Demonstrate how message passing works with a concrete example."""
    print(f"\n" + "="*80)
    print("🔄 UNDERSTANDING MESSAGE PASSING")
    print("="*80)
    
    A = university_data['adjacency_matrix']
    X = university_data['feature_matrix']
    node_mapping = university_data['node_mapping']
    nodes = university_data['nodes']
    
    # Get matrix index for example student
    student_matrix_idx = node_mapping[example_student]
    
    print(f"\n🎯 Focus: How {nodes[example_student]['name']} receives messages")
    print(f"   Matrix index: {student_matrix_idx}")
    print(f"   Original features: {X[student_matrix_idx]}")
    
    # Find neighbors in the adjacency matrix
    neighbors = np.where(A[student_matrix_idx] > 0)[0]
    print(f"   Number of neighbors: {len(neighbors)}")
    
    print(f"\n📨 Messages from neighbors:")
    for i, neighbor_idx in enumerate(neighbors[:5]):  # Show first 5 neighbors
        neighbor_original_id = university_data['idx_to_node'][neighbor_idx]
        neighbor_node = nodes[neighbor_original_id]
        neighbor_features = X[neighbor_idx]
        
        print(f"   {i+1}. {neighbor_node.get('name', f'Node_{neighbor_original_id}')} ({neighbor_node['type']})")
        print(f"      Features: {neighbor_features}")
        print(f"      Connection strength: {A[student_matrix_idx, neighbor_idx]:.3f}")
    
    # Show what happens after normalization
    A_norm = normalize_adjacency(A, add_self_loops=True)
    print(f"\n🔧 After normalization:")
    print(f"   Sum of connection weights: {np.sum(A_norm[student_matrix_idx]):.3f}")
    print(f"   Self-loop weight: {A_norm[student_matrix_idx, student_matrix_idx]:.3f}")
    
    return student_matrix_idx, neighbors

def demonstrate_layer_transformations(university_data, student_matrix_idx):
    """Show how features transform through GCN layers."""
    print(f"\n" + "="*80)
    print("🧠 LAYER-BY-LAYER TRANSFORMATIONS")
    print("="*80)
    
    A = university_data['adjacency_matrix']
    X = university_data['feature_matrix']
    
    # Normalize adjacency matrix
    A_norm = normalize_adjacency(A, add_self_loops=True)
    
    # Create a simple 2-layer GCN for clearer demonstration
    gcn = GCN(layer_dims=[4, 8, 3], activation='relu')
    
    print(f"\n📊 Original features for student at index {student_matrix_idx}:")
    print(f"   {X[student_matrix_idx]}")
    print(f"   Meaning: [year={X[student_matrix_idx,0]}, gpa={X[student_matrix_idx,1]:.2f}, "
          f"major_encoded={int(X[student_matrix_idx,2])}, is_senior={X[student_matrix_idx,3]}]")
    
    # Manual forward pass with detailed output
    h = X.copy()
    
    for i, layer in enumerate(gcn.layers):
        print(f"\n--- Layer {i+1} ---")
        print(f"Input to layer {i+1}: {h[student_matrix_idx]}")
        
        # Linear transformation
        h_linear = h @ layer.W + (layer.b if layer.use_bias else 0)
        print(f"After linear transform: {h_linear[student_matrix_idx]}")
        
        # Graph convolution (message passing)
        h_conv = A_norm @ h_linear
        print(f"After graph convolution: {h_conv[student_matrix_idx]}")
        
        # Activation
        if i < len(gcn.layers) - 1:  # Not last layer
            h_conv = np.maximum(0, h_conv)  # ReLU
            print(f"After ReLU activation: {h_conv[student_matrix_idx]}")
        
        h = h_conv
        print(f"Final output from layer {i+1}: {h[student_matrix_idx]}")
    
    print(f"\n🎯 Final embedding for our student: {h[student_matrix_idx]}")
    
    return h

def analyze_learned_representations(university_data, final_embeddings):
    """Analyze what the GCN has learned."""
    print(f"\n" + "="*80)
    print("🔍 ANALYZING LEARNED REPRESENTATIONS")
    print("="*80)
    
    nodes = university_data['nodes']
    idx_to_node = university_data['idx_to_node']
    node_types = university_data['node_types']
    
    # Group nodes by type
    type_embeddings = {'student': [], 'course': [], 'professor': []}
    type_indices = {'student': [], 'course': [], 'professor': []}
    
    for matrix_idx in range(len(final_embeddings)):
        node_id = idx_to_node[matrix_idx]
        node_type = node_types[node_id]
        type_embeddings[node_type].append(final_embeddings[matrix_idx])
        type_indices[node_type].append(matrix_idx)
    
    # Compute average embeddings for each type
    print(f"\n📊 Average embeddings by node type:")
    for node_type in ['student', 'course', 'professor']:
        if type_embeddings[node_type]:
            avg_embedding = np.mean(type_embeddings[node_type], axis=0)
            print(f"   {node_type.capitalize()}: {avg_embedding}")
            print(f"   Count: {len(type_embeddings[node_type])}")
    
    # Find most similar cross-type pairs
    print(f"\n🔗 Most similar student-course pairs:")
    student_indices = type_indices['student'][:10]  # First 10 students
    course_indices = type_indices['course'][:5]    # First 5 courses
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(final_embeddings, axis=1, keepdims=True)
    embeddings_normalized = final_embeddings / (norms + 1e-8)
    
    max_similarity = -1
    best_pairs = []
    
    for s_idx in student_indices:
        for c_idx in course_indices:
            similarity = np.dot(embeddings_normalized[s_idx], embeddings_normalized[c_idx])
            best_pairs.append((similarity, s_idx, c_idx))
    
    # Sort and show top 3
    best_pairs.sort(reverse=True)
    for i, (similarity, s_idx, c_idx) in enumerate(best_pairs[:3]):
        student_id = idx_to_node[s_idx]
        course_id = idx_to_node[c_idx]
        student_name = nodes[student_id]['name']
        course_name = nodes[course_id]['name']
        
        print(f"   {i+1}. {student_name} ↔ {course_name}: {similarity:.4f}")

def create_simple_visualization(final_embeddings, university_data):
    """Create a simple 2D visualization of embeddings."""
    from sklearn.decomposition import PCA
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(final_embeddings)
    
    plt.figure(figsize=(10, 8))
    
    node_types = university_data['node_types']
    idx_to_node = university_data['idx_to_node']
    colors = {'student': 'blue', 'course': 'red', 'professor': 'green'}
    
    for matrix_idx in range(len(embeddings_2d)):
        node_id = idx_to_node[matrix_idx]
        node_type = node_types[node_id]
        plt.scatter(embeddings_2d[matrix_idx, 0], embeddings_2d[matrix_idx, 1], 
                   c=colors[node_type], alpha=0.6, s=30)
    
    # Create legend
    for node_type, color in colors.items():
        plt.scatter([], [], c=color, alpha=0.6, s=50, label=f'{node_type.capitalize()}s')
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('University Graph: Node Embeddings in 2D Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = '/Users/rchandran/Library/CloudStorage/OneDrive-DiligentCorporation/RESEARCH/GNN/simple_embedding_viz.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   Saved simple visualization to: {plot_path}")

def educational_walkthrough():
    """Complete educational walkthrough."""
    print("🎓 WELCOME TO THE GNN EDUCATIONAL DEMO!")
    print("   We'll walk through a Graph Neural Network step by step")
    print("   using a realistic university graph with students, courses, and professors.")
    
    # Load data
    print("\n📚 Loading university dataset...")
    university_data = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'university_dataset.npy'), allow_pickle=True).item()
    
    # Step 1: Understand the graph
    example_student, student_courses = explain_graph_structure(university_data)
    
    # Step 2: Message passing
    student_idx, neighbors = demonstrate_message_passing(university_data, example_student, student_courses)
    
    # Step 3: Layer transformations
    final_embeddings = demonstrate_layer_transformations(university_data, student_idx)
    
    # Step 4: Analyze representations
    analyze_learned_representations(university_data, final_embeddings)
    
    # Step 5: Visualize
    print(f"\n" + "="*80)
    print("🎨 CREATING VISUALIZATION")
    print("="*80)
    create_simple_visualization(final_embeddings, university_data)
    
    print(f"\n🎉 Educational walkthrough complete!")
    print(f"   Key takeaways:")
    print(f"   1. GNNs aggregate information from neighbors")
    print(f"   2. Multiple layers allow long-range information propagation")
    print(f"   3. Learned embeddings capture both node features and graph structure")
    print(f"   4. Similar nodes (by structure/features) end up with similar embeddings")

if __name__ == "__main__":
    educational_walkthrough()
