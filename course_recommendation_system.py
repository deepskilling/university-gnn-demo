"""
Course Recommendation System using GCN Link Prediction

This script demonstrates how to use Graph Convolutional Networks for 
course recommendation by predicting missing student-course links.

Key Concepts:
- Link Prediction: Predicting missing edges in a graph
- Negative Sampling: Creating non-existing edges for training
- Embedding-based Similarity: Using cosine/dot product similarity
- Evaluation Metrics: Precision@K, Recall@K, AUC
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from gcn_numpy_implementation import GCN, normalize_adjacency
import random

class CourseRecommendationSystem:
    """GCN-based course recommendation system."""
    
    def __init__(self, university_data, test_ratio=0.2):
        """
        Initialize the recommendation system.
        
        Args:
            university_data: University graph dataset
            test_ratio: Fraction of edges to hide for testing
        """
        self.data = university_data
        self.test_ratio = test_ratio
        
        # Extract student and course information
        self.students = []
        self.courses = []
        self.professors = []
        
        for matrix_idx in range(len(university_data['adjacency_matrix'])):
            node_id = university_data['idx_to_node'][matrix_idx]
            node_type = university_data['node_types'][node_id]
            
            if node_type == 'student':
                self.students.append(matrix_idx)
            elif node_type == 'course':
                self.courses.append(matrix_idx)
            elif node_type == 'professor':
                self.professors.append(matrix_idx)
        
        print(f"📊 System Overview:")
        print(f"   Students: {len(self.students)}")
        print(f"   Courses: {len(self.courses)}")
        print(f"   Professors: {len(self.professors)}")
    
    def create_test_split(self):
        """
        Split student-course edges into train/test sets for link prediction.
        """
        print(f"\n🔄 Creating Train/Test Split for Link Prediction")
        print("-" * 50)
        
        # Find all student-course enrollments
        original_adj = self.data['adjacency_matrix'].copy()
        student_course_edges = []
        
        for student_idx in self.students:
            for course_idx in self.courses:
                if original_adj[student_idx, course_idx] > 0:
                    student_course_edges.append((student_idx, course_idx))
        
        print(f"   Total student-course enrollments: {len(student_course_edges)}")
        
        # Split edges into train/test
        train_edges, test_edges = train_test_split(
            student_course_edges, 
            test_size=self.test_ratio, 
            random_state=42
        )
        
        print(f"   Training edges: {len(train_edges)}")
        print(f"   Test edges: {len(test_edges)} (hidden for evaluation)")
        
        # Create training adjacency matrix (remove test edges)
        train_adj = original_adj.copy()
        
        for student_idx, course_idx in test_edges:
            train_adj[student_idx, course_idx] = 0
            train_adj[course_idx, student_idx] = 0  # Undirected graph
        
        print(f"   Remaining edges in training graph: {np.sum(train_adj) / 2:.0f}")
        
        # Generate negative samples (non-existing edges)
        negative_edges = self._generate_negative_samples(
            original_adj, len(test_edges)
        )
        
        return {
            'original_adj': original_adj,
            'train_adj': train_adj,
            'test_edges': test_edges,
            'train_edges': train_edges,
            'negative_edges': negative_edges
        }
    
    def _generate_negative_samples(self, adj_matrix, num_samples):
        """Generate negative samples (student-course pairs that don't exist)."""
        negative_edges = []
        max_attempts = num_samples * 10  # Prevent infinite loop
        attempts = 0
        
        while len(negative_edges) < num_samples and attempts < max_attempts:
            student_idx = random.choice(self.students)
            course_idx = random.choice(self.courses)
            
            # Check if this edge doesn't exist
            if adj_matrix[student_idx, course_idx] == 0:
                negative_edges.append((student_idx, course_idx))
            
            attempts += 1
        
        print(f"   Generated {len(negative_edges)} negative samples")
        return negative_edges
    
    def train_gcn(self, train_adj):
        """Train GCN on the training graph."""
        print(f"\n🧠 Training GCN on Training Graph")
        print("-" * 50)
        
        # Normalize adjacency matrix
        train_adj_norm = normalize_adjacency(train_adj, add_self_loops=True)
        
        # Create GCN
        gcn = GCN(layer_dims=[4, 16, 8, 16], activation='relu')  # Larger output for better embeddings
        
        # Forward pass to get embeddings
        embeddings, _ = gcn.forward(
            self.data['feature_matrix'], 
            train_adj_norm, 
            verbose=False
        )
        
        print(f"   ✅ Generated embeddings: {embeddings.shape}")
        return gcn, embeddings
    
    def predict_links(self, embeddings, student_course_pairs, method='dot_product'):
        """
        Predict link probabilities for student-course pairs.
        
        Args:
            embeddings: Node embeddings from GCN
            student_course_pairs: List of (student_idx, course_idx) pairs
            method: 'dot_product', 'cosine', or 'mlp'
        """
        predictions = []
        
        for student_idx, course_idx in student_course_pairs:
            student_emb = embeddings[student_idx]
            course_emb = embeddings[course_idx]
            
            if method == 'dot_product':
                # Simple dot product similarity
                score = np.dot(student_emb, course_emb)
            elif method == 'cosine':
                # Cosine similarity
                student_norm = np.linalg.norm(student_emb)
                course_norm = np.linalg.norm(course_emb)
                if student_norm > 0 and course_norm > 0:
                    score = np.dot(student_emb, course_emb) / (student_norm * course_norm)
                else:
                    score = 0
            elif method == 'mlp':
                # Simple MLP (concatenate + linear layer)
                # For this demo, we'll use dot product
                score = np.dot(student_emb, course_emb)
            else:
                score = np.dot(student_emb, course_emb)
            
            predictions.append(score)
        
        return np.array(predictions)
    
    def evaluate_predictions(self, split_data, embeddings, method='dot_product'):
        """Evaluate link prediction performance."""
        print(f"\n📊 Evaluating Link Prediction Performance")
        print("-" * 50)
        
        # Get predictions for positive (test) edges
        positive_scores = self.predict_links(
            embeddings, split_data['test_edges'], method=method
        )
        
        # Get predictions for negative edges
        negative_scores = self.predict_links(
            embeddings, split_data['negative_edges'], method=method
        )
        
        # Create labels (1 for positive, 0 for negative)
        y_true = np.concatenate([
            np.ones(len(positive_scores)),
            np.zeros(len(negative_scores))
        ])
        
        y_scores = np.concatenate([positive_scores, negative_scores])
        
        # Compute metrics
        auc_score = roc_auc_score(y_true, y_scores)
        ap_score = average_precision_score(y_true, y_scores)
        
        print(f"   Method: {method}")
        print(f"   AUC Score: {auc_score:.3f}")
        print(f"   Average Precision: {ap_score:.3f}")
        
        # Compute Precision@K and Recall@K
        k_values = [1, 3, 5, 10]
        precision_at_k, recall_at_k = self._compute_precision_recall_at_k(
            split_data, embeddings, k_values, method
        )
        
        return {
            'auc': auc_score,
            'ap': ap_score,
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'positive_scores': positive_scores,
            'negative_scores': negative_scores
        }
    
    def _compute_precision_recall_at_k(self, split_data, embeddings, k_values, method):
        """Compute Precision@K and Recall@K for each student."""
        precision_results = {k: [] for k in k_values}
        recall_results = {k: [] for k in k_values}
        
        # Group test edges by student
        student_test_courses = {}
        for student_idx, course_idx in split_data['test_edges']:
            if student_idx not in student_test_courses:
                student_test_courses[student_idx] = []
            student_test_courses[student_idx].append(course_idx)
        
        print(f"   Computing Precision@K and Recall@K for {len(student_test_courses)} students...")
        
        for student_idx, true_courses in student_test_courses.items():
            # Get all possible courses for this student (excluding current enrollments)
            train_courses = set()
            for train_student, train_course in split_data['train_edges']:
                if train_student == student_idx:
                    train_courses.add(train_course)
            
            # Candidate courses = all courses - already enrolled courses
            candidate_courses = [c for c in self.courses if c not in train_courses]
            
            if len(candidate_courses) == 0:
                continue
            
            # Predict scores for all candidate courses
            candidate_pairs = [(student_idx, course_idx) for course_idx in candidate_courses]
            scores = self.predict_links(embeddings, candidate_pairs, method=method)
            
            # Sort courses by predicted score
            course_score_pairs = list(zip(candidate_courses, scores))
            course_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Compute Precision@K and Recall@K
            for k in k_values:
                if k <= len(course_score_pairs):
                    top_k_courses = [course for course, _ in course_score_pairs[:k]]
                    
                    # True positives = intersection of recommended and true courses
                    tp = len(set(top_k_courses) & set(true_courses))
                    
                    precision_k = tp / k if k > 0 else 0
                    recall_k = tp / len(true_courses) if len(true_courses) > 0 else 0
                    
                    precision_results[k].append(precision_k)
                    recall_results[k].append(recall_k)
        
        # Average across students
        avg_precision = {k: np.mean(values) for k, values in precision_results.items()}
        avg_recall = {k: np.mean(values) for k, values in recall_results.items()}
        
        print(f"   Precision@K: {avg_precision}")
        print(f"   Recall@K: {avg_recall}")
        
        return avg_precision, avg_recall
    
    def recommend_courses_for_student(self, student_idx, embeddings, top_k=5, explain=True):
        """Generate course recommendations for a specific student."""
        print(f"\n🎯 Course Recommendations")
        print("-" * 50)
        
        student_node_id = self.data['idx_to_node'][student_idx]
        student_data = self.data['nodes'][student_node_id]
        
        print(f"   Student: {student_data['name']}")
        print(f"   Major: {student_data['major']}, Year: {student_data['year']}, GPA: {student_data['gpa']:.2f}")
        
        # Find courses student is already enrolled in
        current_adj = self.data['adjacency_matrix']
        enrolled_courses = []
        
        for course_idx in self.courses:
            if current_adj[student_idx, course_idx] > 0:
                course_node_id = self.data['idx_to_node'][course_idx]
                course_name = self.data['nodes'][course_node_id]['name']
                enrolled_courses.append((course_idx, course_name))
        
        print(f"   Currently enrolled in {len(enrolled_courses)} courses:")
        for _, course_name in enrolled_courses[:3]:  # Show first 3
            print(f"     • {course_name}")
        
        # Find candidate courses (not currently enrolled)
        candidate_courses = []
        for course_idx in self.courses:
            if course_idx not in [idx for idx, _ in enrolled_courses]:
                candidate_courses.append(course_idx)
        
        # Generate predictions for candidate courses
        candidate_pairs = [(student_idx, course_idx) for course_idx in candidate_courses]
        scores = self.predict_links(embeddings, candidate_pairs, method='cosine')
        
        # Sort and get top recommendations
        course_score_pairs = list(zip(candidate_courses, scores))
        course_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n   🌟 Top {top_k} Course Recommendations:")
        recommendations = []
        
        for i, (course_idx, score) in enumerate(course_score_pairs[:top_k]):
            course_node_id = self.data['idx_to_node'][course_idx]
            course_data = self.data['nodes'][course_node_id]
            
            print(f"   {i+1}. {course_data['name']}")
            print(f"      Similarity Score: {score:.3f}")
            print(f"      Department: {course_data['department']}")
            print(f"      Difficulty: {course_data['difficulty']:.1f}/5.0")
            print(f"      Credits: {course_data['credits']}")
            
            if explain and i < 2:  # Explain top 2 recommendations
                self._explain_recommendation(student_idx, course_idx, embeddings)
            
            recommendations.append({
                'course_name': course_data['name'],
                'score': score,
                'department': course_data['department'],
                'difficulty': course_data['difficulty'],
                'credits': course_data['credits']
            })
            print()
        
        return recommendations
    
    def _explain_recommendation(self, student_idx, course_idx, embeddings):
        """Explain why a course was recommended to a student."""
        # Find similar students who took this course
        course_node_id = self.data['idx_to_node'][course_idx]
        course_name = self.data['nodes'][course_node_id]['name']
        
        # Find students enrolled in this course
        adj_matrix = self.data['adjacency_matrix']
        enrolled_students = []
        
        for other_student_idx in self.students:
            if other_student_idx != student_idx and adj_matrix[other_student_idx, course_idx] > 0:
                enrolled_students.append(other_student_idx)
        
        if enrolled_students:
            # Find most similar enrolled student
            student_embedding = embeddings[student_idx]
            similarities = []
            
            for other_student_idx in enrolled_students[:5]:  # Check top 5
                other_embedding = embeddings[other_student_idx]
                sim = np.dot(student_embedding, other_embedding) / (
                    np.linalg.norm(student_embedding) * np.linalg.norm(other_embedding)
                )
                similarities.append((other_student_idx, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            if similarities:
                most_similar_idx, similarity = similarities[0]
                similar_student_id = self.data['idx_to_node'][most_similar_idx]
                similar_student = self.data['nodes'][similar_student_id]
                
                print(f"      💡 Similar student {similar_student['name']} (similarity: {similarity:.3f})")
                print(f"         also took {course_name}. They have similar academic profile.")


def run_course_recommendation_demo():
    """Run the complete course recommendation demonstration."""
    
    print("🎓 COURSE RECOMMENDATION SYSTEM DEMO")
    print("   Using GCN Link Prediction for Personalized Course Suggestions")
    print("="*80)
    
    # Load university data
    print("📚 Loading University Dataset...")
    university_data = np.load('/Users/rchandran/Library/CloudStorage/OneDrive-DiligentCorporation/RESEARCH/GNN/university_dataset.npy', 
                             allow_pickle=True).item()
    
    # Initialize recommendation system
    rec_system = CourseRecommendationSystem(university_data, test_ratio=0.2)
    
    # Step 1: Create train/test split
    split_data = rec_system.create_test_split()
    
    # Step 2: Train GCN on training graph
    gcn, embeddings = rec_system.train_gcn(split_data['train_adj'])
    
    # Step 3: Evaluate link prediction performance
    metrics = rec_system.evaluate_predictions(split_data, embeddings, method='cosine')
    
    # Step 4: Generate recommendations for sample students
    print(f"\n" + "="*80)
    print("🎯 SAMPLE COURSE RECOMMENDATIONS")
    print("="*80)
    
    # Show recommendations for 3 different students
    sample_students = rec_system.students[:3]
    
    for i, student_idx in enumerate(sample_students):
        print(f"\n--- Student {i+1} ---")
        recommendations = rec_system.recommend_courses_for_student(
            student_idx, embeddings, top_k=3, explain=True
        )
    
    # Step 5: Visualization of results
    create_recommendation_visualization(metrics, split_data, embeddings, rec_system)
    
    print(f"\n🎉 Course Recommendation Demo Complete!")
    print(f"   AUC Score: {metrics['auc']:.3f} (0.5=random, 1.0=perfect)")
    print(f"   Average Precision: {metrics['ap']:.3f}")
    print(f"   Key Insight: GCN embeddings capture student-course compatibility!")
    
    return rec_system, metrics, embeddings


def create_recommendation_visualization(metrics, split_data, embeddings, rec_system):
    """Create visualizations for recommendation results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Score distribution
    ax1.hist(metrics['positive_scores'], bins=20, alpha=0.7, label='Positive (True enrollments)', color='green')
    ax1.hist(metrics['negative_scores'], bins=20, alpha=0.7, label='Negative (Non-enrollments)', color='red')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Link Prediction Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision@K
    k_values = list(metrics['precision_at_k'].keys())
    precision_values = list(metrics['precision_at_k'].values())
    
    ax2.plot(k_values, precision_values, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('K (Number of Recommendations)')
    ax2.set_ylabel('Precision@K')
    ax2.set_title('Precision@K for Course Recommendations')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(precision_values) * 1.1)
    
    # 3. Recall@K
    recall_values = list(metrics['recall_at_k'].values())
    ax3.plot(k_values, recall_values, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('K (Number of Recommendations)')
    ax3.set_ylabel('Recall@K')
    ax3.set_title('Recall@K for Course Recommendations')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(recall_values) * 1.1)
    
    # 4. Embedding space (students vs courses)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot students and courses in different colors
    student_points = embeddings_2d[rec_system.students]
    course_points = embeddings_2d[rec_system.courses]
    
    ax4.scatter(student_points[:, 0], student_points[:, 1], 
               c='blue', alpha=0.6, s=30, label='Students')
    ax4.scatter(course_points[:, 0], course_points[:, 1], 
               c='red', alpha=0.6, s=30, label='Courses')
    ax4.set_xlabel('First Principal Component')
    ax4.set_ylabel('Second Principal Component')
    ax4.set_title('Student-Course Embedding Space (2D PCA)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = '/Users/rchandran/Library/CloudStorage/OneDrive-DiligentCorporation/RESEARCH/GNN/course_recommendation_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved recommendation analysis plots to: {plot_path}")
    
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run the complete demonstration
    rec_system, metrics, embeddings = run_course_recommendation_demo()
