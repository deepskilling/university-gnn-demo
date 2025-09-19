"""
GCN Prediction Tasks - What Can We Predict?

This module demonstrates various prediction tasks that can be performed
using the learned node embeddings from our university GCN model.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from gcn_numpy_implementation import GCN, normalize_adjacency
import matplotlib.pyplot as plt

class UniversityPredictor:
    """Prediction tasks using GCN embeddings for university graph."""
    
    def __init__(self, university_data, embeddings):
        self.data = university_data
        self.embeddings = embeddings
        self.nodes = university_data['nodes']
        self.idx_to_node = university_data['idx_to_node']
        self.node_types = university_data['node_types']
        
    def predict_student_major(self, test_size=0.3):
        """
        Task 1: NODE CLASSIFICATION - Predict student major from embeddings
        
        This demonstrates how GCN embeddings can be used for classification tasks.
        """
        print("="*70)
        print("🎯 TASK 1: STUDENT MAJOR PREDICTION")
        print("="*70)
        
        # Extract student data
        student_indices = []
        student_majors = []
        
        for matrix_idx in range(len(self.embeddings)):
            node_id = self.idx_to_node[matrix_idx]
            if self.node_types[node_id] == 'student':
                student_indices.append(matrix_idx)
                major = self.nodes[node_id]['major']
                student_majors.append(major)
        
        student_embeddings = self.embeddings[student_indices]
        
        print(f"📊 Dataset: {len(student_embeddings)} students")
        print(f"   Embedding dimension: {student_embeddings.shape[1]}")
        print(f"   Unique majors: {len(set(student_majors))}")
        print(f"   Major distribution: {dict(zip(*np.unique(student_majors, return_counts=True)))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            student_embeddings, student_majors, test_size=test_size, random_state=42
        )
        
        # Train classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Show some example predictions
        print(f"\n🔍 Sample Predictions:")
        for i in range(min(5, len(y_test))):
            actual = y_test[i]
            predicted = y_pred[i]
            confidence = classifier.predict_proba(X_test[i:i+1])[0].max()
            status = "✅" if actual == predicted else "❌"
            print(f"   {status} Actual: {actual:12} | Predicted: {predicted:12} | Confidence: {confidence:.3f}")
        
        return accuracy, classifier
    
    def predict_course_difficulty(self, test_size=0.3):
        """
        Task 2: REGRESSION/CLASSIFICATION - Predict course difficulty levels
        """
        print("\n" + "="*70)
        print("🎯 TASK 2: COURSE DIFFICULTY PREDICTION") 
        print("="*70)
        
        # Extract course data
        course_indices = []
        course_difficulties = []
        
        for matrix_idx in range(len(self.embeddings)):
            node_id = self.idx_to_node[matrix_idx]
            if self.node_types[node_id] == 'course':
                course_indices.append(matrix_idx)
                difficulty = self.nodes[node_id]['difficulty']
                # Convert to categorical: Easy (1-2.5), Medium (2.5-3.5), Hard (3.5-5)
                if difficulty <= 2.5:
                    level = 'Easy'
                elif difficulty <= 3.5:
                    level = 'Medium' 
                else:
                    level = 'Hard'
                course_difficulties.append(level)
        
        course_embeddings = self.embeddings[course_indices]
        
        print(f"📊 Dataset: {len(course_embeddings)} courses")
        print(f"   Difficulty distribution: {dict(zip(*np.unique(course_difficulties, return_counts=True)))}")
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            course_embeddings, course_difficulties, test_size=test_size, random_state=42
        )
        
        classifier = RandomForestClassifier(random_state=42, n_estimators=100)
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        
        return accuracy, classifier
    
    def link_prediction_course_recommendation(self, student_idx=0, top_k=5):
        """
        Task 3: LINK PREDICTION - Recommend courses to students
        
        Uses embedding similarity to recommend courses that are similar
        to courses the student is already taking.
        """
        print("\n" + "="*70)
        print("🎯 TASK 3: COURSE RECOMMENDATION (Link Prediction)")
        print("="*70)
        
        # Get student info
        student_node_id = self.idx_to_node[student_idx]
        student_data = self.nodes[student_node_id]
        student_embedding = self.embeddings[student_idx:student_idx+1]
        
        print(f"🎓 Recommending courses for: {student_data['name']}")
        print(f"   Major: {student_data['major']}, Year: {student_data['year']}, GPA: {student_data['gpa']:.2f}")
        
        # Find courses student is already taking
        current_courses = []
        adj_matrix = self.data['adjacency_matrix']
        
        for matrix_idx in range(len(self.embeddings)):
            if adj_matrix[student_idx, matrix_idx] > 0:  # Connected
                node_id = self.idx_to_node[matrix_idx]
                if self.node_types[node_id] == 'course':
                    current_courses.append((matrix_idx, self.nodes[node_id]['name']))
        
        print(f"   Currently enrolled in {len(current_courses)} courses:")
        for _, course_name in current_courses[:3]:
            print(f"     • {course_name}")
        
        # Find all courses not currently taken
        available_courses = []
        course_embeddings = []
        
        for matrix_idx in range(len(self.embeddings)):
            node_id = self.idx_to_node[matrix_idx]
            if (self.node_types[node_id] == 'course' and 
                matrix_idx not in [idx for idx, _ in current_courses]):
                available_courses.append((matrix_idx, self.nodes[node_id]))
                course_embeddings.append(self.embeddings[matrix_idx])
        
        if not course_embeddings:
            print("   No available courses to recommend!")
            return []
        
        course_embeddings = np.array(course_embeddings)
        
        # Calculate similarities
        similarities = cosine_similarity(student_embedding, course_embeddings)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        recommendations = []
        
        print(f"\n🎯 Top {top_k} Course Recommendations:")
        for i, idx in enumerate(top_indices):
            course_matrix_idx, course_data = available_courses[idx]
            similarity = similarities[idx]
            
            print(f"   {i+1}. {course_data['name']}")
            print(f"      Similarity: {similarity:.3f} | Difficulty: {course_data['difficulty']:.1f}")
            print(f"      Department: {course_data['department']} | Credits: {course_data['credits']}")
            
            recommendations.append({
                'course_name': course_data['name'],
                'similarity': similarity,
                'difficulty': course_data['difficulty'],
                'department': course_data['department']
            })
        
        return recommendations
    
    def anomaly_detection_students(self, contamination=0.1):
        """
        Task 4: ANOMALY DETECTION - Find unusual students
        
        Identifies students whose embeddings are outliers, which might
        indicate unusual academic patterns.
        """
        print("\n" + "="*70)
        print("🎯 TASK 4: ACADEMIC ANOMALY DETECTION")
        print("="*70)
        
        from sklearn.ensemble import IsolationForest
        
        # Extract student embeddings
        student_indices = []
        for matrix_idx in range(len(self.embeddings)):
            node_id = self.idx_to_node[matrix_idx]
            if self.node_types[node_id] == 'student':
                student_indices.append(matrix_idx)
        
        student_embeddings = self.embeddings[student_indices]
        
        # Detect anomalies
        clf = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = clf.fit_predict(student_embeddings)
        
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        normal_indices = np.where(anomaly_labels == 1)[0]
        
        print(f"📊 Anomaly Detection Results:")
        print(f"   Normal students: {len(normal_indices)}")
        print(f"   Anomalous students: {len(anomaly_indices)}")
        
        print(f"\n🚨 Potentially Unusual Students:")
        for i, anomaly_idx in enumerate(anomaly_indices[:5]):
            student_matrix_idx = student_indices[anomaly_idx]
            student_node_id = self.idx_to_node[student_matrix_idx]
            student_data = self.nodes[student_node_id]
            
            print(f"   {i+1}. {student_data['name']}")
            print(f"      Year: {student_data['year']}, GPA: {student_data['gpa']:.2f}, Major: {student_data['major']}")
            print(f"      Embedding: {self.embeddings[student_matrix_idx]}")
        
        return anomaly_indices, normal_indices
    
    def similarity_search(self, query_node_idx, top_k=5, node_type_filter=None):
        """
        Task 5: SIMILARITY SEARCH - Find most similar nodes
        """
        print("\n" + "="*70)
        print("🎯 TASK 5: NODE SIMILARITY SEARCH")
        print("="*70)
        
        query_node_id = self.idx_to_node[query_node_idx]
        query_data = self.nodes[query_node_id]
        query_embedding = self.embeddings[query_node_idx:query_node_idx+1]
        
        print(f"🔍 Finding nodes similar to: {query_data.get('name', f'Node_{query_node_id}')} ({query_data['type']})")
        
        # Calculate similarities to all other nodes
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        similarities[query_node_idx] = -1  # Exclude self
        
        # Filter by node type if specified
        if node_type_filter:
            for i in range(len(similarities)):
                node_id = self.idx_to_node[i]
                if self.node_types[node_id] != node_type_filter:
                    similarities[i] = -1
        
        # Get top similar nodes
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        print(f"\n🎯 Top {top_k} Most Similar Nodes:")
        for i, idx in enumerate(top_indices):
            if similarities[idx] == -1:
                continue
                
            similar_node_id = self.idx_to_node[idx]
            similar_data = self.nodes[similar_node_id]
            similarity = similarities[idx]
            
            print(f"   {i+1}. {similar_data.get('name', f'Node_{similar_node_id}')} ({similar_data['type']})")
            print(f"      Similarity: {similarity:.3f}")
            
            if similar_data['type'] == 'student':
                print(f"      Details: Year {similar_data['year']}, GPA {similar_data['gpa']:.2f}, {similar_data['major']}")
            elif similar_data['type'] == 'course':
                print(f"      Details: {similar_data['department']}, Difficulty {similar_data['difficulty']:.1f}")
        
        return top_indices, similarities[top_indices]


def run_all_prediction_tasks():
    """Run all prediction tasks demonstration."""
    
    print("🎓 GCN PREDICTION TASKS DEMONSTRATION")
    print("🔬 What can we predict with our university graph embeddings?")
    print("="*80)
    
    # Load data and run GCN
    print("📚 Loading data and computing embeddings...")
    university_data = np.load('/Users/rchandran/Library/CloudStorage/OneDrive-DiligentCorporation/RESEARCH/GNN/university_dataset.npy', 
                             allow_pickle=True).item()
    
    A = university_data['adjacency_matrix']
    X = university_data['feature_matrix']
    A_norm = normalize_adjacency(A, add_self_loops=True)
    
    # Run GCN to get embeddings
    gcn = GCN(layer_dims=[4, 16, 8, 3], activation='relu')
    final_embeddings, _ = gcn.forward(X, A_norm, verbose=False)
    
    print(f"   ✅ Computed embeddings: {final_embeddings.shape}")
    
    # Initialize predictor
    predictor = UniversityPredictor(university_data, final_embeddings)
    
    # Run all prediction tasks
    results = {}
    
    # Task 1: Student major prediction
    major_accuracy, major_classifier = predictor.predict_student_major()
    results['major_prediction'] = major_accuracy
    
    # Task 2: Course difficulty prediction  
    difficulty_accuracy, difficulty_classifier = predictor.predict_course_difficulty()
    results['difficulty_prediction'] = difficulty_accuracy
    
    # Task 3: Course recommendations
    recommendations = predictor.link_prediction_course_recommendation(student_idx=0)
    results['course_recommendations'] = recommendations
    
    # Task 4: Anomaly detection
    anomalies, normal = predictor.anomaly_detection_students()
    results['anomaly_detection'] = len(anomalies)
    
    # Task 5: Similarity search
    similar_indices, similarities = predictor.similarity_search(query_node_idx=0, node_type_filter='student')
    results['similarity_search'] = similarities
    
    # Summary
    print("\n" + "="*80)
    print("📊 PREDICTION TASKS SUMMARY")
    print("="*80)
    print(f"✅ Major Prediction Accuracy: {results['major_prediction']:.3f}")
    print(f"✅ Difficulty Prediction Accuracy: {results['difficulty_prediction']:.3f}")
    print(f"✅ Course Recommendations Generated: {len(results['course_recommendations'])}")
    print(f"✅ Academic Anomalies Detected: {results['anomaly_detection']}")
    print(f"✅ Similarity Search Completed")
    
    print(f"\n🎯 Key Insights:")
    print(f"   • GCN embeddings enable accurate classification of node properties")
    print(f"   • Graph structure helps with recommendation systems")
    print(f"   • Embedding similarity reveals meaningful relationships")
    print(f"   • Anomaly detection can identify unusual academic patterns")
    
    return results


if __name__ == "__main__":
    results = run_all_prediction_tasks()
