"""
University Graph Dataset Creator

This script creates a realistic university graph with students, courses, and professors.
We'll use this as input for our GCN implementation.
"""

import numpy as np
import random
import os
from typing import Dict, List, Tuple, Any

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class UniversityGraphDataset:
    def __init__(self):
        """Initialize the university graph dataset generator."""
        # Define constants
        self.num_students = 100
        self.num_courses = 25
        self.num_professors = 12
        
        # Define mappings
        self.majors = ['CS', 'Math', 'Physics', 'Biology', 'Chemistry', 'Economics', 'English', 'History']
        self.departments = ['STEM', 'Liberal Arts', 'Business', 'Sciences']
        self.course_names = [
            'Data Structures', 'Calculus I', 'Calculus II', 'Physics I', 'Physics II',
            'Organic Chemistry', 'Biology', 'Microeconomics', 'Macroeconomics',
            'Linear Algebra', 'Statistics', 'Machine Learning', 'Databases',
            'Algorithms', 'English Lit', 'World History', 'Philosophy',
            'Quantum Mechanics', 'Differential Equations', 'Thermodynamics',
            'Molecular Biology', 'Financial Markets', 'Game Theory', 'Poetry',
            'Art History'
        ]
        
        self.nodes = {}  # Will store all node features
        self.edges = []  # Will store all edges
        self.node_types = {}  # Maps node_id to node_type
        
    def create_students(self) -> Dict[int, Dict[str, Any]]:
        """Create student nodes with realistic features."""
        students = {}
        
        for i in range(self.num_students):
            student_id = i  # Student IDs: 0-99
            
            # Generate realistic student features
            year = np.random.choice([1, 2, 3, 4], p=[0.3, 0.25, 0.25, 0.2])
            gpa = np.random.normal(3.2, 0.6)
            gpa = np.clip(gpa, 0.0, 4.0)  # Clip to valid GPA range
            
            major = np.random.choice(self.majors)
            major_encoded = self.majors.index(major)
            
            # Create feature vector: [year, gpa, major_encoded, is_senior]
            is_senior = 1.0 if year == 4 else 0.0
            features = np.array([year, gpa, major_encoded, is_senior], dtype=np.float32)
            
            students[student_id] = {
                'features': features,
                'type': 'student',
                'year': year,
                'gpa': gpa,
                'major': major,
                'name': f'Student_{i:03d}'
            }
            
        return students
    
    def create_courses(self) -> Dict[int, Dict[str, Any]]:
        """Create course nodes with realistic features."""
        courses = {}
        
        for i in range(self.num_courses):
            course_id = self.num_students + i  # Course IDs: 100-124
            
            course_name = self.course_names[i]
            credits = np.random.choice([3, 4], p=[0.7, 0.3])
            difficulty = np.random.uniform(1.0, 5.0)  # 1=easy, 5=very hard
            
            # Assign department based on course name
            if any(keyword in course_name.lower() for keyword in ['cs', 'data', 'algorithm', 'machine', 'database']):
                dept = 'STEM'
            elif any(keyword in course_name.lower() for keyword in ['calculus', 'linear', 'statistics', 'differential']):
                dept = 'STEM'
            elif any(keyword in course_name.lower() for keyword in ['physics', 'chemistry', 'biology', 'quantum', 'thermo']):
                dept = 'Sciences'
            elif any(keyword in course_name.lower() for keyword in ['economics', 'financial', 'game']):
                dept = 'Business'
            else:
                dept = 'Liberal Arts'
                
            dept_encoded = self.departments.index(dept)
            
            # Create feature vector: [credits, difficulty, dept_encoded, is_advanced]
            is_advanced = 1.0 if difficulty > 3.5 else 0.0
            features = np.array([credits, difficulty, dept_encoded, is_advanced], dtype=np.float32)
            
            courses[course_id] = {
                'features': features,
                'type': 'course',
                'name': course_name,
                'credits': credits,
                'difficulty': difficulty,
                'department': dept
            }
            
        return courses
    
    def create_professors(self) -> Dict[int, Dict[str, Any]]:
        """Create professor nodes with realistic features."""
        professors = {}
        
        for i in range(self.num_professors):
            prof_id = self.num_students + self.num_courses + i  # Professor IDs: 125-136
            
            years_experience = np.random.exponential(8.0)  # Average 8 years experience
            years_experience = np.clip(years_experience, 1.0, 40.0)
            
            rating = np.random.normal(4.2, 0.8)  # Average rating 4.2/5
            rating = np.clip(rating, 1.0, 5.0)
            
            department = np.random.choice(self.departments)
            dept_encoded = self.departments.index(department)
            
            # Create feature vector: [years_experience, rating, dept_encoded, is_senior_prof]
            is_senior_prof = 1.0 if years_experience > 15 else 0.0
            features = np.array([years_experience, rating, dept_encoded, is_senior_prof], dtype=np.float32)
            
            professors[prof_id] = {
                'features': features,
                'type': 'professor',
                'name': f'Prof_{i:02d}',
                'years_experience': years_experience,
                'rating': rating,
                'department': department
            }
            
        return professors
    
    def create_edges(self, students: Dict, courses: Dict, professors: Dict) -> List[Tuple[int, int, str]]:
        """Create realistic edges between nodes."""
        edges = []
        
        # 1. Student-Course enrollments
        for student_id, student_data in students.items():
            # Each student enrolls in 3-6 courses
            num_courses = np.random.randint(3, 7)
            
            # Bias course selection based on student's major
            student_major = student_data['major']
            course_probs = []
            
            for course_id, course_data in courses.items():
                base_prob = 0.1  # Base probability
                
                # Higher probability if course matches major
                if student_major == 'CS' and any(keyword in course_data['name'].lower() 
                                               for keyword in ['data', 'algorithm', 'machine', 'database']):
                    base_prob = 0.8
                elif student_major == 'Math' and any(keyword in course_data['name'].lower() 
                                                   for keyword in ['calculus', 'linear', 'statistics', 'differential']):
                    base_prob = 0.8
                elif student_major == 'Physics' and any(keyword in course_data['name'].lower() 
                                                      for keyword in ['physics', 'quantum', 'calculus']):
                    base_prob = 0.8
                # Add more major-specific biases...
                
                course_probs.append(base_prob)
            
            # Normalize probabilities
            course_probs = np.array(course_probs)
            course_probs = course_probs / course_probs.sum()
            
            # Select courses
            selected_courses = np.random.choice(
                list(courses.keys()), 
                size=num_courses, 
                replace=False, 
                p=course_probs
            )
            
            for course_id in selected_courses:
                edges.append((student_id, course_id, 'enrolls'))
        
        # 2. Professor-Course teaching assignments
        course_ids = list(courses.keys())
        for course_id in course_ids:
            # Each course has 1 professor
            # Bias professor selection based on department
            course_dept = courses[course_id]['department']
            prof_probs = []
            
            for prof_id, prof_data in professors.items():
                if prof_data['department'] == course_dept:
                    prob = 0.7  # High probability for same department
                else:
                    prob = 0.1  # Low probability for different department
                prof_probs.append(prob)
            
            prof_probs = np.array(prof_probs)
            prof_probs = prof_probs / prof_probs.sum()
            
            selected_prof = np.random.choice(
                list(professors.keys()), 
                p=prof_probs
            )
            
            edges.append((selected_prof, course_id, 'teaches'))
        
        return edges
    
    def create_adjacency_matrix(self) -> Tuple[np.ndarray, Dict[int, int]]:
        """Create adjacency matrix from edges."""
        total_nodes = self.num_students + self.num_courses + self.num_professors
        adj_matrix = np.zeros((total_nodes, total_nodes), dtype=np.float32)
        
        # Create node mapping (original_id -> matrix_index)
        node_mapping = {}
        for i, node_id in enumerate(sorted(self.nodes.keys())):
            node_mapping[node_id] = i
        
        # Fill adjacency matrix
        for source, target, edge_type in self.edges:
            source_idx = node_mapping[source]
            target_idx = node_mapping[target]
            
            # Make the graph undirected
            adj_matrix[source_idx, target_idx] = 1.0
            adj_matrix[target_idx, source_idx] = 1.0
        
        return adj_matrix, node_mapping
    
    def create_feature_matrix(self, node_mapping: Dict[int, int]) -> np.ndarray:
        """Create node feature matrix."""
        total_nodes = len(node_mapping)
        feature_dim = 4  # All node types have 4 features
        
        feature_matrix = np.zeros((total_nodes, feature_dim), dtype=np.float32)
        
        for node_id, node_data in self.nodes.items():
            matrix_idx = node_mapping[node_id]
            feature_matrix[matrix_idx] = node_data['features']
        
        return feature_matrix
    
    def generate_dataset(self) -> Dict[str, Any]:
        """Generate the complete university graph dataset."""
        print("🎓 Creating University Graph Dataset...")
        
        # Create nodes
        students = self.create_students()
        courses = self.create_courses()
        professors = self.create_professors()
        
        # Combine all nodes
        self.nodes.update(students)
        self.nodes.update(courses)
        self.nodes.update(professors)
        
        # Store node types
        for node_id, node_data in self.nodes.items():
            self.node_types[node_id] = node_data['type']
        
        # Create edges
        self.edges = self.create_edges(students, courses, professors)
        
        # Create matrices
        adj_matrix, node_mapping = self.create_adjacency_matrix()
        feature_matrix = self.create_feature_matrix(node_mapping)
        
        # Create reverse mapping for easier interpretation
        idx_to_node = {idx: node_id for node_id, idx in node_mapping.items()}
        
        dataset = {
            'adjacency_matrix': adj_matrix,
            'feature_matrix': feature_matrix,
            'node_mapping': node_mapping,
            'idx_to_node': idx_to_node,
            'edges': self.edges,
            'nodes': self.nodes,
            'node_types': self.node_types,
            'num_students': self.num_students,
            'num_courses': self.num_courses,
            'num_professors': self.num_professors,
            'majors': self.majors,
            'departments': self.departments
        }
        
        self._print_dataset_stats(dataset)
        return dataset
    
    def _print_dataset_stats(self, dataset: Dict[str, Any]) -> None:
        """Print dataset statistics."""
        adj = dataset['adjacency_matrix']
        features = dataset['feature_matrix']
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Nodes: {adj.shape[0]} ({self.num_students} students, {self.num_courses} courses, {self.num_professors} professors)")
        print(f"   Edges: {len(self.edges)} (undirected)")
        print(f"   Features per node: {features.shape[1]}")
        print(f"   Graph density: {np.sum(adj) / (adj.shape[0] * (adj.shape[0] - 1)):.4f}")
        
        # Count edges by type
        edge_types = {}
        for _, _, edge_type in self.edges:
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print(f"   Edge types: {edge_types}")


if __name__ == "__main__":
    # Generate the dataset
    dataset_generator = UniversityGraphDataset()
    university_data = dataset_generator.generate_dataset()
    
    # Save the dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'university_dataset.npy')
    np.save(dataset_path, university_data)
    print(f"\n💾 Dataset saved to {dataset_path}")
    
    # Display some sample data
    print(f"\n🔍 Sample Data Preview:")
    print(f"Adjacency matrix shape: {university_data['adjacency_matrix'].shape}")
    print(f"Feature matrix shape: {university_data['feature_matrix'].shape}")
    
    # Show a few example nodes
    print(f"\n📚 Sample Nodes:")
    for i, (node_id, node_data) in enumerate(list(university_data['nodes'].items())[:5]):
        print(f"   {node_data['type'].capitalize()}: {node_data.get('name', f'Node_{node_id}')} - Features: {node_data['features']}")
