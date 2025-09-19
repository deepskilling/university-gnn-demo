# University Graph Neural Network (GCN) Demo 🎓

A comprehensive, educational implementation of Graph Convolutional Networks (GCNs) using pure NumPy, demonstrated on a realistic university graph dataset.

## 🚀 What This Project Does

This project creates a **step-by-step walkthrough** of Graph Neural Networks using an intuitive university scenario:

- **100 Students** with features (year, GPA, major)
- **25 Courses** with features (credits, difficulty, department)  
- **12 Professors** with features (experience, rating, department)
- **Realistic relationships**: student enrollments, professor teaching assignments

The GCN learns to create **meaningful embeddings** that capture both individual node features and graph structure.

## 📁 Project Structure

```
university-gnn-demo/
├── README.md                       # Main documentation
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
├── .gitignore                     # Git ignore rules
│
├── src/                           # Core implementation
│   ├── gcn_numpy_implementation.py     # GCN layers and model
│   ├── university_graph_dataset.py    # Dataset generation
│   └── course_recommendation_system.py # Link prediction system
│
├── demos/                         # Educational demonstrations
│   ├── gnn_educational_demo.py         # Step-by-step walkthrough
│   ├── quick_demo.py                   # Quick highlights
│   ├── prediction_tasks.py             # All prediction capabilities
│   ├── focused_recommendation_test.py  # Detailed analysis
│   ├── link_prediction_summary.py     # Complete workflow
│   ├── improving_predictions.py        # Training improvements
│   └── prediction_summary.py          # Performance summary
│
├── data/                          # Generated datasets
│   └── university_dataset.npy         # Graph data
│
├── results/                       # Visualization outputs
│   ├── embedding_visualization.png     # 2D embeddings
│   ├── course_recommendation_results.png # Performance charts
│   └── simple_embedding_viz.png        # Simple visualization
│
├── docs/                          # Documentation
│   ├── GITHUB_SETUP.md                # Setup instructions
│   └── SECURE_GIT_USAGE.md            # Git security guide
│
└── scripts/                       # Utility scripts
    └── git_with_env.sh                # Git helper (local)
```

## 🔧 Key Concepts Demonstrated

### 1. **Graph Structure**
- Nodes: Students, courses, professors
- Edges: "enrolls in" (student-course), "teaches" (professor-course)
- Features: Meaningful numerical representations for each node type

### 2. **Message Passing**
```
Student embedding = f(
    own_features + 
    aggregated_neighbor_features
)
```

### 3. **GCN Architecture**
```
Layer 1: 4 features → 16 hidden (ReLU)
Layer 2: 16 → 8 hidden (ReLU)  
Layer 3: 8 → 3 final embeddings
```

### 4. **Adjacency Normalization**
```
A_norm = D^(-1/2) @ (A + I) @ D^(-1/2)
```
Where D is the degree matrix and I adds self-loops.

## 🎯 Running the Demo

### Quick Start
```bash
conda activate graph  # or your preferred environment

# Generate dataset
python src/university_graph_dataset.py

# Run core demonstrations
python src/gcn_numpy_implementation.py    # Core GCN walkthrough
python demos/gnn_educational_demo.py     # Detailed educational demo
python demos/quick_demo.py               # Quick highlights
```

### What You'll See

1. **Dataset Generation**: Creation of realistic university relationships
2. **Adjacency Normalization**: Mathematical preprocessing explained
3. **Layer-by-Layer Forward Pass**: Detailed computation walkthrough
4. **Message Passing**: How nodes aggregate neighbor information
5. **Embedding Analysis**: What the GCN learned about node relationships
6. **Visualization**: 2D plot of learned embeddings

### Link Prediction Demo
```bash
# Comprehensive prediction capabilities
python demos/prediction_tasks.py           # Shows all 5 prediction types

# Course recommendation system (Link Prediction)  
python src/course_recommendation_system.py  # Complete recommendation demo
python demos/focused_recommendation_test.py   # Detailed analysis
python demos/link_prediction_summary.py      # Workflow summary
```

**Link Prediction Results**:
- **Course Recommendations**: Personalized suggestions for each student
- **Performance Metrics**: AUC scores, Precision@K, Recall@K
- **Similarity Analysis**: Why certain courses are recommended
- **Real-world Workflow**: Complete train/test/evaluate pipeline

## 📊 Sample Output

```
🎓 Example: Student_000
   Features: [2.0, 2.53, 2, 0]  # [year, gpa, major_encoded, is_senior]
   Year: 2, GPA: 2.53, Major: Physics
   Enrolled in 4 courses:
     1. Biology (Features: [3, 1.15, 3, 0])
     2. Calculus I (Features: [4, 2.35, 0, 0])
     3. Quantum Mechanics (Features: [3, 4.85, 0, 1])
     4. Microeconomics (Features: [4, 2.44, 0, 0])

📨 Messages from neighbors:
   After normalization: connection weights sum to 0.601
   Self-loop weight: 0.200

🎯 Final embedding: [-1.505, -0.453, -1.057]
```

## 🧠 Educational Value

### Core GNN Concepts Covered:
- **Node Features**: How to represent different entity types numerically
- **Graph Structure**: Adjacency matrices and their normalization
- **Message Passing**: The fundamental GNN operation
- **Multi-layer Learning**: How information propagates through layers
- **Embedding Spaces**: Interpreting learned representations

### Mathematical Foundations:
- **Linear Algebra**: Matrix operations, eigenvalues/eigenvectors concepts
- **Graph Theory**: Degrees, adjacency, neighborhood aggregation
- **Neural Networks**: Weight matrices, activations, forward propagation

## 🔍 Key Insights from Results

1. **Similar Nodes Cluster**: Students with similar majors/years have similar embeddings
2. **Cross-type Relationships**: High similarity between students and their enrolled courses
3. **Structural Information**: Graph connections influence final representations
4. **Feature + Structure**: Embeddings capture both individual features AND graph position

## 🎨 Visualizations

The project generates:
- **2D Embedding Plot**: PCA reduction showing student/course/professor clusters
- **Similarity Analysis**: Most similar node pairs across different types
- **Layer-wise Transformations**: How features evolve through each GCN layer

## 🎯 Prediction Capabilities

This GCN model enables **5 major types of predictions**:

### 📊 1. Node Classification (80-95% accuracy in production)
- **Student Major Prediction**: Classify students by academic focus
- **Course Difficulty Assessment**: Easy/Medium/Hard classification  
- **Faculty Department Assignment**: STEM/Liberal Arts/Business
- **Academic Performance Prediction**: GPA ranges, graduation likelihood

### 🔗 2. Link Prediction (70-90% accuracy in production)
- **Course Recommendations**: Suggest courses students might enjoy
- **Academic Advisor Matching**: Connect students with suitable professors
- **Study Group Formation**: Find students who should collaborate
- **Future Enrollment Prediction**: Predict course selections

### 🚨 3. Anomaly Detection (85-95% precision in production)
- **Academic Risk Assessment**: Identify students at risk of dropping out
- **Grade Inconsistencies**: Detect unusual performance patterns
- **Course Load Analysis**: Find overloaded/underloaded students
- **Fraud Detection**: Identify potentially fake academic records

### 🔍 4. Similarity & Recommendation Systems
- **"Students Like You"**: Find academically similar peers
- **Course Similarity**: Group courses by content/difficulty
- **Academic Path Recommendations**: Suggest career trajectories
- **Transfer Student Placement**: Match transfers to appropriate courses

### 📈 5. Graph-Level Analytics
- **Department Popularity Trends**: Predict enrollment patterns
- **Course Success Rates**: Forecast completion rates
- **Academic Network Analysis**: Understand learning communities
- **Resource Allocation**: Optimize course offerings and professor assignments

**Note**: Our demo shows lower accuracy (20-38%) because it uses untrained, randomly initialized weights. Production GCNs achieve 80-95% accuracy with proper training, larger datasets, and advanced techniques.

## 🚀 Extensions & Next Steps

### Easy Extensions:
- **Proper Training**: Implement gradient descent to achieve 80%+ accuracy
- **Attention Mechanisms**: Add GAT (Graph Attention Networks)
- **Different Aggregators**: Mean, max, LSTM-based aggregation
- **More Features**: GPA history, prerequisite chains, temporal data

### Advanced Extensions:
- **Heterogeneous Graphs**: Different edge types with different weights
- **Dynamic Graphs**: Time-evolving university relationships
- **Graph Generation**: Create new realistic university graphs
- **Production Deployment**: PyTorch Geometric, GPU acceleration, API serving

## 📚 Dependencies

```python
numpy>=1.20.0
matplotlib>=3.3.0
scikit-learn>=0.24.0  # For PCA visualization
```

## 🎓 Learning Objectives

After running this demo, you should understand:

✅ **What graphs are** and how to represent them mathematically  
✅ **How GNNs differ** from regular neural networks  
✅ **Message passing mechanics** - the core of all GNNs  
✅ **Why normalization matters** in graph convolutions  
✅ **How to interpret** learned node embeddings  
✅ **When to use GNNs** vs other ML approaches  

## 🤝 Contributing

This is an educational project! Feel free to:
- Add more realistic features (student GPAs over time, course prerequisites)
- Implement other GNN variants (GraphSAGE, GAT, etc.)
- Add more comprehensive visualizations
- Create interactive exploration tools

## 📖 References

- [Kipf & Welling (2016): Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- [Hamilton (2020): Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/)
- [Veličković et al. (2017): Graph Attention Networks](https://arxiv.org/abs/1710.10903)

---

**Happy Learning!** 🎉 This implementation prioritizes **understanding over performance** - every step is designed to be educational and transparent.
