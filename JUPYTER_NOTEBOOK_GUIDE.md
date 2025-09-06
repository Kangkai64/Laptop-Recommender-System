# Using Laptop Recommender System in Jupyter Notebooks

This guide shows you how to use the Laptop Recommender System and its comprehensive evaluation framework in Jupyter notebooks.

## 🚀 Quick Start

### 1. Open Jupyter Notebook
```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### 2. Create a New Notebook
- Click "New" → "Python 3" (or your preferred kernel)
- Name it something like "Laptop_Recommender_Analysis.ipynb"

### 3. Copy and Paste Code Blocks
Copy the code blocks from `jupyter_notebook_demo.py` into separate cells in your notebook.

## 📋 Step-by-Step Instructions

### Cell 1: Setup and Imports
```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import our recommendation system
from Laptop_Recommender_System import create_laptop_recommender_system
from evaluation_metrics import create_evaluator, create_satisfaction_survey
from user_satisfaction_system import create_satisfaction_system
from ab_testing_framework import create_ab_testing_framework
from evaluate_recommender_system import RecommenderSystemEvaluator

print("✅ All imports successful!")
```

### Cell 2: Initialize the System
```python
# Initialize the recommendation system
print("Initializing Laptop Recommender System...")
recommender = create_laptop_recommender_system()

# Load and preprocess data
print("Loading and preprocessing data...")
df_laptop, df_rating = recommender.load_and_preprocess_data()

# Initialize recommendation engines
print("Initializing recommendation engines...")
recommender.initialize_recommendation_engines()

print("✅ System initialized successfully!")
print(f"📊 Dataset loaded: {len(df_laptop)} laptops, {len(df_rating)} ratings")
```

### Cell 3: Get Recommendations
```python
# Get content-based recommendations
print("🎯 Content-Based Recommendations:")
gaming_preferences = {
    'search_terms': ['gaming', 'performance', 'graphics'],
    'min_rating': 4.0,
    'max_price': 8000
}

gaming_recs = recommender.get_content_based_recommendations(gaming_preferences, 5)
for i, rec in enumerate(gaming_recs, 1):
    print(f"{i}. {rec.get('title', 'Unknown')} - RM {rec.get('price_myr', 0):.0f} - Rating: {rec.get('rating', 0):.1f}")
```

### Cell 4: Run Evaluation
```python
# Create evaluator and run evaluation
print("📊 Creating evaluation system...")
evaluator = create_evaluator(df_laptop, df_rating)

print("🔍 Running comprehensive evaluation...")
evaluation_results = evaluator.evaluate_system_performance(recommender)

print("✅ Evaluation completed!")
print(f"⏱️ Duration: {evaluation_results.get('evaluation_duration', 0):.2f} seconds")

# Display results
print("\n📈 EVALUATION RESULTS:")
if 'content_based' in evaluation_results:
    cb = evaluation_results['content_based']
    print(f"Content-Based - Precision: {cb.get('precision', 0):.3f}, Recall: {cb.get('recall', 0):.3f}, F1: {cb.get('f1_score', 0):.3f}")
```

### Cell 5: User Satisfaction System
```python
# Create satisfaction system
satisfaction_system = create_satisfaction_system()

# Start a session and submit responses
session_id = satisfaction_system.start_satisfaction_session(
    user_id="notebook_user_1",
    recommendation_method="hybrid"
)

# Submit sample responses
sample_responses = [
    ("overall_satisfaction", 4),
    ("relevance", 4),
    ("diversity", 3),
    ("accuracy", 5),
    ("speed", 4)
]

for question_id, response_value in sample_responses:
    satisfaction_system.submit_satisfaction_response(
        session_id=session_id,
        question_id=question_id,
        response_value=response_value
    )

# Calculate metrics
metrics = satisfaction_system.calculate_satisfaction_metrics()
print(f"Overall Satisfaction: {metrics.get('avg_satisfaction', 0):.2f}/5")
```

### Cell 6: A/B Testing
```python
# Create A/B testing framework
ab_framework = create_ab_testing_framework()

# Create an experiment
experiment_id = ab_framework.create_experiment(
    name="Content-Based vs Collaborative Filtering",
    description="Compare recommendation algorithms",
    variants=[
        {"name": "A", "config": {"algorithm": "content_based"}},
        {"name": "B", "config": {"algorithm": "collaborative"}}
    ],
    metrics=["click_rate", "conversion_rate", "satisfaction_score"],
    duration_days=7,
    sample_size=50,
    confidence_level=0.95
)

# Start experiment
ab_framework.start_experiment(experiment_id)
print(f"✅ Experiment created: {experiment_id}")
```

### Cell 7: Create Visualizations
```python
# Create evaluation comparison chart
import matplotlib.pyplot as plt

metrics_data = {
    'Content-Based': {
        'Precision': 0.785,
        'Recall': 0.732,
        'F1 Score': 0.758
    },
    'Collaborative': {
        'Precision': 0.812,
        'Recall': 0.768,
        'F1 Score': 0.789
    },
    'Hybrid': {
        'Precision': 0.845,
        'Recall': 0.801,
        'F1 Score': 0.823
    }
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(['Precision', 'Recall', 'F1 Score']):
    ax = axes[i]
    methods = list(metrics_data.keys())
    values = [metrics_data[method][metric] for method in methods]
    
    bars = ax.bar(methods, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

## 🎯 Key Features for Jupyter Notebooks

### 1. **Interactive Data Exploration**
```python
# Explore the dataset
print("Dataset Info:")
print(f"Shape: {df_laptop.shape}")
print(f"Columns: {list(df_laptop.columns)}")

# Display sample data
df_laptop.head()

# Basic statistics
df_laptop.describe()
```

### 2. **Real-time Evaluation**
```python
# Run evaluation and get immediate results
evaluator = create_evaluator(df_laptop, df_rating)
results = evaluator.evaluate_system_performance(recommender)

# Display results in a nice format
pd.DataFrame([results['content_based_evaluation']]).T
```

### 3. **Interactive Visualizations**
```python
# Create interactive plots
import plotly.express as px
import plotly.graph_objects as go

# Satisfaction scores over time
fig = px.bar(x=list(category_scores.keys()), 
             y=list(category_scores.values()),
             title="User Satisfaction by Category")
fig.show()
```

### 4. **A/B Testing Analysis**
```python
# Analyze A/B test results
results = ab_framework.analyze_experiment(experiment_id)

# Create comparison table
comparison_data = []
for metric in results.variant_a_results.keys():
    comparison_data.append({
        'Metric': metric,
        'Variant A': results.variant_a_results[metric]['mean'],
        'Variant B': results.variant_b_results[metric]['mean'],
        'Significant': results.statistical_significance[metric],
        'P-value': results.p_values[metric]
    })

pd.DataFrame(comparison_data)
```

## 🔧 Customization Examples

### 1. **Custom Evaluation Metrics**
```python
# Define custom evaluation scenarios
custom_scenarios = [
    {
        'name': 'Gaming Laptops',
        'preferences': {
            'search_terms': ['gaming', 'performance', 'graphics'],
            'min_rating': 4.0,
            'max_price': 8000
        }
    },
    {
        'name': 'Student Laptops',
        'preferences': {
            'search_terms': ['student', 'budget', 'portable'],
            'min_rating': 3.5,
            'max_price': 3000
        }
    }
]

# Evaluate each scenario
for scenario in custom_scenarios:
    print(f"\n🎯 Evaluating {scenario['name']}:")
    recs = recommender.get_content_based_recommendations(scenario['preferences'], 5)
    print(f"Found {len(recs)} recommendations")
```

### 2. **Performance Monitoring**
```python
# Monitor system performance over time
import time

def benchmark_recommendation_speed(n_tests=10):
    times = []
    for i in range(n_tests):
        start_time = time.time()
        recs = recommender.get_content_based_recommendations(gaming_preferences, 10)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }

performance = benchmark_recommendation_speed()
print(f"Average recommendation time: {performance['avg_time']:.3f}s ± {performance['std_time']:.3f}s")
```

### 3. **Data Analysis and Insights**
```python
# Analyze laptop data
print("📊 Laptop Data Analysis:")

# Price distribution
plt.figure(figsize=(10, 6))
plt.hist(df_laptop['price_myr'], bins=50, alpha=0.7, color='skyblue')
plt.title('Laptop Price Distribution')
plt.xlabel('Price (RM)')
plt.ylabel('Frequency')
plt.show()

# Rating distribution
plt.figure(figsize=(10, 6))
plt.hist(df_laptop['average_rating'], bins=20, alpha=0.7, color='lightgreen')
plt.title('Laptop Rating Distribution')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.show()

# Brand analysis
brand_counts = df_laptop['brand'].value_counts().head(10)
plt.figure(figsize=(12, 6))
brand_counts.plot(kind='bar', color='coral')
plt.title('Top 10 Laptop Brands')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 📊 Advanced Analytics

### 1. **Recommendation Quality Analysis**
```python
# Analyze recommendation quality
def analyze_recommendation_quality(recommendations, preferences):
    quality_metrics = {
        'avg_rating': np.mean([r.get('rating', 0) for r in recommendations]),
        'price_within_budget': sum(1 for r in recommendations if r.get('price_myr', 0) <= preferences.get('max_price', 10000)),
        'avg_price': np.mean([r.get('price_myr', 0) for r in recommendations]),
        'diversity_score': len(set(r.get('brand', 'Unknown') for r in recommendations)) / len(recommendations)
    }
    return quality_metrics

# Test different preference scenarios
scenarios = [
    {'search_terms': ['gaming'], 'min_rating': 4.0, 'max_price': 8000},
    {'search_terms': ['student'], 'min_rating': 3.5, 'max_price': 3000},
    {'search_terms': ['business'], 'min_rating': 4.0, 'max_price': 6000}
]

for i, scenario in enumerate(scenarios, 1):
    recs = recommender.get_content_based_recommendations(scenario, 10)
    quality = analyze_recommendation_quality(recs, scenario)
    print(f"Scenario {i}: {quality}")
```

### 2. **User Behavior Analysis**
```python
# Analyze user rating patterns
user_rating_stats = df_rating.groupby('user_id_encoded')['rating'].agg([
    'count', 'mean', 'std'
]).reset_index()

print("User Rating Statistics:")
print(f"Average ratings per user: {user_rating_stats['count'].mean():.1f}")
print(f"Average user rating: {user_rating_stats['mean'].mean():.2f}")
print(f"Most active user: {user_rating_stats['count'].max()} ratings")

# Plot user activity distribution
plt.figure(figsize=(10, 6))
plt.hist(user_rating_stats['count'], bins=30, alpha=0.7, color='purple')
plt.title('User Activity Distribution')
plt.xlabel('Number of Ratings per User')
plt.ylabel('Number of Users')
plt.show()
```

## 🚀 Tips for Jupyter Notebook Usage

### 1. **Use Magic Commands**
```python
# Enable automatic reloading
%load_ext autoreload
%autoreload 2

# Measure execution time
%time recommender.get_content_based_recommendations(gaming_preferences, 10)

# Profile memory usage
%memit recommender.initialize_recommendation_engines()
```

### 2. **Save and Load Results**
```python
# Save evaluation results
import pickle

with open('evaluation_results.pkl', 'wb') as f:
    pickle.dump(evaluation_results, f)

# Load results later
with open('evaluation_results.pkl', 'rb') as f:
    loaded_results = pickle.load(f)
```

### 3. **Export Visualizations**
```python
# Save plots
plt.savefig('evaluation_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('evaluation_comparison.pdf', bbox_inches='tight')
```

### 4. **Create Interactive Widgets**
```python
# Use ipywidgets for interactive analysis
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

def analyze_recommendations(search_terms, min_rating, max_price, n_recs):
    preferences = {
        'search_terms': search_terms.split(','),
        'min_rating': min_rating,
        'max_price': max_price
    }
    recs = recommender.get_content_based_recommendations(preferences, n_recs)
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec.get('title', 'Unknown')} - RM {rec.get('price_myr', 0):.0f}")

interact(analyze_recommendations,
         search_terms='gaming,performance',
         min_rating=(1.0, 5.0, 0.1),
         max_price=(1000, 20000, 500),
         n_recs=(1, 20, 1))
```

## 📚 Additional Resources

- **Complete Demo**: `jupyter_notebook_demo.py`
- **Documentation**: `EVALUATION_SYSTEM_README.md`
- **Web Interface**: Run `python app.py` and visit `/analytics`
- **Evaluation System**: `demo_evaluation_system.py`

## 🎉 Happy Analyzing!

The Jupyter notebook environment provides an excellent platform for interactive analysis of your laptop recommender system. You can experiment with different parameters, visualize results, and gain insights into your recommendation algorithms.

Remember to save your notebook regularly and consider using version control (Git) to track your analysis experiments!
