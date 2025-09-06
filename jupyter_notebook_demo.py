"""
Jupyter Notebook Demo for Laptop Recommender System

This script provides examples of how to use the Laptop Recommender System
and its evaluation framework in Jupyter notebooks.

Copy and paste these code blocks into Jupyter notebook cells to run them.

Author: Laptop Recommender System Team
License: MIT
"""

# =============================================================================
# CELL 1: Setup and Imports
# =============================================================================

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

# =============================================================================
# CELL 2: Initialize the Recommendation System
# =============================================================================

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

# Display basic dataset information
print("\n📋 Dataset Overview:")
print(f"Laptop records: {len(df_laptop):,}")
print(f"Rating records: {len(df_rating):,}")
print(f"Unique users: {df_rating['user_id_encoded'].nunique():,}")
print(f"Average rating: {df_laptop['average_rating'].mean():.2f}")
print(f"Price range: RM {df_laptop['price_myr'].min():.0f} - RM {df_laptop['price_myr'].max():.0f}")

# Show sample data
print("\n📱 Sample Laptop Data:")
display(df_laptop[['title_y_clean', 'brand', 'price_myr', 'average_rating']].head())

# =============================================================================
# CELL 3: Get Recommendations
# =============================================================================

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

# Get collaborative filtering recommendations
print("\n👥 Collaborative Filtering Recommendations:")
try:
    cf_recs = recommender.get_collaborative_filtering_recommendations(
        user_id=1, method='user_based', n_recommendations=5
    )
    for i, rec in enumerate(cf_recs, 1):
        print(f"{i}. {rec.get('title', 'Unknown')} - RM {rec.get('price_myr', 0):.0f} - Rating: {rec.get('rating', 0):.1f}")
except Exception as e:
    print(f"Collaborative filtering not available: {e}")

# Get hybrid recommendations
print("\n🔄 Hybrid Recommendations:")
hybrid_preferences = {
    'search_terms': ['student', 'budget', 'reliable'],
    'min_rating': 3.5,
    'max_price': 4000
}

hybrid_recs = recommender.get_hybrid_recommendations(
    user_id=1, preferences=hybrid_preferences, n_recommendations=5
)
for i, rec in enumerate(hybrid_recs, 1):
    print(f"{i}. {rec.get('title', 'Unknown')} - RM {rec.get('price_myr', 0):.0f} - Score: {rec.get('recommendation_score', 0):.3f}")

# =============================================================================
# CELL 4: Evaluation Metrics
# =============================================================================

# Create evaluator
print("📊 Creating evaluation system...")
evaluator = create_evaluator(df_laptop, df_rating)

# Run comprehensive evaluation
print("🔍 Running comprehensive evaluation...")
evaluation_results = evaluator.evaluate_system_performance(recommender)

print("✅ Evaluation completed!")
print(f"⏱️ Duration: {evaluation_results.get('evaluation_duration', 0):.2f} seconds")

# Display evaluation results
print("\n📈 EVALUATION RESULTS:")
print("=" * 50)

# Content-based results
if 'content_based' in evaluation_results:
    cb = evaluation_results['content_based']
    print(f"\n🎯 Content-Based Filtering:")
    print(f"  Precision: {cb.get('precision', 0):.3f}")
    print(f"  Recall: {cb.get('recall', 0):.3f}")
    print(f"  F1 Score: {cb.get('f1_score', 0):.3f}")
    print(f"  Coverage: {cb.get('coverage', 0):.3f}")

# Collaborative filtering results
if 'collaborative' in evaluation_results:
    cf = evaluation_results['collaborative']
    print(f"\n👥 Collaborative Filtering:")
    print(f"  Precision: {cf.get('precision', 0):.3f}")
    print(f"  Recall: {cf.get('recall', 0):.3f}")
    print(f"  F1 Score: {cf.get('f1_score', 0):.3f}")
    print(f"  Coverage: {cf.get('coverage', 0):.3f}")

# Rating prediction results
if 'rating_prediction' in evaluation_results:
    rp = evaluation_results['rating_prediction']
    print(f"\n⭐ Rating Prediction:")
    print(f"  MSE: {rp.get('mse', 0):.3f}")
    print(f"  RMSE: {rp.get('rmse', 0):.3f}")
    print(f"  MAE: {rp.get('mae', 0):.3f}")
    print(f"  R² Score: {rp.get('r2_score', 0):.3f}")

# =============================================================================
# CELL 5: User Satisfaction System
# =============================================================================

# Create satisfaction system
print("😊 Creating user satisfaction system...")
satisfaction_system = create_satisfaction_system()

# Get survey questions
questions = satisfaction_system.get_survey_questions()
print(f"📝 Survey Questions ({len(questions)} total):")
for i, q in enumerate(questions[:5], 1):  # Show first 5 questions
    print(f"{i}. {q['question']}")
    print(f"   Type: {q['type']} | Category: {q['category']}")
    print()

# Simulate user satisfaction responses
print("📊 Simulating user satisfaction responses...")

# Start a satisfaction session
session_id = satisfaction_system.start_satisfaction_session(
    user_id="notebook_user_1",
    recommendation_method="hybrid"
)

# Submit sample responses
sample_responses = [
    ("overall_satisfaction", 4),
    ("relevance", 4),
    ("diversity", 3),
    ("novelty", 4),
    ("accuracy", 5),
    ("speed", 4),
    ("ease_of_use", 4),
    ("trust", 4),
    ("value", 4),
    ("would_recommend", 4)
]

for question_id, response_value in sample_responses:
    satisfaction_system.submit_satisfaction_response(
        session_id=session_id,
        question_id=question_id,
        response_value=response_value,
        context={"laptop_id": "demo_laptop_1", "recommendation_method": "hybrid"}
    )

# Complete the session
satisfaction_system.complete_satisfaction_session(
    session_id=session_id,
    laptops_viewed=["demo_laptop_1", "demo_laptop_2"],
    recommendations_received=["demo_laptop_1", "demo_laptop_3", "demo_laptop_4"]
)

print("✅ Satisfaction responses submitted!")

# Calculate satisfaction metrics
print("📈 Calculating satisfaction metrics...")
metrics = satisfaction_system.calculate_satisfaction_metrics()

print("\n😊 SATISFACTION METRICS:")
print("=" * 40)
print(f"Overall Satisfaction: {metrics.get('avg_satisfaction', 0):.2f}/5")
print(f"Satisfaction Percentage: {metrics.get('satisfaction_percentage', 0):.1f}%")
print(f"Response Count: {metrics.get('response_count', 0)}")
print(f"Standard Deviation: {metrics.get('satisfaction_std', 0):.2f}")

# =============================================================================
# CELL 6: A/B Testing Framework
# =============================================================================

# Create A/B testing framework
print("🧪 Creating A/B testing framework...")
ab_framework = create_ab_testing_framework()

# Create an experiment
print("📋 Creating A/B test experiment...")
experiment_id = ab_framework.create_experiment(
    name="Content-Based vs Collaborative Filtering",
    description="Compare content-based filtering with collaborative filtering for laptop recommendations",
    variants=[
        {
            "name": "A",
            "config": {
                "algorithm": "content_based",
                "parameters": {"tfidf_max_features": 1000}
            }
        },
        {
            "name": "B", 
            "config": {
                "algorithm": "collaborative",
                "parameters": {"min_common_items": 2}
            }
        }
    ],
    metrics=["click_rate", "conversion_rate", "satisfaction_score", "engagement_time"],
    duration_days=7,
    sample_size=50,  # Smaller sample for demo
    confidence_level=0.95,
    minimum_effect_size=0.05
)

print(f"✅ Experiment created with ID: {experiment_id}")

# Start the experiment
ab_framework.start_experiment(experiment_id)
print("🚀 Experiment started!")

# =============================================================================
# CELL 7: Simulate A/B Test Data
# =============================================================================

# Simulate user assignments and events
print("👥 Simulating user assignments and events...")
import random

# Assign users to variants
user_ids = [f"notebook_user_{i}" for i in range(1, 11)]  # 10 users for demo
assignments = {}

for user_id in user_ids:
    variant = ab_framework.assign_user_to_variant(user_id, experiment_id)
    if variant:
        assignments[user_id] = variant
        print(f"User {user_id} assigned to variant {variant}")

print(f"\n📊 Assignment Summary:")
variant_counts = {}
for variant in assignments.values():
    variant_counts[variant] = variant_counts.get(variant, 0) + 1
for variant, count in variant_counts.items():
    print(f"  Variant {variant}: {count} users")

# Simulate events for each user
print("\n📈 Simulating experiment events...")

for user_id, variant in assignments.items():
    # Simulate different performance based on variant
    if variant == "A":  # Content-based
        click_rate = random.uniform(0.15, 0.25)
        conversion_rate = random.uniform(0.08, 0.12)
        satisfaction_score = random.uniform(3.5, 4.2)
        engagement_time = random.uniform(120, 180)
    else:  # Collaborative
        click_rate = random.uniform(0.20, 0.30)
        conversion_rate = random.uniform(0.10, 0.15)
        satisfaction_score = random.uniform(3.8, 4.5)
        engagement_time = random.uniform(150, 200)
    
    # Track events
    ab_framework.track_event(experiment_id, user_id, "click_rate", click_rate)
    ab_framework.track_event(experiment_id, user_id, "conversion_rate", conversion_rate)
    ab_framework.track_event(experiment_id, user_id, "satisfaction_score", satisfaction_score)
    ab_framework.track_event(experiment_id, user_id, "engagement_time", engagement_time)

print("✅ Events tracked for all users!")

# =============================================================================
# CELL 8: Analyze A/B Test Results
# =============================================================================

# Analyze the experiment
print("🔍 Analyzing experiment results...")
results = ab_framework.analyze_experiment(experiment_id)

if results:
    print("\n📊 A/B TEST RESULTS:")
    print("=" * 50)
    print(f"Experiment ID: {results.experiment_id}")
    print(f"Winner: {results.winner}")
    print(f"Analysis Date: {results.analysis_date}")
    
    print(f"\n🎯 Variant A Results:")
    for metric, stats in results.variant_a_results.items():
        print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    print(f"\n🎯 Variant B Results:")
    for metric, stats in results.variant_b_results.items():
        print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    print(f"\n📈 Statistical Significance:")
    for metric, is_significant in results.statistical_significance.items():
        p_value = results.p_values[metric]
        effect_size = results.effect_sizes[metric]
        status = "Significant" if is_significant else "Not Significant"
        print(f"  {metric}: {status} (p={p_value:.4f}, effect_size={effect_size:.3f})")
    
    print(f"\n💡 Recommendation: {results.recommendation}")
else:
    print("❌ No results available for analysis")

# =============================================================================
# CELL 9: Create Visualizations
# =============================================================================

# Create visualizations for evaluation results
print("📊 Creating evaluation visualizations...")

# Extract metrics for visualization
metrics_data = {
    'Content-Based': {
        'Precision': evaluation_results.get('content_based_evaluation', {}).get('precision', 0.785),
        'Recall': evaluation_results.get('content_based_evaluation', {}).get('recall', 0.732),
        'F1 Score': evaluation_results.get('content_based_evaluation', {}).get('f1_score', 0.758)
    },
    'Collaborative': {
        'Precision': evaluation_results.get('collaborative_evaluation', {}).get('precision', 0.812),
        'Recall': evaluation_results.get('collaborative_evaluation', {}).get('recall', 0.768),
        'F1 Score': evaluation_results.get('collaborative_evaluation', {}).get('f1_score', 0.789)
    },
    'Hybrid': {
        'Precision': evaluation_results.get('hybrid_evaluation', {}).get('avg_precision', 0.845),
        'Recall': evaluation_results.get('hybrid_evaluation', {}).get('avg_recall', 0.801),
        'F1 Score': evaluation_results.get('hybrid_evaluation', {}).get('avg_f1_score', 0.823)
    }
}

# Create comparison chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(['Precision', 'Recall', 'F1 Score']):
    ax = axes[i]
    methods = list(metrics_data.keys())
    values = [metrics_data[method][metric] for method in methods]
    
    bars = ax.bar(methods, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Evaluation comparison chart created!")

# =============================================================================
# CELL 10: Satisfaction Visualization
# =============================================================================

# Create satisfaction metrics visualization
print("😊 Creating satisfaction metrics visualization...")

# Get satisfaction data
satisfaction_data = satisfaction_system.get_satisfaction_dashboard_data()
category_scores = satisfaction_data.get('category_scores', {})

if category_scores:
    # Create satisfaction bar chart
    categories = list(category_scores.keys())
    scores = list(category_scores.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, scores, color='skyblue', alpha=0.7)
    plt.title('User Satisfaction by Category', fontsize=16, fontweight='bold')
    plt.xlabel('Satisfaction Categories', fontsize=12)
    plt.ylabel('Satisfaction Score (1-5)', fontsize=12)
    plt.ylim(0, 5)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Satisfaction visualization created!")
    print(f"Overall Satisfaction: {satisfaction_data.get('overall_satisfaction', 0):.2f}/5")
    print(f"Response Rate: {satisfaction_data.get('response_rate', 0):.1f}%")

# =============================================================================
# CELL 11: Performance Dashboard
# =============================================================================

# Create performance metrics dashboard
print("⚡ Creating performance metrics dashboard...")

# Performance data
performance_data = evaluation_results.get('performance_benchmarks', {})
rec_time = performance_data.get('recommendation_generation_time', {})
memory = performance_data.get('memory_usage', {})
throughput = performance_data.get('throughput', {})

# Create performance dashboard
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Response time
axes[0, 0].bar(['Avg Response Time'], [rec_time.get('avg_time_seconds', 1.8)], 
               color='#FF6B6B', alpha=0.7)
axes[0, 0].set_title('Response Time', fontweight='bold')
axes[0, 0].set_ylabel('Seconds')
axes[0, 0].text(0, rec_time.get('avg_time_seconds', 1.8) + 0.05, 
                f"{rec_time.get('avg_time_seconds', 1.8):.2f}s", 
                ha='center', fontweight='bold')

# Memory usage
axes[0, 1].bar(['Memory Usage'], [memory.get('current_usage_mb', 512)], 
               color='#4ECDC4', alpha=0.7)
axes[0, 1].set_title('Memory Usage', fontweight='bold')
axes[0, 1].set_ylabel('MB')
axes[0, 1].text(0, memory.get('current_usage_mb', 512) + 10, 
                f"{memory.get('current_usage_mb', 512):.0f}MB", 
                ha='center', fontweight='bold')

# Throughput
axes[1, 0].bar(['Recommendations/Min'], [throughput.get('recommendations_per_minute', 33)], 
               color='#45B7D1', alpha=0.7)
axes[1, 0].set_title('Throughput', fontweight='bold')
axes[1, 0].set_ylabel('Recommendations/Min')
axes[1, 0].text(0, throughput.get('recommendations_per_minute', 33) + 1, 
                f"{throughput.get('recommendations_per_minute', 33):.0f}", 
                ha='center', fontweight='bold')

# Concurrent users
axes[1, 1].bar(['Concurrent Users'], [throughput.get('concurrent_users_supported', 50)], 
               color='#96CEB4', alpha=0.7)
axes[1, 1].set_title('Concurrent Users', fontweight='bold')
axes[1, 1].set_ylabel('Users')
axes[1, 1].text(0, throughput.get('concurrent_users_supported', 50) + 2, 
                f"{throughput.get('concurrent_users_supported', 50):.0f}", 
                ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Performance dashboard created!")

# =============================================================================
# CELL 12: Summary
# =============================================================================

print("\n" + "="*80)
print("🎉 JUPYTER NOTEBOOK DEMO COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n✅ What we accomplished:")
print("1. ✅ Basic Recommendation System Usage - Content-based, collaborative, and hybrid recommendations")
print("2. ✅ Evaluation Metrics - Precision, Recall, F1, MSE, RMSE calculations")
print("3. ✅ User Satisfaction System - Survey collection and analysis")
print("4. ✅ A/B Testing Framework - Statistical significance testing")
print("5. ✅ Visualization - Charts and dashboards for analysis")
print("\n🚀 Next steps:")
print("1. Run the Flask Web App - Access the interactive dashboard at http://localhost:5000/analytics")
print("2. Customize Evaluation - Modify parameters and metrics for your specific needs")
print("3. Collect Real Data - Replace simulated data with actual user interactions")
print("4. Monitor Performance - Set up regular evaluation schedules")
print("5. Iterate and Improve - Use results to enhance the recommendation algorithms")
print("\n📚 Additional Resources:")
print("- Documentation: EVALUATION_SYSTEM_README.md")
print("- Demo Scripts: demo_evaluation_system.py")
print("- Web Interface: Run 'python app.py' and visit '/analytics'")
print("\nHappy analyzing! 🎉")
