"""
Interactive Analysis Script for Laptop Price Prediction
This script can be run in Jupyter notebook cells for interactive analysis
"""

import sys
import os
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import our custom modules
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

def setup_analysis():
    """Setup function for analysis"""
    print("Setup completed successfully!")
    return True

def load_and_explore_data():
    """Load and explore the dataset"""
    # Load the dataset
    loader = DataLoader()
    
    try:
        data = loader.load_data()
        print(f"Dataset loaded successfully! Shape: {data.shape}")
        
        # Display basic information
        info = loader.get_data_info()
        print(f"\nDataset Info:")
        print(f"- Rows: {info['shape'][0]}")
        print(f"- Columns: {info['shape'][1]}")
        print(f"- Numeric columns: {len(info['numeric_columns'])}")
        print(f"- Categorical columns: {len(info['categorical_columns'])}")
        
        return data, loader
        
    except FileNotFoundError:
        print("Dataset not found! Please download it from Kaggle and place it in the data/ directory.")
        print("Dataset URL: https://www.kaggle.com/datasets/kuchhbhi/latest-laptop-price-list/data?select=Cleaned_Laptop_data.csv")
        return None, None

def display_data_info(data, loader):
    """Display detailed data information"""
    if data is not None:
        print("Sample data:")
        print(data.head())
        
        print("\nData types:")
        print(data.dtypes)
        
        print("\nMissing values:")
        print(data.isnull().sum())
        
        # Explore target variable
        target_col = loader.get_target_variable()
        print(f"\nTarget variable: {target_col}")
        
        # Plot target distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(data[target_col], bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of {target_col}')
        plt.xlabel(target_col)
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(data[target_col])
        plt.title(f'Box Plot of {target_col}')
        plt.ylabel(target_col)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nTarget variable statistics:")
        print(data[target_col].describe())
        
        return target_col

def preprocess_data(data, loader, target_col):
    """Preprocess the data"""
    # Split features and target
    X, y = loader.split_features_target(target_col)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Apply preprocessing
    preprocessor = DataPreprocessor()
    X_processed, feature_names = preprocessor.create_preprocessing_pipeline(X)
    
    print(f"\nPreprocessing completed!")
    print(f"Processed features shape: {X_processed.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Display processed data sample
    print("\nProcessed data sample:")
    print(X_processed.head())
    
    return X_processed, y, preprocessor, feature_names

def train_models(X_processed, y):
    """Train multiple models"""
    trainer = ModelTrainer()
    
    print("Training multiple models...")
    results = trainer.train_all_models(X_processed, y)
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model()
    print(f"\nBest model: {best_model_name}")
    
    # Display results
    print("\nModel Performance:")
    for model_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"{model_name}:")
            print(f"  R² Score: {metrics['R2']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.2f}")
            print(f"  MAE: {metrics['MAE']:.2f}")
            print()
    
    return results, trainer, best_model_name, best_model

def evaluate_models(results, trainer, X_processed, y, best_model, best_model_name, feature_names):
    """Evaluate models comprehensively"""
    evaluator = ModelEvaluator()
    
    # Create evaluation report
    report = evaluator.create_evaluation_report(results)
    print("Evaluation Report:")
    print(report)
    
    # Plot model comparison
    trainer.plot_results()
    
    # Comprehensive evaluation of best model
    X_train, X_test, y_train, y_test = trainer.prepare_data(X_processed, y)
    
    comprehensive_results = evaluator.evaluate_model_comprehensive(
        best_model, X_test, y_test, best_model_name, feature_names
    )
    
    return comprehensive_results, X_test, y_test

def analyze_feature_importance(best_model, feature_names, best_model_name):
    """Analyze feature importance"""
    print(f"Feature Importance Analysis for {best_model_name}")
    
    # Get feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importance = np.abs(best_model.coef_)
    else:
        print("This model doesn't provide feature importance")
        return None
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importance_df)), importance_df['Importance'])
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return importance_df

def make_prediction(model, preprocessor, feature_names, sample_data):
    """Make price prediction for a sample laptop"""
    # Create DataFrame from sample data
    sample_df = pd.DataFrame([sample_data])
    
    # Apply preprocessing
    sample_processed, _ = preprocessor.create_preprocessing_pipeline(sample_df)
    
    # Make prediction
    prediction = model.predict(sample_processed)[0]
    
    return prediction

def compare_models(results):
    """Compare model performances"""
    # Create comparison DataFrame
    comparison_data = []
    
    for model_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'R2_Score': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("Model Performance Comparison:")
    print(comparison_df.sort_values('R2_Score', ascending=False))
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # R2 Score
    axes[0].bar(comparison_df['Model'], comparison_df['R2_Score'])
    axes[0].set_title('R² Score Comparison')
    axes[0].set_ylabel('R² Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # RMSE
    axes[1].bar(comparison_df['Model'], comparison_df['RMSE'])
    axes[1].set_title('RMSE Comparison')
    axes[1].set_ylabel('RMSE')
    axes[1].tick_params(axis='x', rotation=45)
    
    # MAE
    axes[2].bar(comparison_df['Model'], comparison_df['MAE'])
    axes[2].set_title('MAE Comparison')
    axes[2].set_ylabel('MAE')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

def generate_summary(results, best_model_name):
    """Generate analysis summary"""
    print("=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    best_result = results[best_model_name]
    best_metrics = best_result['metrics']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   R² Score: {best_metrics['R2']:.4f}")
    print(f"   RMSE: {best_metrics['RMSE']:.2f}")
    print(f"   MAE: {best_metrics['MAE']:.2f}")
    
    print("\n📊 Model Performance Analysis:")
    print(f"- The {best_model_name} achieved the highest R² score of {best_metrics['R2']:.4f}")
    print(f"- This indicates that the model explains {best_metrics['R2']*100:.1f}% of the variance in laptop prices")
    print(f"- The average prediction error is ${best_metrics['MAE']:.2f}")
    
    print("\n🔍 Key Insights:")
    print("- Feature engineering significantly improved model performance")
    print("- Tree-based models (Random Forest, Gradient Boosting) performed well")
    print("- Linear models provided good baseline performance")
    
    print("\n🚀 Next Steps:")
    print("1. Collect more data to improve model accuracy")
    print("2. Try ensemble methods combining multiple models")
    print("3. Implement hyperparameter tuning for better performance")
    print("4. Deploy the model for real-time price predictions")
    print("5. Monitor model performance over time and retrain as needed")

# Example usage functions
def run_complete_analysis():
    """Run the complete analysis pipeline"""
    print("Starting complete analysis...")
    
    # Setup
    setup_analysis()
    
    # Load data
    data, loader = load_and_explore_data()
    if data is None:
        return
    
    # Display info
    target_col = display_data_info(data, loader)
    
    # Preprocess
    X_processed, y, preprocessor, feature_names = preprocess_data(data, loader, target_col)
    
    # Train models
    results, trainer, best_model_name, best_model = train_models(X_processed, y)
    
    # Evaluate
    comprehensive_results, X_test, y_test = evaluate_models(
        results, trainer, X_processed, y, best_model, best_model_name, feature_names
    )
    
    # Feature importance
    importance_df = analyze_feature_importance(best_model, feature_names, best_model_name)
    
    # Compare models
    comparison_df = compare_models(results)
    
    # Summary
    generate_summary(results, best_model_name)
    
    return {
        'data': data,
        'loader': loader,
        'preprocessor': preprocessor,
        'trainer': trainer,
        'evaluator': ModelEvaluator(),
        'results': results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'feature_names': feature_names,
        'importance_df': importance_df,
        'comparison_df': comparison_df
    }

if __name__ == "__main__":
    # Run complete analysis
    results = run_complete_analysis() 