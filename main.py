"""
Main Script for Laptop Price Prediction
Orchestrates the entire ML pipeline from data loading to model evaluation
"""

import os
import sys
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the complete ML pipeline"""
    
    print("=" * 60)
    print("LAPTOP PRICE PREDICTION - MACHINE LEARNING PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Data Loading
        print("Step 1: Loading Data")
        print("-" * 30)
        loader = DataLoader()
        
        try:
            data = loader.load_data()
            print(f"✓ Dataset loaded successfully. Shape: {data.shape}")
            
            # Display dataset info
            info = loader.get_data_info()
            print(f"✓ Dataset has {info['shape'][0]} samples and {info['shape'][1]} features")
            print(f"✓ Target variable: {loader.get_target_variable()}")
            
        except FileNotFoundError:
            print("✗ Dataset not found!")
            print("Please download the dataset from Kaggle:")
            print("https://www.kaggle.com/datasets/kuchhbhi/latest-laptop-price-list/data?select=Cleaned_Laptop_data.csv")
            print("And place it in the 'data/' directory as 'Cleaned_Laptop_data.csv'")
            return
        
        # Step 2: Data Preprocessing
        print("\nStep 2: Data Preprocessing")
        print("-" * 30)
        
        # Get target variable and split data
        target_col = loader.get_target_variable()
        X, y = loader.split_features_target(target_col)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Apply preprocessing pipeline
        X_processed, feature_names = preprocessor.create_preprocessing_pipeline(X)
        print(f"✓ Data preprocessing completed. Final shape: {X_processed.shape}")
        print(f"✓ {len(feature_names)} features prepared for training")
        
        # Step 3: Model Training
        print("\nStep 3: Model Training")
        print("-" * 30)
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train all models
        print("Training multiple models...")
        results = trainer.train_all_models(X_processed, y)
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model()
        print(f"✓ Best model: {best_model_name}")
        
        # Save best model
        model_path = trainer.save_model(best_model, best_model_name)
        print(f"✓ Best model saved to: {model_path}")
        
        # Step 4: Model Evaluation
        print("\nStep 4: Model Evaluation")
        print("-" * 30)
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Create evaluation report
        report = evaluator.create_evaluation_report(results)
        print("✓ Evaluation report generated")
        
        # Comprehensive evaluation of best model
        print(f"\nPerforming comprehensive evaluation of {best_model_name}...")
        X_train, X_test, y_train, y_test = trainer.prepare_data(X_processed, y)
        
        comprehensive_results = evaluator.evaluate_model_comprehensive(
            best_model, X_test, y_test, best_model_name, feature_names
        )
        
        print("✓ Comprehensive evaluation completed")
        
        # Step 5: Results Summary
        print("\nStep 5: Results Summary")
        print("-" * 30)
        
        print("\nModel Performance Summary:")
        print("-" * 50)
        for model_name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"{model_name}:")
                print(f"  R² Score: {metrics['R2']:.4f}")
                print(f"  RMSE: {metrics['RMSE']:.2f}")
                print(f"  MAE: {metrics['MAE']:.2f}")
                if 'MAPE' in metrics and not pd.isna(metrics['MAPE']):
                    print(f"  MAPE: {metrics['MAPE']:.2f}%")
                print()
        
        # Print best model details
        best_result = results[best_model_name]
        best_metrics = best_result['metrics']
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"   R² Score: {best_metrics['R2']:.4f}")
        print(f"   RMSE: {best_metrics['RMSE']:.2f}")
        print(f"   MAE: {best_metrics['MAE']:.2f}")
        
        # Step 6: Generate plots
        print("\nStep 6: Generating Visualizations")
        print("-" * 30)
        
        # Plot model comparison
        trainer.plot_results()
        print("✓ Model comparison plot generated")
        
        # Plot comprehensive evaluation for best model
        evaluator.plot_predictions_vs_actual(
            y_test.values, best_result['predictions'], best_model_name
        )
        evaluator.plot_residuals(
            y_test.values, best_result['predictions'], best_model_name
        )
        evaluator.plot_feature_importance(
            best_model, feature_names, best_model_name
        )
        print("✓ Comprehensive evaluation plots generated")
        
        # Final summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✓ Results saved in 'results/' directory")
        print(f"✓ Best model saved in 'models/' directory")
        print(f"✓ Training log saved as 'training.log'")
        print(f"✓ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Instructions for next steps
        print("\nNext Steps:")
        print("1. Check the 'results/' directory for evaluation reports and plots")
        print("2. Use the trained model for predictions")
        print("3. Explore the Jupyter notebook for interactive analysis")
        print("4. Modify hyperparameters in model_training.py for better performance")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        print("Check the training.log file for detailed error information")
        raise

if __name__ == "__main__":
    main() 