"""
Model Evaluation Module for Laptop Price Prediction
Provides comprehensive evaluation metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.model_selection import cross_val_score
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class to handle comprehensive model evaluation"""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize ModelEvaluator
        
        Args:
            results_dir (str): Directory to save evaluation results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics['Explained_Variance'] = explained_variance_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        try:
            metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            metrics['MAPE'] = np.nan
        
        # Mean Absolute Error Percentage
        metrics['MAE_Percentage'] = (metrics['MAE'] / np.mean(y_true)) * 100
        
        # Root Mean Square Error Percentage
        metrics['RMSE_Percentage'] = (metrics['RMSE'] / np.mean(y_true)) * 100
        
        return metrics
    
    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                           cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on a model
        
        Args:
            model: Trained model
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Cross-validation results
        """
        cv_results = {}
        
        # R2 score
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        cv_results['R2_Mean'] = r2_scores.mean()
        cv_results['R2_Std'] = r2_scores.std()
        
        # RMSE score
        rmse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        cv_results['RMSE_Mean'] = -rmse_scores.mean()
        cv_results['RMSE_Std'] = rmse_scores.std()
        
        # MAE score
        mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        cv_results['MAE_Mean'] = -mae_scores.mean()
        cv_results['MAE_Std'] = mae_scores.std()
        
        return cv_results
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  model_name: str = "Model", save_plot: bool = True):
        """
        Plot predicted vs actual values
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Name of the model
            save_plot (bool): Whether to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R2 for the plot
        r2 = r2_score(y_true, y_pred)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Predicted vs Actual Values\nR² = {r2:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plot_path = os.path.join(self.results_dir, f'{model_name.lower().replace(" ", "_")}_predictions.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {plot_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "Model", save_plot: bool = True):
        """
        Plot residuals analysis
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Name of the model
            save_plot (bool): Whether to save the plot
        """
        residuals = y_true - y_pred
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot of Residuals')
        ax3.grid(True, alpha=0.3)
        
        # Residuals vs Index
        ax4.plot(range(len(residuals)), residuals, alpha=0.6)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Index')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals vs Index')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.results_dir, f'{model_name.lower().replace(" ", "_")}_residuals.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residuals plot saved to {plot_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                              model_name: str = "Model", save_plot: bool = True):
        """
        Plot feature importance (for tree-based models)
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names (List[str]): List of feature names
            model_name (str): Name of the model
            save_plot (bool): Whether to save the plot
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                logger.warning(f"Model {model_name} doesn't have feature importance")
                return
            
            # Create DataFrame for easier plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True)
            
            plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
            plt.barh(range(len(importance_df)), importance_df['Importance'])
            plt.yticks(range(len(importance_df)), importance_df['Feature'])
            plt.xlabel('Importance')
            plt.title(f'{model_name} - Feature Importance')
            plt.grid(True, alpha=0.3)
            
            if save_plot:
                plot_path = os.path.join(self.results_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
    
    def create_evaluation_report(self, model_results: Dict[str, Dict], 
                               save_report: bool = True) -> str:
        """
        Create a comprehensive evaluation report
        
        Args:
            model_results (Dict[str, Dict]): Results from model training
            save_report (bool): Whether to save the report
            
        Returns:
            str: Report content
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("LAPTOP PRICE PREDICTION - MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Summary table
        report_lines.append("MODEL PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Model':<25} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'MAPE':<8}")
        report_lines.append("-" * 40)
        
        best_model = None
        best_r2 = -float('inf')
        
        for model_name, result in model_results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                r2 = metrics.get('R2', 0)
                rmse = metrics.get('RMSE', 0)
                mae = metrics.get('MAE', 0)
                mape = metrics.get('MAPE', 0)
                
                report_lines.append(f"{model_name:<25} {r2:<8.4f} {rmse:<10.2f} {mae:<10.2f} {mape:<8.2f}")
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_name
        
        report_lines.append("-" * 40)
        report_lines.append(f"Best Model: {best_model} (R² = {best_r2:.4f})")
        report_lines.append("")
        
        # Detailed metrics for each model
        for model_name, result in model_results.items():
            if 'metrics' in result:
                report_lines.append(f"DETAILED METRICS - {model_name.upper()}")
                report_lines.append("-" * 40)
                
                metrics = result['metrics']
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"{metric_name}: {value:.4f}")
                    else:
                        report_lines.append(f"{metric_name}: {value}")
                
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if save_report:
            report_path = os.path.join(self.results_dir, 'evaluation_report.txt')
            with open(report_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Evaluation report saved to {report_path}")
        
        return report_content
    
    def evaluate_model_comprehensive(self, model: Any, X_test: pd.DataFrame, 
                                   y_test: pd.Series, model_name: str = "Model",
                                   feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of a single model
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            feature_names (List[str]): List of feature names
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test.values, y_pred)
        
        # Cross-validation
        cv_results = self.cross_validate_model(model, X_test, y_test)
        
        # Create plots
        self.plot_predictions_vs_actual(y_test.values, y_pred, model_name)
        self.plot_residuals(y_test.values, y_pred, model_name)
        
        if feature_names:
            self.plot_feature_importance(model, feature_names, model_name)
        
        # Compile results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'cv_results': cv_results,
            'predictions': y_pred,
            'actual': y_test.values
        }
        
        logger.info(f"Comprehensive evaluation completed for {model_name}")
        return results

def main():
    """Main function to test the ModelEvaluator"""
    from data_loader import DataLoader
    from data_preprocessing import DataPreprocessor
    from model_training import ModelTrainer
    
    try:
        # Load and preprocess data
        loader = DataLoader()
        data = loader.load_data()
        
        # Get target variable
        target_col = loader.get_target_variable()
        X, y = loader.split_features_target(target_col)
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_processed, feature_names = preprocessor.create_preprocessing_pipeline(X)
        
        # Train models
        trainer = ModelTrainer()
        results = trainer.train_all_models(X_processed, y)
        
        # Evaluate models
        evaluator = ModelEvaluator()
        
        # Create evaluation report
        report = evaluator.create_evaluation_report(results)
        print(report)
        
        # Comprehensive evaluation of best model
        best_model_name, best_model = trainer.get_best_model()
        print(f"\nPerforming comprehensive evaluation of {best_model_name}...")
        
        # Get test data for evaluation
        X_train, X_test, y_train, y_test = trainer.prepare_data(X_processed, y)
        
        comprehensive_results = evaluator.evaluate_model_comprehensive(
            best_model, X_test, y_test, best_model_name, feature_names
        )
        
        print(f"\nComprehensive evaluation completed for {best_model_name}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main() 