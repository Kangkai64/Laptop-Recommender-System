"""
Model Training Module for Laptop Price Prediction
Trains multiple ML models and compares their performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class to handle model training and evaluation"""
    
    def __init__(self, models_dir: str = "models", results_dir: str = "results"):
        """
        Initialize ModelTrainer
        
        Args:
            models_dir (str): Directory to save trained models
            results_dir (str): Directory to save results
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.models = {}
        self.results = {}
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different ML models"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for training by splitting into train and test sets
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train a specific model
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Any: Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        logger.info(f"Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        logger.info(f"{model_name} training completed")
        return model
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return metrics
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """
        Train all models and evaluate their performance
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Dict[str, Dict]: Results for all models
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        results = {}
        
        for model_name in self.models.keys():
            try:
                # Train model
                model = self.train_model(model_name, X_train, y_train)
                
                # Evaluate model
                metrics = self.evaluate_model(model, X_test, y_test)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': model.predict(X_test)
                }
                
                logger.info(f"{model_name} - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.2f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def hyperparameter_tuning(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            model_name (str): Name of the model to tune
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Any: Best model with tuned hyperparameters
        """
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No hyperparameter grid defined for {model_name}")
            return self.models[model_name]
        
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best R2 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_model(self, model: Any, model_name: str) -> str:
        """
        Save a trained model to disk
        
        Args:
            model: Trained model to save
            model_name (str): Name for the model file
            
        Returns:
            str: Path to saved model
        """
        filename = f"{model_name.replace(' ', '_').lower()}.joblib"
        filepath = os.path.join(self.models_dir, filename)
        
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a trained model from disk
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            Any: Loaded model
        """
        filename = f"{model_name.replace(' ', '_').lower()}.joblib"
        filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def plot_results(self):
        """Plot comparison of model results"""
        if not self.results:
            logger.warning("No results to plot. Train models first.")
            return
        
        # Prepare data for plotting
        model_names = []
        r2_scores = []
        rmse_scores = []
        
        for model_name, result in self.results.items():
            if 'metrics' in result:
                model_names.append(model_name)
                r2_scores.append(result['metrics']['R2'])
                rmse_scores.append(result['metrics']['RMSE'])
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R2 scores
        ax1.bar(model_names, r2_scores, color='skyblue')
        ax1.set_title('R² Scores Comparison')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # RMSE scores
        ax2.bar(model_names, rmse_scores, color='lightcoral')
        ax2.set_title('RMSE Scores Comparison')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Results plot saved to {plot_path}")
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on R² score
        
        Returns:
            Tuple[str, Any]: Name and model of the best performer
        """
        if not self.results:
            raise ValueError("No models have been trained yet")
        
        best_model_name = None
        best_r2 = -float('inf')
        best_model = None
        
        for model_name, result in self.results.items():
            if 'metrics' in result:
                r2_score = result['metrics']['R2']
                if r2_score > best_r2:
                    best_r2 = r2_score
                    best_model_name = model_name
                    best_model = result['model']
        
        return best_model_name, best_model

def main():
    """Main function to run model training"""
    from data_loader import DataLoader
    from data_preprocessing import DataPreprocessor
    
    try:
        # Load and preprocess data
        loader = DataLoader()
        data = loader.load_data()
        
        # Get target variable
        target_col = loader.get_target_variable()
        print(f"Target variable: {target_col}")
        
        # Split features and target
        X, y = loader.split_features_target(target_col)
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_processed, feature_names = preprocessor.create_preprocessing_pipeline(X)
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train all models
        results = trainer.train_all_models(X_processed, y)
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model()
        print(f"\nBest model: {best_model_name}")
        
        # Save best model
        trainer.save_model(best_model, best_model_name)
        
        # Plot results
        trainer.plot_results()
        
        # Print summary
        print("\nModel Performance Summary:")
        print("-" * 50)
        for model_name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"{model_name}:")
                print(f"  R² Score: {metrics['R2']:.4f}")
                print(f"  RMSE: {metrics['RMSE']:.2f}")
                print(f"  MAE: {metrics['MAE']:.2f}")
                print()
        
    except Exception as e:
        print(f"Error during model training: {e}")

if __name__ == "__main__":
    main() 