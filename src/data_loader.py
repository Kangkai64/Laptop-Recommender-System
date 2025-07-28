"""
Data Loader Module for Laptop Price Prediction
Handles loading and basic exploration of the dataset
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Class to handle data loading and basic exploration"""
    
    def __init__(self, data_path: str = "data/Cleaned_Laptop_data.csv"):
        """
        Initialize DataLoader
        
        Args:
            data_path (str): Path to the CSV file
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Dataset not found at {self.data_path}. "
                                     f"Please download the dataset from Kaggle and place it in the data/ directory.")
            
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def get_data_info(self) -> dict:
        """
        Get basic information about the dataset
        
        Returns:
            dict: Dictionary containing dataset information
        """
        if self.data is None:
            self.load_data()
            
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': self.data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.data.select_dtypes(include=['object']).columns.tolist()
        }
        
        return info
    
    def display_sample(self, n_samples: int = 5) -> pd.DataFrame:
        """
        Display a sample of the dataset
        
        Args:
            n_samples (int): Number of samples to display
            
        Returns:
            pd.DataFrame: Sample of the dataset
        """
        if self.data is None:
            self.load_data()
            
        return self.data.head(n_samples)
    
    def get_target_variable(self) -> str:
        """
        Identify the target variable (price column)
        
        Returns:
            str: Name of the target variable
        """
        if self.data is None:
            self.load_data()
            
        # Look for price-related columns
        price_columns = [col for col in self.data.columns if 'price' in col.lower()]
        
        if price_columns:
            return price_columns[0]
        else:
            # If no price column found, return the last numeric column
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return numeric_cols[-1]
            else:
                raise ValueError("No suitable target variable found in the dataset")
    
    def split_features_target(self, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split the dataset into features and target
        
        Args:
            target_col (str, optional): Name of the target column. If None, will be auto-detected
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        if self.data is None:
            self.load_data()
            
        if target_col is None:
            target_col = self.get_target_variable()
            
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
            
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

def main():
    """Main function to test the DataLoader"""
    loader = DataLoader()
    
    try:
        # Load data
        data = loader.load_data()
        
        # Get info
        info = loader.get_data_info()
        print("Dataset Information:")
        print(f"Shape: {info['shape']}")
        print(f"Columns: {info['columns']}")
        print(f"Missing values: {info['missing_values']}")
        
        # Display sample
        print("\nSample data:")
        print(loader.display_sample())
        
        # Get target variable
        target = loader.get_target_variable()
        print(f"\nTarget variable: {target}")
        
        # Split features and target
        X, y = loader.split_features_target()
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the dataset from Kaggle and place it in the data/ directory.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 