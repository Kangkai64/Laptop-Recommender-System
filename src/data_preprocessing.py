"""
Data Preprocessing Module for Laptop Price Prediction
Handles feature engineering, encoding, and data cleaning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re
import logging
from typing import Tuple, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class to handle data preprocessing for laptop price prediction"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        self.categorical_columns = []
        self.numeric_columns = []
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and data types
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # For categorical columns, fill with mode
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown')
            else:
                # For numeric columns, fill with median
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        logger.info("Data cleaning completed")
        return df_clean
    
    def extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract numeric features from text columns (e.g., RAM, Storage)
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with extracted numeric features
        """
        df_extracted = df.copy()
        
        # Extract RAM size (assuming it's in GB)
        if 'RAM' in df_extracted.columns:
            df_extracted['RAM_GB'] = df_extracted['RAM'].apply(self._extract_ram_size)
        
        # Extract storage size
        if 'Storage' in df_extracted.columns:
            df_extracted['Storage_GB'] = df_extracted['Storage'].apply(self._extract_storage_size)
        
        # Extract screen size
        if 'Screen_Size' in df_extracted.columns:
            df_extracted['Screen_Size_Inches'] = df_extracted['Screen_Size'].apply(self._extract_screen_size)
        
        # Extract weight
        if 'Weight' in df_extracted.columns:
            df_extracted['Weight_Kg'] = df_extracted['Weight'].apply(self._extract_weight)
        
        logger.info("Numeric feature extraction completed")
        return df_extracted
    
    def _extract_ram_size(self, ram_str: str) -> float:
        """Extract RAM size in GB from string"""
        if pd.isna(ram_str):
            return 8.0  # Default value
        
        try:
            # Extract numbers from string like "8 GB", "16GB", etc.
            numbers = re.findall(r'\d+', str(ram_str))
            if numbers:
                return float(numbers[0])
        except:
            pass
        return 8.0  # Default value
    
    def _extract_storage_size(self, storage_str: str) -> float:
        """Extract storage size in GB from string"""
        if pd.isna(storage_str):
            return 512.0  # Default value
        
        try:
            storage_str = str(storage_str).upper()
            if 'TB' in storage_str:
                numbers = re.findall(r'\d+', storage_str)
                if numbers:
                    return float(numbers[0]) * 1024  # Convert TB to GB
            elif 'GB' in storage_str:
                numbers = re.findall(r'\d+', storage_str)
                if numbers:
                    return float(numbers[0])
        except:
            pass
        return 512.0  # Default value
    
    def _extract_screen_size(self, screen_str: str) -> float:
        """Extract screen size in inches from string"""
        if pd.isna(screen_str):
            return 15.6  # Default value
        
        try:
            numbers = re.findall(r'\d+\.?\d*', str(screen_str))
            if numbers:
                return float(numbers[0])
        except:
            pass
        return 15.6  # Default value
    
    def _extract_weight(self, weight_str: str) -> float:
        """Extract weight in kg from string"""
        if pd.isna(weight_str):
            return 2.0  # Default value
        
        try:
            weight_str = str(weight_str).upper()
            if 'KG' in weight_str:
                numbers = re.findall(r'\d+\.?\d*', weight_str)
                if numbers:
                    return float(numbers[0])
            elif 'G' in weight_str:
                numbers = re.findall(r'\d+', weight_str)
                if numbers:
                    return float(numbers[0]) / 1000  # Convert g to kg
        except:
            pass
        return 2.0  # Default value
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        # Encode each categorical column
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numeric features
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Scaling method ('standard' or 'minmax')
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        df_scaled = df.copy()
        
        # Identify numeric columns
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Scale numeric features
        df_scaled[numeric_cols] = self.scaler.fit_transform(df_scaled[numeric_cols])
        
        logger.info(f"Scaled {len(numeric_cols)} numeric columns using {method} scaling")
        return df_scaled
    
    def create_preprocessing_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create a complete preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Preprocessed dataframe and feature names
        """
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Extract numeric features
        df_extracted = self.extract_numeric_features(df_clean)
        
        # Step 3: Encode categorical features
        df_encoded = self.encode_categorical_features(df_extracted)
        
        # Step 4: Scale features
        df_scaled = self.scale_features(df_encoded)
        
        # Get feature names
        feature_names = [col for col in df_scaled.columns if col != 'Price']  # Assuming 'Price' is target
        
        logger.info(f"Preprocessing completed. Final shape: {df_scaled.shape}")
        return df_scaled, feature_names
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the preprocessing steps
        
        Returns:
            Dict[str, Any]: Summary of preprocessing
        """
        summary = {
            'label_encoders': list(self.label_encoders.keys()),
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'feature_names': self.feature_names
        }
        return summary

def main():
    """Main function to test the DataPreprocessor"""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    try:
        data = loader.load_data()
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Apply preprocessing
        df_processed, feature_names = preprocessor.create_preprocessing_pipeline(data)
        
        print("Preprocessing completed!")
        print(f"Original shape: {data.shape}")
        print(f"Processed shape: {df_processed.shape}")
        print(f"Feature names: {feature_names}")
        
        # Show preprocessing summary
        summary = preprocessor.get_preprocessing_summary()
        print(f"Preprocessing summary: {summary}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    main() 