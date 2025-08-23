"""
Data Preprocessing Module for Laptop Recommender System
Handles data loading, cleaning, and currency conversion from INR to MYR
"""

import pandas as pd
import numpy as np
import requests
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LaptopDataPreprocessor:
    """
    A comprehensive data preprocessor for laptop recommendation system.
    Handles data loading, cleaning, and currency conversion.
    """
    
    def __init__(self, data_path: str = "data/Cleaned_Laptop_data.csv"):
        """
        Initialize the preprocessor.
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = data_path
        self.df = None
        self.exchange_rate = None
        self.processed_df = None
        
    def get_exchange_rate(self, from_currency: str = "INR", to_currency: str = "MYR") -> float:
        """
        Get current exchange rate from INR to MYR using a free API.
        
        Args:
            from_currency (str): Source currency (default: INR)
            to_currency (str): Target currency (default: MYR)
            
        Returns:
            float: Exchange rate
        """
        try:
            # Using a free exchange rate API
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            exchange_rate = data['rates'].get(to_currency)
            
            if exchange_rate is None:
                logger.warning(f"Exchange rate not found for {to_currency}, using fallback rate")
                # Fallback rate (approximate as of recent data)
                exchange_rate = 0.056  # 1 INR ≈ 0.056 MYR
            
            logger.info(f"Exchange rate {from_currency} to {to_currency}: {exchange_rate}")
            return exchange_rate
            
        except Exception as e:
            logger.error(f"Error fetching exchange rate: {e}")
            # Fallback to approximate rate
            exchange_rate = 0.056
            logger.info(f"Using fallback exchange rate: {exchange_rate}")
            return exchange_rate
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the laptop dataset from CSV file.
        
        Returns:
            pd.DataFrame: Raw dataset
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, data types, and inconsistencies.
        
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Starting data cleaning process...")
        
        # Create a copy to avoid modifying original data
        df_clean = self.df.copy()
        
        # 1. Handle missing values in categorical columns
        categorical_columns = ['brand', 'model', 'processor_brand', 'processor_name', 
                             'processor_generation', 'ram_type', 'os', 'os_bit', 'weight']
        
        for col in categorical_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
                # Replace 'Missing' with 'Unknown'
                df_clean[col] = df_clean[col].replace('Missing', 'Unknown')
        
        # 2. Clean numerical columns
        numerical_columns = ['ram_gb', 'ssd', 'hdd', 'graphic_card_gb', 'display_size', 
                           'warranty', 'latest_price', 'old_price', 'discount', 
                           'star_rating', 'ratings', 'reviews']
        
        for col in numerical_columns:
            if col in df_clean.columns:
                # Remove 'GB' suffix from storage and RAM columns and convert to numeric
                if col in ['ram_gb', 'ssd', 'hdd', 'graphic_card_gb']:
                    df_clean[col] = df_clean[col].astype(str).str.replace(' GB', '').str.replace('GB', '')
                
                # Convert to numeric, coerce errors to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Fill missing values with appropriate defaults
                if col in ['ram_gb', 'ssd', 'hdd', 'graphic_card_gb']:
                    df_clean[col] = df_clean[col].fillna(0)
                elif col == 'display_size':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif col == 'warranty':
                    df_clean[col] = df_clean[col].fillna(1)  # Default 1 year warranty
                elif col in ['star_rating', 'ratings', 'reviews']:
                    df_clean[col] = df_clean[col].fillna(0)
        
        # 3. Clean boolean columns
        boolean_columns = ['touchscreen', 'microsoft_office']
        for col in boolean_columns:
            if col in df_clean.columns:
                # Convert to boolean, treating 'Yes' as True, 'No' as False
                df_clean[col] = df_clean[col].map({'Yes': True, 'No': False})
                df_clean[col] = df_clean[col].fillna(False)
        
        # 4. Clean price columns - remove any non-numeric characters
        price_columns = ['latest_price', 'old_price']
        for col in price_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                # Remove rows with invalid prices
                df_clean = df_clean.dropna(subset=[col])
        
        # 5. Calculate discount percentage if missing
        if 'discount' in df_clean.columns and 'old_price' in df_clean.columns and 'latest_price' in df_clean.columns:
            mask = (df_clean['discount'].isna()) & (df_clean['old_price'] > 0)
            df_clean.loc[mask, 'discount'] = (
                (df_clean.loc[mask, 'old_price'] - df_clean.loc[mask, 'latest_price']) / 
                df_clean.loc[mask, 'old_price'] * 100
            )
            df_clean['discount'] = df_clean['discount'].fillna(0)
        
        # 6. Remove duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        final_rows = len(df_clean)
        logger.info(f"Removed {initial_rows - final_rows} duplicate rows")
        
        # 7. Remove rows with invalid data
        # Remove rows where both SSD and HDD are 0 (invalid storage configuration)
        # But allow cases where at least one storage type is present
        storage_mask = (df_clean['ssd'] == 0) & (df_clean['hdd'] == 0)
        storage_removed = storage_mask.sum()
        df_clean = df_clean[~storage_mask]
        if storage_removed > 0:
            logger.info(f"Removed {storage_removed} rows with no storage")
        
        # Remove rows with extremely low or high prices (outliers)
        # Only apply if we have enough data
        if len(df_clean) > 10:
            price_q1 = df_clean['latest_price'].quantile(0.01)
            price_q99 = df_clean['latest_price'].quantile(0.99)
            price_mask = (df_clean['latest_price'] >= price_q1) & (df_clean['latest_price'] <= price_q99)
            price_removed = (~price_mask).sum()
            df_clean = df_clean[price_mask]
            if price_removed > 0:
                logger.info(f"Removed {price_removed} price outliers")
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def convert_currency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert prices from INR to MYR.
        
        Args:
            df (pd.DataFrame): DataFrame with prices in INR
            
        Returns:
            pd.DataFrame: DataFrame with prices converted to MYR
        """
        logger.info("Converting currency from INR to MYR...")
        
        # Get exchange rate
        self.exchange_rate = self.get_exchange_rate()
        
        # Create a copy to avoid modifying original data
        df_converted = df.copy()
        
        # Convert price columns
        price_columns = ['latest_price', 'old_price']
        for col in price_columns:
            if col in df_converted.columns:
                df_converted[f'{col}_myr'] = df_converted[col] * self.exchange_rate
                df_converted[f'{col}_myr'] = df_converted[f'{col}_myr'].round(2)
        
        # Update discount calculation for MYR prices
        if 'old_price_myr' in df_converted.columns and 'latest_price_myr' in df_converted.columns:
            df_converted['discount_myr'] = (
                (df_converted['old_price_myr'] - df_converted['latest_price_myr']) / 
                df_converted['old_price_myr'] * 100
            ).round(2)
        
        logger.info("Currency conversion completed")
        return df_converted
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for better recommendation analysis.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        logger.info("Adding derived features...")
        
        df_enhanced = df.copy()
        
        # 1. Total storage capacity
        df_enhanced['total_storage_gb'] = df_enhanced['ssd'] + df_enhanced['hdd']
        
        # 2. Storage type classification
        def classify_storage_type(row):
            if row['ssd'] > 0 and row['hdd'] > 0:
                return 'Hybrid'
            elif row['ssd'] > 0:
                return 'SSD'
            elif row['hdd'] > 0:
                return 'HDD'
            else:
                return 'Unknown'
        
        df_enhanced['storage_type'] = df_enhanced.apply(classify_storage_type, axis=1)
        
        # 3. Performance category based on processor and RAM
        def classify_performance(row):
            ram = row['ram_gb']
            processor = str(row['processor_name']).lower()
            
            if ram >= 16 or 'i7' in processor or 'i9' in processor or 'ryzen 7' in processor or 'ryzen 9' in processor:
                return 'High'
            elif ram >= 8 or 'i5' in processor or 'ryzen 5' in processor:
                return 'Medium'
            else:
                return 'Basic'
        
        df_enhanced['performance_category'] = df_enhanced.apply(classify_performance, axis=1)
        
        # 4. Price category
        def classify_price_category(row):
            price = row['latest_price_myr'] if 'latest_price_myr' in row else row['latest_price']
            if price < 2000:
                return 'Budget'
            elif price < 4000:
                return 'Mid-range'
            elif price < 8000:
                return 'High-end'
            else:
                return 'Premium'
        
        df_enhanced['price_category'] = df_enhanced.apply(classify_price_category, axis=1)
        
        # 5. Brand popularity (based on number of models)
        brand_counts = df_enhanced['brand'].value_counts()
        df_enhanced['brand_popularity'] = df_enhanced['brand'].map(brand_counts)
        
        # 6. Value for money score
        df_enhanced['value_score'] = (
            (df_enhanced['star_rating'] * df_enhanced['ratings']) / 
            (df_enhanced['latest_price_myr'] if 'latest_price_myr' in df_enhanced.columns else df_enhanced['latest_price'])
        ).fillna(0)
        
        logger.info("Derived features added successfully")
        return df_enhanced
    
    def preprocess_pipeline(self) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            pd.DataFrame: Fully processed dataset
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # 1. Load data
        self.load_data()
        
        # 2. Clean data
        cleaned_df = self.clean_data()
        
        # 3. Convert currency
        converted_df = self.convert_currency(cleaned_df)
        
        # 4. Add derived features
        final_df = self.add_derived_features(converted_df)
        
        # 5. Save processed data
        self.processed_df = final_df
        self.save_processed_data()
        
        logger.info("Preprocessing pipeline completed successfully!")
        return final_df
    
    def save_processed_data(self, output_path: str = "data/processed_laptop_data.csv"):
        """
        Save the processed dataset to CSV file.
        
        Args:
            output_path (str): Path to save the processed data
        """
        if self.processed_df is None:
            logger.warning("No processed data to save")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            self.processed_df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the processed dataset.
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if self.processed_df is None:
            raise ValueError("No processed data available. Run preprocess_pipeline() first.")
        
        summary = {
            'total_records': len(self.processed_df),
            'total_features': len(self.processed_df.columns),
            'price_range_myr': {
                'min': self.processed_df['latest_price_myr'].min(),
                'max': self.processed_df['latest_price_myr'].max(),
                'mean': self.processed_df['latest_price_myr'].mean(),
                'median': self.processed_df['latest_price_myr'].median()
            },
            'brands_count': self.processed_df['brand'].nunique(),
            'storage_types': self.processed_df['storage_type'].value_counts().to_dict(),
            'performance_categories': self.processed_df['performance_category'].value_counts().to_dict(),
            'price_categories': self.processed_df['price_category'].value_counts().to_dict(),
            'exchange_rate_used': self.exchange_rate
        }
        
        return summary
    
    def export_feature_columns(self) -> Dict[str, list]:
        """
        Export the feature columns organized by type for the recommendation system.
        
        Returns:
            Dict[str, list]: Dictionary with feature columns organized by type
        """
        if self.processed_df is None:
            raise ValueError("No processed data available. Run preprocess_pipeline() first.")
        
        feature_columns = {
            'categorical_features': [
                'brand', 'model', 'processor_brand', 'processor_name', 
                'processor_generation', 'ram_type', 'os', 'os_bit', 'weight',
                'storage_type', 'performance_category', 'price_category'
            ],
            'numerical_features': [
                'ram_gb', 'ssd', 'hdd', 'graphic_card_gb', 'display_size', 
                'warranty', 'latest_price_myr', 'old_price_myr', 'discount_myr',
                'star_rating', 'ratings', 'reviews', 'total_storage_gb',
                'brand_popularity', 'value_score'
            ],
            'boolean_features': [
                'touchscreen', 'microsoft_office'
            ],
            'target_features': [
                'latest_price_myr', 'star_rating', 'value_score'
            ]
        }
        
        # Filter to only include columns that exist in the dataset
        for category in feature_columns:
            feature_columns[category] = [
                col for col in feature_columns[category] 
                if col in self.processed_df.columns
            ]
        
        return feature_columns


def main():
    """
    Main function to run the preprocessing pipeline.
    """
    try:
        # Initialize preprocessor
        preprocessor = LaptopDataPreprocessor()
        
        # Run complete pipeline
        processed_data = preprocessor.preprocess_pipeline()
        
        # Get and display summary
        summary = preprocessor.get_data_summary()
        print("\n" + "="*50)
        print("DATA PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Total Records: {summary['total_records']}")
        print(f"Total Features: {summary['total_features']}")
        print(f"Exchange Rate (INR to MYR): {summary['exchange_rate_used']:.4f}")
        print(f"Price Range (MYR): RM {summary['price_range_myr']['min']:.2f} - RM {summary['price_range_myr']['max']:.2f}")
        print(f"Average Price (MYR): RM {summary['price_range_myr']['mean']:.2f}")
        print(f"Number of Brands: {summary['brands_count']}")
        
        print("\nStorage Types Distribution:")
        for storage_type, count in summary['storage_types'].items():
            print(f"  {storage_type}: {count}")
        
        print("\nPerformance Categories:")
        for perf_cat, count in summary['performance_categories'].items():
            print(f"  {perf_cat}: {count}")
        
        print("\nPrice Categories:")
        for price_cat, count in summary['price_categories'].items():
            print(f"  {price_cat}: {count}")
        
        # Export feature columns
        feature_cols = preprocessor.export_feature_columns()
        print(f"\nFeature Columns Available:")
        for category, columns in feature_cols.items():
            print(f"  {category}: {len(columns)} features")
        
        print("\nPreprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
