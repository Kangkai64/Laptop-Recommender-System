"""
Data Preprocessing Module for Laptop Recommender System
Handles data loading, cleaning, and processing for Amazon laptop reviews enriched dataset
"""

import pandas as pd
import numpy as np
import requests
from typing import Optional, Dict, Any, List, Tuple
import logging
from datetime import datetime
import os
from datasets import load_dataset
import json
import re
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LaptopDataPreprocessor:
    """
    A comprehensive data preprocessor for laptop recommendation system.
    Handles data loading, cleaning, and processing for Amazon laptop reviews enriched dataset.
    """
    
    def __init__(self):
        """
        Initialize the preprocessor.
        """
        self.df = None
        self.processed_df = None
        self.df_laptop = None
        self.df_rating = None
        self.scalers = {}
        self.label_encoders = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the Amazon laptop reviews enriched dataset from Hugging Face.
        
        Returns:
            pd.DataFrame: Raw dataset
        """
        try:
            logger.info("Loading Amazon laptop reviews enriched dataset from Hugging Face...")
            
            # Load dataset from Hugging Face
            dataset = load_dataset("naga-jay/amazon-laptop-reviews-enriched")
            
            # Convert to pandas DataFrame
            self.df = dataset['train'].to_pandas()
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Columns: {list(self.df.columns)}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def separate_dataframes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate the main dataframe into laptop product data and rating data.
        
        Args:
            df (pd.DataFrame): Cleaned dataset
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_laptop, df_rating)
        """
        logger.info("Separating data into laptop and rating dataframes...")
        
        # Define columns for each dataframe
        laptop_columns = [
            'asin', 'parent_asin', 'title_y', 'brand', 'os', 'color', 'store',
            'average_rating', 'rating_number', 'features', 'price',
            'images_y'
        ]
        
        rating_columns = [
            'asin', 'parent_asin', 'user_id', 'timestamp', 'rating', 'title_x',
            'text', 'helpful_vote'
        ]
        
        # Filter to only include columns that exist in the dataset
        available_laptop_cols = [col for col in laptop_columns if col in df.columns]
        available_rating_cols = [col for col in rating_columns if col in df.columns]
        
        # Create separate dataframes
        df_laptop = df[available_laptop_cols].drop_duplicates(subset=['asin']).reset_index(drop=True)
        df_rating = df[available_rating_cols].reset_index(drop=True)
        
        logger.info(f"Laptop dataframe shape: {df_laptop.shape}")
        logger.info(f"Rating dataframe shape: {df_rating.shape}")
        
        self.df_laptop = df_laptop
        self.df_rating = df_rating
        
        return df_laptop, df_rating

    def add_price_conversion(self, df_laptop: pd.DataFrame) -> pd.DataFrame:
        """
        Add price conversion from USD to Malaysian Ringgit (MYR).
        Current exchange rate: 1 USD ≈ 4.75 MYR (approximate)
        
        Args:
            df_laptop (pd.DataFrame): Laptop dataframe
            
        Returns:
            pd.DataFrame: Laptop dataframe with price conversion
        """
        logger.info("Adding price conversion to Malaysian Ringgit...")
        
        df_enhanced = df_laptop.copy()
        
        # Extract numeric price from price column
        if 'price' in df_enhanced.columns:
            df_enhanced['price_usd'] = df_enhanced['price'].astype(str).apply(self._extract_price)
            df_enhanced['price_usd'] = pd.to_numeric(df_enhanced['price_usd'], errors='coerce')
            
            # Convert USD to MYR (approximate exchange rate)
            exchange_rate = 4.75
            df_enhanced['price_myr'] = df_enhanced['price_usd'] * exchange_rate
            
            # Create price categories in MYR
            df_enhanced['price_category_myr'] = pd.cut(
                df_enhanced['price_myr'],
                bins=[0, 2375, 4750, 9500, float('inf')],  # 500, 1000, 2000 USD equivalents
                labels=['Budget', 'Mid-range', 'High-end', 'Premium'],
                include_lowest=True
            )
            
            logger.info(f"Price conversion added. Price range: RM {df_enhanced['price_myr'].min():.2f} - RM {df_enhanced['price_myr'].max():.2f}")
        
        return df_enhanced

    def normalize_laptop_data(self, df_laptop: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize laptop dataframe data for better performance.
        
        Args:
            df_laptop (pd.DataFrame): Laptop dataframe
            
        Returns:
            pd.DataFrame: Normalized laptop dataframe with only cleaned columns
        """
        logger.info("Normalizing laptop data...")
        
        df_normalized = df_laptop.copy()
        
        # 1. Normalize numerical columns
        numerical_columns = ['average_rating', 'rating_number']
        available_numerical = [col for col in numerical_columns if col in df_normalized.columns]
        
        if available_numerical:
            scaler = MinMaxScaler()
            df_normalized[available_numerical] = scaler.fit_transform(df_normalized[available_numerical].fillna(0))
            self.scalers['laptop_numerical'] = scaler
        
        # 2. Encode categorical columns
        categorical_columns = ['brand', 'os', 'color', 'store']
        available_categorical = [col for col in categorical_columns if col in df_normalized.columns]
        
        for col in available_categorical:
            if col in df_normalized.columns:
                le = LabelEncoder()
                df_normalized[f'{col}_encoded'] = le.fit_transform(df_normalized[col].fillna('Unknown'))
                self.label_encoders[f'laptop_{col}'] = le
        
        # 3. Clean and normalize text columns
        text_columns = ['title_y', 'features']
        for col in text_columns:
            if col in df_normalized.columns:
                df_normalized[f'{col}_clean'] = df_normalized[col].astype(str).apply(self._clean_text)
        
        # 4. Keep only essential columns and cleaned versions
        essential_columns = ['asin', 'parent_asin', 'price_usd', 'price_myr', 'price_category_myr']
        encoded_columns = [col for col in df_normalized.columns if col.endswith('_encoded')]
        clean_columns = [col for col in df_normalized.columns if col.endswith('_clean')]
        numerical_columns = ['average_rating', 'rating_number']
        
        # Add specification columns
        specification_columns = [col for col in df_normalized.columns if any(x in col for x in [
            'ram_gb', 'storage_gb', 'screen_size_inches', 'storage_type', 'ram_type', 
            'processor_model', 'gpu_model', 'storage_category', 'ram_category', 'screen_category',
            'storage_display'
        ])]
        
        final_columns = essential_columns + encoded_columns + clean_columns + numerical_columns + specification_columns
        available_final_columns = [col for col in final_columns if col in df_normalized.columns]
        
        df_final = df_normalized[available_final_columns]
        
        logger.info("Laptop data normalization completed")
        return df_final

    def normalize_rating_data(self, df_rating: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize rating dataframe data for better performance.
        
        Args:
            df_rating (pd.DataFrame): Rating dataframe
            
        Returns:
            pd.DataFrame: Normalized rating dataframe with only cleaned columns
        """
        logger.info("Normalizing rating data...")
        
        df_normalized = df_rating.copy()
        
        # 1. Normalize numerical columns
        numerical_columns = ['rating', 'helpful_vote']
        available_numerical = [col for col in numerical_columns if col in df_normalized.columns]
        
        if available_numerical:
            scaler = MinMaxScaler()
            df_normalized[available_numerical] = scaler.fit_transform(df_normalized[available_numerical].fillna(0))
            self.scalers['rating_numerical'] = scaler
        
        # 2. Encode user_id
        if 'user_id' in df_normalized.columns:
            le = LabelEncoder()
            df_normalized['user_id_encoded'] = le.fit_transform(df_normalized['user_id'].fillna('unknown'))
            self.label_encoders['rating_user_id'] = le
        
        # 3. Clean text columns
        text_columns = ['title_x', 'text']
        for col in text_columns:
            if col in df_normalized.columns:
                df_normalized[f'{col}_clean'] = df_normalized[col].astype(str).apply(self._clean_text)
        
        # 4. Convert timestamp to datetime features
        if 'timestamp' in df_normalized.columns:
            df_normalized['timestamp'] = pd.to_datetime(df_normalized['timestamp'], errors='coerce')
            df_normalized['year'] = df_normalized['timestamp'].dt.year
            df_normalized['month'] = df_normalized['timestamp'].dt.month
            df_normalized['day_of_week'] = df_normalized['timestamp'].dt.dayofweek
        
        # 5. Keep only essential columns and cleaned versions
        essential_columns = ['asin', 'parent_asin', 'user_id_encoded', 'rating', 'helpful_vote']
        clean_columns = [col for col in df_normalized.columns if col.endswith('_clean')]
        temporal_columns = ['year', 'month', 'day_of_week']
        
        final_columns = essential_columns + clean_columns + temporal_columns
        available_final_columns = [col for col in final_columns if col in df_normalized.columns]
        
        df_final = df_normalized[available_final_columns]
        
        logger.info("Rating data normalization completed")
        return df_final

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
        categorical_columns = ['brand', 'os', 'color', 'store', 'main_category']
        
        for col in categorical_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
                # Replace 'Missing' with 'Unknown'
                df_clean[col] = df_clean[col].replace('Missing', 'Unknown')
        
        # 2. Clean numerical columns
        numerical_columns = ['rating', 'helpful_vote', 'average_rating', 'rating_number', 'num_reviews', 'avg_helpful_votes']
        
        for col in numerical_columns:
            if col in df_clean.columns:
                # Convert to numeric, coerce errors to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Fill missing values with appropriate defaults
                if col in ['rating', 'average_rating']:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif col in ['helpful_vote', 'rating_number', 'num_reviews', 'avg_helpful_votes']:
                    df_clean[col] = df_clean[col].fillna(0)
        
        # 3. Clean price column - extract numeric values
        if 'price' in df_clean.columns:
            df_clean['price_numeric'] = df_clean['price'].astype(str).apply(self._extract_price)
            df_clean['price_numeric'] = pd.to_numeric(df_clean['price_numeric'], errors='coerce')
            # Remove rows with invalid prices
            df_clean = df_clean.dropna(subset=['price_numeric'])
        
        # 4. Clean text columns
        text_columns = ['title_x', 'text', 'title_y']
        
        for col in text_columns:
            if col in df_clean.columns:
                # Remove HTML tags and clean text
                df_clean[col] = df_clean[col].astype(str).apply(self._clean_text)
                # Fill missing values
                df_clean[col] = df_clean[col].fillna('')
        
        # 5. Clean boolean columns
        if 'verified_purchase' in df_clean.columns:
            df_clean['verified_purchase'] = df_clean['verified_purchase'].fillna(False)
        
        # 6. Process features and details columns
        if 'features' in df_clean.columns:
            # Skip features processing for now to avoid numpy array issues
            df_clean['features_clean'] = df_clean['features'].astype(str)
        
        if 'details' in df_clean.columns:
            df_clean['details_parsed'] = df_clean['details'].apply(self._parse_details)
        
        # 7. Remove duplicate rows
        initial_rows = len(df_clean)
        
        # Convert numpy arrays to strings to make them hashable for drop_duplicates
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Check if column contains numpy arrays
                sample_val = df_clean[col].iloc[0] if len(df_clean) > 0 else None
                if hasattr(sample_val, '__iter__') and not isinstance(sample_val, str):
                    df_clean[col] = df_clean[col].astype(str)
        
        df_clean = df_clean.drop_duplicates()
        final_rows = len(df_clean)
        logger.info(f"Removed {initial_rows - final_rows} duplicate rows")
        
        # 8. Remove rows with invalid data
        # Remove rows with no text content
        text_mask = (df_clean['text'].str.len() < 10) & (df_clean['title_x'].str.len() < 5)
        text_removed = text_mask.sum()
        df_clean = df_clean[~text_mask]
        if text_removed > 0:
            logger.info(f"Removed {text_removed} rows with insufficient text content")
        
        # Remove rows with extremely low or high prices (outliers)
        if 'price_numeric' in df_clean.columns and len(df_clean) > 10:
            price_q1 = df_clean['price_numeric'].quantile(0.01)
            price_q99 = df_clean['price_numeric'].quantile(0.99)
            price_mask = (df_clean['price_numeric'] >= price_q1) & (df_clean['price_numeric'] <= price_q99)
            price_removed = (~price_mask).sum()
            df_clean = df_clean[price_mask]
            if price_removed > 0:
                logger.info(f"Removed {price_removed} price outliers")
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def _extract_price(self, price_str: str) -> float:
        """
        Extract numeric price from price string.
        
        Args:
            price_str (str): Price string
            
        Returns:
            float: Extracted price value
        """
        if pd.isna(price_str) or price_str == 'nan':
            return np.nan
        
        # Remove currency symbols and extract numbers
        price_str = str(price_str)
        # Remove common currency symbols and text
        price_str = re.sub(r'[^\d.,]', '', price_str)
        
        # Handle different price formats
        if ',' in price_str and '.' in price_str:
            # Format like "1,299.99"
            price_str = price_str.replace(',', '')
        elif ',' in price_str:
            # Format like "1,299"
            price_str = price_str.replace(',', '')
        
        try:
            return float(price_str)
        except ValueError:
            return np.nan
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML tags and special characters.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ''
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    def _process_features(self, features) -> str:
        """
        Process features list into a clean string.
        
        Args:
            features: Features list or string
            
        Returns:
            str: Processed features string
        """
        try:
            if pd.isna(features).any() if hasattr(pd.isna(features), 'any') else pd.isna(features):
                return ''
            
            # Handle numpy arrays
            if hasattr(features, '__iter__') and not isinstance(features, str):
                try:
                    return ' | '.join([str(f) for f in features if f])
                except:
                    return str(features)
            else:
                return str(features)
        except:
            return str(features)
    
    def _parse_details(self, details) -> Dict:
        """
        Parse details dictionary or string into structured format.
        
        Args:
            details: Details data
            
        Returns:
            Dict: Parsed details
        """
        if pd.isna(details) or details == 'nan':
            return {}
        
        if isinstance(details, dict):
            return details
        elif isinstance(details, str):
            try:
                return json.loads(details)
            except:
                return {'raw_details': details}
        else:
            return {}
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for better recommendation analysis.
        
        Args:
            df (pd.DataFrame): Cleaned dataset
            
        Returns:
            pd.DataFrame: Dataset with derived features
        """
        logger.info("Adding derived features...")
        
        df_enhanced = df.copy()
        
        # 1. Create price categories
        if 'price_numeric' in df_enhanced.columns:
            df_enhanced['price_category'] = pd.cut(
                df_enhanced['price_numeric'],
                bins=[0, 500, 1000, 2000, float('inf')],
                labels=['Budget', 'Mid-range', 'High-end', 'Premium'],
                include_lowest=True
            )
        
        # 2. Create rating categories
        if 'rating' in df_enhanced.columns:
            df_enhanced['rating_category'] = pd.cut(
                df_enhanced['rating'],
                bins=[0, 2, 3, 4, 5],
                labels=['Poor', 'Fair', 'Good', 'Excellent'],
                include_lowest=True
            )
        
        # 3. Extract specifications from details
        if 'details_parsed' in df_enhanced.columns:
            # Extract common specs
            df_enhanced['ram_gb'] = df_enhanced['details_parsed'].apply(
                lambda x: self._extract_spec(x, 'RAM', 'GB') if isinstance(x, dict) else None
            )
            df_enhanced['storage_gb'] = df_enhanced['details_parsed'].apply(
                lambda x: self._extract_storage(x) if isinstance(x, dict) else None
            )
            df_enhanced['screen_size'] = df_enhanced['details_parsed'].apply(
                lambda x: self._extract_spec(x, 'Screen Size', 'Inches') if isinstance(x, dict) else None
            )
            df_enhanced['processor'] = df_enhanced['details_parsed'].apply(
                lambda x: self._extract_processor(x) if isinstance(x, dict) else None
            )
        
        # 4. Create text length features
        df_enhanced['review_length'] = df_enhanced['text'].str.len()
        df_enhanced['title_length'] = df_enhanced['title_x'].str.len()
        
        # 5. Create helpfulness ratio
        if 'helpful_vote' in df_enhanced.columns and 'num_reviews' in df_enhanced.columns:
            df_enhanced['helpfulness_ratio'] = (
                df_enhanced['helpful_vote'] / df_enhanced['num_reviews'].replace(0, 1)
            ).fillna(0)
        
        # 6. Create brand popularity
        if 'brand' in df_enhanced.columns:
            brand_counts = df_enhanced['brand'].value_counts()
            df_enhanced['brand_popularity'] = df_enhanced['brand'].map(brand_counts)
        
        logger.info("Derived features added successfully")
        return df_enhanced
    
    def _extract_spec(self, details: Dict, key: str, unit: str) -> Optional[float]:
        """
        Extract specification value from details dictionary.
        
        Args:
            details (Dict): Details dictionary
            key (str): Key to search for
            unit (str): Unit to extract
            
        Returns:
            Optional[float]: Extracted value
        """
        if not isinstance(details, dict):
            return None
        
        for k, v in details.items():
            if key.lower() in k.lower() and str(v):
                # Extract numeric value
                match = re.search(rf'(\d+(?:\.\d+)?)\s*{unit}', str(v), re.IGNORECASE)
                if match:
                    return float(match.group(1))
        return None
    
    def _extract_storage(self, details: Dict) -> Optional[float]:
        """
        Extract storage capacity from details.
        
        Args:
            details (Dict): Details dictionary
            
        Returns:
            Optional[float]: Storage capacity in GB
        """
        if not isinstance(details, dict):
            return None
        
        for k, v in details.items():
            if any(term in k.lower() for term in ['hard drive', 'ssd', 'storage', 'flash memory']):
                if str(v):
                    # Extract numeric value
                    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:GB|TB)', str(v), re.IGNORECASE)
                    if match:
                        value = float(match.group(1))
                        # Convert TB to GB
                        if 'TB' in str(v).upper():
                            value *= 1024
                        return value
        return None
    
    def _extract_processor(self, details: Dict) -> Optional[str]:
        """
        Extract processor information from details.
        
        Args:
            details (Dict): Details dictionary
            
        Returns:
            Optional[str]: Processor information
        """
        if not isinstance(details, dict):
            return None
        
        for k, v in details.items():
            if 'processor' in k.lower() and str(v):
                return str(v)
        return None
    
    def preprocess_pipeline(self) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            pd.DataFrame: Fully processed dataset
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Clean data
        cleaned_data = self.clean_data()
        
        # Step 3: Add derived features
        processed_data = self.add_derived_features(cleaned_data)
        
        # Step 4: Save processed data
        self.save_processed_data(processed_data)
        
        self.processed_df = processed_data
        logger.info("Preprocessing pipeline completed successfully")
        
        return processed_data
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str = "data/processed_laptop_data.csv"):
        """
        Save the processed dataset to CSV file.
        
        Args:
            df (pd.DataFrame): Processed dataset
            filepath (str): Output file path
        """
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
    
    def get_data_summary(self) -> Dict:
        """
        Get a comprehensive summary of the processed dataset.
        
        Returns:
            Dict: Dataset summary
        """
        if self.processed_df is None:
            raise ValueError("No processed data available. Run preprocess_pipeline() first.")
        
        df = self.processed_df
        
        summary = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'price_range': {
                'min': df['price_numeric'].min() if 'price_numeric' in df.columns else None,
                'max': df['price_numeric'].max() if 'price_numeric' in df.columns else None,
                'mean': df['price_numeric'].mean() if 'price_numeric' in df.columns else None
            },
            'brands_count': df['brand'].nunique() if 'brand' in df.columns else 0,
            'rating_stats': {
                'mean': df['rating'].mean() if 'rating' in df.columns else None,
                'median': df['rating'].median() if 'rating' in df.columns else None
            },
            'text_stats': {
                'avg_review_length': df['review_length'].mean() if 'review_length' in df.columns else None,
                'total_reviews': len(df[df['text'].str.len() > 10]) if 'text' in df.columns else 0
            }
        }
        
        return summary
    
    def export_feature_columns(self) -> Dict[str, List[str]]:
        """
        Export feature columns organized by category.
        
        Returns:
            Dict[str, List[str]]: Feature columns by category
        """
        if self.processed_df is None:
            raise ValueError("No processed data available. Run preprocess_pipeline() first.")
        
        df = self.processed_df
        
        feature_categories = {
            'Basic Info': ['asin', 'parent_asin', 'user_id', 'timestamp'],
            'Product Details': ['title_x', 'title_y', 'brand', 'os', 'color', 'store'],
            'Reviews & Ratings': ['rating', 'text', 'helpful_vote', 'verified_purchase', 'average_rating', 'rating_number'],
            'Pricing': ['price', 'price_numeric', 'price_category'],
            'Specifications': ['ram_gb', 'storage_gb', 'screen_size', 'processor'],
            'Analytics': ['num_reviews', 'avg_helpful_votes', 'helpfulness_ratio', 'brand_popularity'],
            'Content': ['features_clean', 'details_parsed'],
            'Derived': ['review_length', 'title_length', 'rating_category']
        }
        
        # Filter to only include columns that exist in the dataset
        available_features = {}
        for category, columns in feature_categories.items():
            available_cols = [col for col in columns if col in df.columns]
            if available_cols:
                available_features[category] = available_cols
        
        return available_features

    def preprocess_separated_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete preprocessing pipeline with separated dataframes.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_laptop, df_rating)
        """
        logger.info("Starting separated preprocessing pipeline...")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Clean data
        cleaned_data = self.clean_data()
        
        # Step 3: Separate dataframes
        df_laptop, df_rating = self.separate_dataframes(cleaned_data)
        
        # Step 4: Add price conversion
        df_laptop = self.add_price_conversion(df_laptop)
        
        # Step 5: Add specifications using benchmark scraper
        df_laptop = self.add_specifications_from_benchmark_scraper(df_laptop)
        
        # Step 6: Normalize data
        df_laptop_normalized = self.normalize_laptop_data(df_laptop)
        df_rating_normalized = self.normalize_rating_data(df_rating)
        
        self.df_laptop = df_laptop_normalized
        self.df_rating = df_rating_normalized
        
        logger.info("Separated preprocessing pipeline completed successfully")
        
        return df_laptop_normalized, df_rating_normalized

    def add_specifications_from_benchmark_scraper(self, df_laptop: pd.DataFrame) -> pd.DataFrame:
        """
        Add laptop specifications using the benchmark scraper.
        
        Args:
            df_laptop (pd.DataFrame): Laptop dataframe
            
        Returns:
            pd.DataFrame: Laptop dataframe with added specifications
        """
        try:
            from benchmark_scraper import BenchmarkScraper
            
            logger.info("Initializing benchmark scraper for specification extraction...")
            scraper = BenchmarkScraper()
            
            # Extract specifications from text columns
            df_with_specs = scraper.add_specifications_from_columns(df_laptop)
            
            logger.info("Specifications extracted successfully using benchmark scraper")
            return df_with_specs
            
        except ImportError as e:
            logger.warning(f"Could not import benchmark scraper: {e}")
            logger.info("Falling back to basic specification extraction...")
            return self._add_basic_specifications(df_laptop)
        except Exception as e:
            logger.error(f"Error in benchmark scraper specification extraction: {e}")
            logger.info("Falling back to basic specification extraction...")
            return self._add_basic_specifications(df_laptop)
    
    def _add_basic_specifications(self, df_laptop: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic specifications using built-in extraction methods.
        
        Args:
            df_laptop (pd.DataFrame): Laptop dataframe
            
        Returns:
            pd.DataFrame: Laptop dataframe with added specifications
        """
        logger.info("Adding basic specifications using built-in methods...")
        
        df_specs = df_laptop.copy()
        
        # Extract specifications from details_parsed if available
        if 'details_parsed' in df_specs.columns:
            # Extract common specs
            df_specs['ram_gb'] = df_specs['details_parsed'].apply(
                lambda x: self._extract_spec(x, 'RAM', 'GB') if isinstance(x, dict) else None
            )
            df_specs['storage_gb'] = df_specs['details_parsed'].apply(
                lambda x: self._extract_storage(x) if isinstance(x, dict) else None
            )
            df_specs['screen_size'] = df_specs['details_parsed'].apply(
                lambda x: self._extract_spec(x, 'Screen Size', 'Inches') if isinstance(x, dict) else None
            )
            df_specs['processor'] = df_specs['details_parsed'].apply(
                lambda x: self._extract_processor(x) if isinstance(x, dict) else None
            )
        
        logger.info("Basic specifications added successfully")
        return df_specs


    def get_separated_data_summary(self) -> Dict:
        """
        Get a comprehensive summary of the separated processed datasets.
        
        Returns:
            Dict: Dataset summary
        """
        if self.df_laptop is None or self.df_rating is None:
            raise ValueError("No separated data available. Run preprocess_separated_pipeline() first.")
        
        laptop_summary = {
            'total_products': len(self.df_laptop),
            'total_features': len(self.df_laptop.columns),
            'brands_count': self.df_laptop['brand'].nunique() if 'brand' in self.df_laptop.columns else 0,
            'price_range_myr': {
                'min': self.df_laptop['price_myr'].min() if 'price_myr' in self.df_laptop.columns else None,
                'max': self.df_laptop['price_myr'].max() if 'price_myr' in self.df_laptop.columns else None,
                'mean': self.df_laptop['price_myr'].mean() if 'price_myr' in self.df_laptop.columns else None
            },
            'average_rating': self.df_laptop['average_rating'].mean() if 'average_rating' in self.df_laptop.columns else None,
            'specifications': {}
        }
        
        # Add specification statistics if available
        if 'ram_gb' in self.df_laptop.columns:
            ram_stats = self.df_laptop['ram_gb'].describe()
            laptop_summary['specifications']['ram'] = {
                'found': int(self.df_laptop['ram_gb'].notna().sum()),
                'total': len(self.df_laptop),
                'mean_gb': float(ram_stats['mean']) if not pd.isna(ram_stats['mean']) else None,
                'min_gb': int(ram_stats['min']) if not pd.isna(ram_stats['min']) else None,
                'max_gb': int(ram_stats['max']) if not pd.isna(ram_stats['max']) else None
            }
        
        if 'storage_gb' in self.df_laptop.columns:
            storage_stats = self.df_laptop['storage_gb'].describe()
            laptop_summary['specifications']['storage'] = {
                'found': int(self.df_laptop['storage_gb'].notna().sum()),
                'total': len(self.df_laptop),
                'mean_gb': float(storage_stats['mean']) if not pd.isna(storage_stats['mean']) else None,
                'min_gb': int(storage_stats['min']) if not pd.isna(storage_stats['min']) else None,
                'max_gb': int(storage_stats['max']) if not pd.isna(storage_stats['max']) else None
            }
        
        if 'screen_size_inches' in self.df_laptop.columns:
            screen_stats = self.df_laptop['screen_size_inches'].describe()
            laptop_summary['specifications']['screen_size'] = {
                'found': int(self.df_laptop['screen_size_inches'].notna().sum()),
                'total': len(self.df_laptop),
                'mean_inches': float(screen_stats['mean']) if not pd.isna(screen_stats['mean']) else None,
                'min_inches': float(screen_stats['min']) if not pd.isna(screen_stats['min']) else None,
                'max_inches': float(screen_stats['max']) if not pd.isna(screen_stats['max']) else None
            }
        
        # Add column categories
        laptop_cols = list(self.df_laptop.columns)
        laptop_summary['column_categories'] = {
            'product_info': [col for col in laptop_cols if any(x in col for x in ['title', 'brand', 'os', 'color', 'store'])],
            'pricing': [col for col in laptop_cols if 'price' in col],
            'ratings': [col for col in laptop_cols if 'rating' in col],
            'specifications': [col for col in laptop_cols if any(x in col for x in ['ram', 'storage', 'screen', 'processor', 'gpu'])],
            'benchmarks': [col for col in laptop_cols if 'benchmark' in col],
            'categories': [col for col in laptop_cols if 'category' in col],
            'normalized': [col for col in laptop_cols if any(x in col for x in ['encoded', 'clean', 'normalized'])]
        }
        
        rating_summary = {
            'total_reviews': len(self.df_rating),
            'total_features': len(self.df_rating.columns),
            'unique_users': self.df_rating['user_id'].nunique() if 'user_id' in self.df_rating.columns else 0,
            'unique_products': self.df_rating['asin'].nunique() if 'asin' in self.df_rating.columns else 0,
            'rating_stats': {
                'mean': self.df_rating['rating'].mean() if 'rating' in self.df_rating.columns else None,
                'median': self.df_rating['rating'].median() if 'rating' in self.df_rating.columns else None
            }
        }
        
        return {
            'laptop_data': laptop_summary,
            'rating_data': rating_summary
        }


def main():
    """
    Main function to run the preprocessing pipeline.
    """
    try:
        # Initialize preprocessor
        preprocessor = LaptopDataPreprocessor()
        
        # Run separated preprocessing pipeline
        df_laptop, df_rating = preprocessor.preprocess_separated_pipeline()
        
        # Get and display summary
        summary = preprocessor.get_separated_data_summary()
        
        print("\n" + "="*60)
        print("SEPARATED DATA PREPROCESSING SUMMARY")
        print("="*60)
        
        print("\nLAPTOP DATA:")
        print(f"  Total Products: {summary['laptop_data']['total_products']}")
        print(f"  Total Features: {summary['laptop_data']['total_features']}")
        print(f"  Number of Brands: {summary['laptop_data']['brands_count']}")
        print(f"  Price Range (MYR): RM {summary['laptop_data']['price_range_myr']['min']:.2f} - RM {summary['laptop_data']['price_range_myr']['max']:.2f}")
        print(f"  Average Price (MYR): RM {summary['laptop_data']['price_range_myr']['mean']:.2f}")
        print(f"  Average Rating: {summary['laptop_data']['average_rating']:.2f}")
        
        # Display specification information
        if 'specifications' in summary['laptop_data']:
            print("\n  Specifications:")
            specs = summary['laptop_data']['specifications']
            if 'ram' in specs:
                ram_info = specs['ram']
                print(f"    RAM: {ram_info['found']}/{ram_info['total']} products found")
                if ram_info['mean_gb']:
                    print(f"      Range: {ram_info['min_gb']:.0f}GB - {ram_info['max_gb']:.0f}GB, Mean: {ram_info['mean_gb']:.1f}GB")
            
            if 'storage' in specs:
                storage_info = specs['storage']
                print(f"    Storage: {storage_info['found']}/{storage_info['total']} products found")
                if storage_info['mean_gb']:
                    print(f"      Range: {storage_info['min_gb']:.0f}GB - {storage_info['max_gb']:.0f}GB, Mean: {storage_info['mean_gb']:.1f}GB")
            
            if 'screen_size' in specs:
                screen_info = specs['screen_size']
                print(f"    Screen Size: {screen_info['found']}/{screen_info['total']} products found")
                if screen_info['mean_inches']:
                    print(f"      Range: {screen_info['min_inches']:.1f}\" - {screen_info['max_inches']:.1f}\", Mean: {screen_info['mean_inches']:.1f}\"")
        
        # Display column categories
        if 'column_categories' in summary['laptop_data']:
            print("\n  Column Categories:")
            categories = summary['laptop_data']['column_categories']
            for category, cols in categories.items():
                if cols:
                    print(f"    {category.replace('_', ' ').title()}: {len(cols)} columns")
                    if len(cols) <= 5:  # Show all if 5 or fewer
                        print(f"      {', '.join(cols)}")
                    else:  # Show first few if more than 5
                        print(f"      {', '.join(cols[:3])}... and {len(cols)-3} more")
        
        print("\nRATING DATA:")
        print(f"  Total Reviews: {summary['rating_data']['total_reviews']}")
        print(f"  Total Features: {summary['rating_data']['total_features']}")
        print(f"  Unique Users: {summary['rating_data']['unique_users']}")
        print(f"  Unique Products: {summary['rating_data']['unique_products']}")
        print(f"  Mean Rating: {summary['rating_data']['rating_stats']['mean']:.2f}")
        print(f"  Median Rating: {summary['rating_data']['rating_stats']['median']:.2f}")
        
        print("\nKey Features in Laptop Data:")
        laptop_cols = list(df_laptop.columns)
        print(f"  Product Info: {[col for col in laptop_cols if any(x in col for x in ['title', 'brand', 'os', 'color', 'store'])]}")
        print(f"  Pricing: {[col for col in laptop_cols if 'price' in col]}")
        print(f"  Ratings: {[col for col in laptop_cols if 'rating' in col]}")
        print(f"  Normalized: {[col for col in laptop_cols if any(x in col for x in ['encoded', 'clean', 'normalized'])]}")
        
        print("\nKey Features in Rating Data:")
        rating_cols = list(df_rating.columns)
        print(f"  User Info: {[col for col in rating_cols if 'user' in col]}")
        print(f"  Review Content: {[col for col in rating_cols if any(x in col for x in ['title', 'text', 'rating'])]}")
        print(f"  Normalized: {[col for col in rating_cols if any(x in col for x in ['encoded', 'clean', 'normalized'])]}")
        
        print("\nSeparated preprocessing completed successfully!")
        print("Dataframes are ready for use in memory.")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
