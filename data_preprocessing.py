"""
Data Preprocessing Module for Laptop Recommender System
Handles data loading, cleaning, and processing for Amazon laptop reviews enriched dataset
"""

import pandas as pd
import numpy as np
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
        self.brand_mapping = {
            "Brand_8": "Dell",
            "Brand_30": "Acer",
            "Brand_9": "HP",
            "Brand_7": "Lenovo",
            "Brand_10": "Apple",
            "Brand_11": "Asus",
            "Brand_12": "MSI",
            "Brand_13": "Samsung",
            # Add any other brand mappings here
        }
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the Amazon laptop reviews enriched dataset from Hugging Face or local file.
        
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
            logger.warning(f"Failed to load from Hugging Face: {e}")
            logger.info("Attempting to load from local file...")
            
            # Try to load from local file
            try:
                local_file = "data/amazon_laptop_reviews.csv"
                if os.path.exists(local_file):
                    self.df = pd.read_csv(local_file)
                    logger.info(f"Data loaded from local file. Shape: {self.df.shape}")
                    return self.df
                else:
                    # Create sample data if no local file exists
                    logger.info("Creating sample data for testing...")
                    self.df = self._create_sample_data()
                    return self.df
            except Exception as local_error:
                logger.error(f"Failed to load local data: {local_error}")
                logger.info("Creating sample data for testing...")
                self.df = self._create_sample_data()
                return self.df

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
            'images_y', 'videos'  # Include both images and videos
        ]
        
        rating_columns = [
            'asin', 'parent_asin', 'user_id', 'timestamp', 'rating', 'title_x',
            'text', 'helpful_vote'
        ]
        
        # Filter to only include columns that exist in the dataset
        available_laptop_cols = [col for col in laptop_columns if col in df.columns]
        available_rating_cols = [col for col in rating_columns if col in df.columns]
        
        # Log which media columns are available
        media_cols = [col for col in ['images_y', 'videos'] if col in df.columns]
        if media_cols:
            logger.info(f"Media columns found: {media_cols}")
        else:
            logger.warning("No media columns (images_y, videos) found in dataset")
        
        # Keep original brand names for display (before encoding)
        if 'brand' in df.columns:
            df['brand_original'] = df['brand']
        
        # Apply brand mapping before creating separate dataframes
        if 'brand' in df.columns:
            df['brand'] = df['brand'].map(self.brand_mapping).fillna(df['brand'])
        
        # Create separate dataframes
        # Use title_y for deduplication since same products can have different ASINs
        df_laptop = df[available_laptop_cols].drop_duplicates(subset=['title_y']).reset_index(drop=True)
        df_rating = df[available_rating_cols].reset_index(drop=True)
        
        # Add laptop_id as primary key (simple integer index)
        df_laptop['laptop_id'] = range(len(df_laptop))
        
        # Add brand_original column if it exists in the original data
        if 'brand_original' in df.columns:
            # Map brand_original back to laptop dataframe using asin
            brand_mapping = df[['asin', 'brand_original']].drop_duplicates(subset=['asin'])
            df_laptop = df_laptop.merge(brand_mapping, on='asin', how='left')
        
        # Create mapping from asin to laptop_id for rating dataframe
        asin_to_laptop_id = df_laptop.set_index('asin')['laptop_id'].to_dict()
        df_rating['laptop_id'] = df_rating['asin'].map(asin_to_laptop_id)
        
        # Remove ratings that don't have corresponding laptops (due to deduplication)
        df_rating = df_rating.dropna(subset=['laptop_id']).reset_index(drop=True)
        
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
        
        # 1. Normalize numerical columns (except ratings which should stay in 1-5 range)
        numerical_columns = ['rating_number']  # Removed average_rating to keep it in 1-5 range
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
        
        # Preserve brand_original column if it exists
        if 'brand_original' in df_normalized.columns:
            logger.info("Preserving brand_original column for display")
        
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
        
        # Add brand_original if it exists
        if 'brand_original' in df_normalized.columns:
            essential_columns.append('brand_original')
        
        # Add specification columns
        specification_columns = [col for col in df_normalized.columns if any(x in col for x in [
            'ram_gb', 'storage_gb', 'screen_size_inches', 'storage_type', 'ram_type', 
            'processor_model', 'gpu_model', 'storage_category', 'ram_category', 'screen_category',
            'storage_display', 'cpu_benchmark_score', 'gpu_benchmark_score', 'total_benchmark_score',
            'performance_tier', 'gaming_capability'
        ])]
        
        # Add media columns (images and videos)
        media_columns = [col for col in df_normalized.columns if col in ['images_y', 'videos']]
        
        # Add laptop_id column (created during separation)
        id_columns = [col for col in df_normalized.columns if col == 'laptop_id']
        
        final_columns = essential_columns + encoded_columns + clean_columns + numerical_columns + specification_columns + media_columns + id_columns
        available_final_columns = [col for col in final_columns if col in df_normalized.columns]
        
        df_final = df_normalized[available_final_columns]
        
        # Log which media columns were preserved
        preserved_media = [col for col in media_columns if col in df_final.columns]
        if preserved_media:
            logger.info(f"Media columns preserved: {preserved_media}")
        else:
            logger.warning("No media columns preserved in final dataframe")
        
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
        
        # 1. Normalize numerical columns (except ratings and helpful_vote which should stay as integers)
        numerical_columns = []  # Removed helpful_vote to keep it as integer values
        available_numerical = [col for col in numerical_columns if col in df_normalized.columns]
        
        if available_numerical:
            scaler = MinMaxScaler()
            df_normalized[available_numerical] = scaler.fit_transform(df_normalized[available_numerical].fillna(0))
            self.scalers['rating_numerical'] = scaler
        
        # Ensure helpful_vote is integer
        if 'helpful_vote' in df_normalized.columns:
            df_normalized['helpful_vote'] = df_normalized['helpful_vote'].fillna(0).astype(int)
        
        # 2. Preserve original user_id format (no encoding needed for string IDs)
        if 'user_id' in df_normalized.columns:
            # Keep original user_id as user_id_encoded for compatibility with existing code
            df_normalized['user_id_encoded'] = df_normalized['user_id'].fillna('unknown')
            logger.info(f"Preserved original user_id format. Sample IDs: {df_normalized['user_id_encoded'].head(3).tolist()}")
        
        # 3. Clean text columns
        text_columns = ['title_x', 'text']
        for col in text_columns:
            if col in df_normalized.columns:
                df_normalized[f'{col}_clean'] = df_normalized[col].astype(str).apply(self._clean_text)
        
        # 4. Convert timestamp to datetime features
        if 'timestamp' in df_normalized.columns:
            # Handle comma-separated timestamp format (e.g., "1,601,466,998,245")
            def parse_timestamp(ts):
                if pd.isna(ts) or ts is None:
                    return None
                try:
                    # Convert to string and remove commas
                    ts_str = str(ts).replace(',', '')
                    # Try to parse as Unix timestamp (milliseconds)
                    if ts_str.isdigit() and len(ts_str) > 10:
                        # Convert from milliseconds to seconds
                        ts_seconds = int(ts_str) / 1000
                        return pd.to_datetime(ts_seconds, unit='s')
                    else:
                        # Try regular datetime parsing
                        return pd.to_datetime(ts_str, errors='coerce')
                except (ValueError, TypeError):
                    return None
            
            df_normalized['timestamp'] = df_normalized['timestamp'].apply(parse_timestamp)
            # Format timestamp to DD/MM/YYYY hh:mm format
            df_normalized['timestamp'] = df_normalized['timestamp'].dt.strftime('%d/%m/%Y %H:%M')
            df_normalized['year'] = df_normalized['timestamp'].apply(lambda x: int(x.split('/')[2].split(' ')[0]) if pd.notna(x) else None)
            df_normalized['month'] = df_normalized['timestamp'].apply(lambda x: int(x.split('/')[1]) if pd.notna(x) else None)
            df_normalized['day_of_week'] = df_normalized['timestamp'].apply(lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M').dayofweek if pd.notna(x) else None)
        
        # 5. Add user activity counts (views, ratings, comments)
        logger.info("Adding user activity counts...")
        df_normalized = self._add_user_activity_counts(df_normalized)
        
        # 6. Keep only essential columns and cleaned versions
        essential_columns = ['asin', 'parent_asin', 'user_id_encoded', 'timestamp', 'rating', 'helpful_vote']
        clean_columns = [col for col in df_normalized.columns if col.endswith('_clean')]
        temporal_columns = ['year', 'month', 'day_of_week']
        activity_columns = ['user_views_count', 'user_ratings_count', 'user_comments_count']
        
        final_columns = essential_columns + clean_columns + temporal_columns + activity_columns
        available_final_columns = [col for col in final_columns if col in df_normalized.columns]
        
        df_final = df_normalized[available_final_columns]
        
        logger.info("Rating data normalization completed")
        return df_final

    def _add_user_activity_counts(self, df_rating: pd.DataFrame) -> pd.DataFrame:
        """
        Add user activity counts (views, ratings, comments) to the rating dataframe.
        
        Args:
            df_rating (pd.DataFrame): Rating dataframe with user_id_encoded column
            
        Returns:
            pd.DataFrame: Rating dataframe with added user activity counts
        """
        logger.info("Calculating user activity counts...")
        
        df_with_counts = df_rating.copy()
        
        # Ensure user_id_encoded exists
        if 'user_id_encoded' not in df_with_counts.columns:
            logger.warning("user_id_encoded column not found, skipping user activity counts")
            return df_with_counts
        
        # Calculate user activity counts
        user_activity = df_with_counts.groupby('user_id_encoded').agg({
            'rating': 'count',  # Count of ratings per user
            'text': lambda x: (x.notna() & (x.str.len() > 10)).sum()  # Count of meaningful comments per user
        }).rename(columns={
            'rating': 'user_ratings_count',
            'text': 'user_comments_count'
        })
        
        # For views, we'll use the same count as ratings since views data might not exist
        # This is as requested by the user - make views count same as ratings count
        user_activity['user_views_count'] = user_activity['user_ratings_count']
        
        # Reset index to make user_id_encoded a column
        user_activity = user_activity.reset_index()
        
        # Merge the activity counts back to the main dataframe
        df_with_counts = df_with_counts.merge(
            user_activity, 
            on='user_id_encoded', 
            how='left'
        )
        
        # Fill any missing values with 0
        activity_columns = ['user_views_count', 'user_ratings_count', 'user_comments_count']
        for col in activity_columns:
            if col in df_with_counts.columns:
                df_with_counts[col] = df_with_counts[col].fillna(0).astype(int)
        
        # Log some statistics
        if 'user_ratings_count' in df_with_counts.columns:
            logger.info(f"User activity statistics:")
            logger.info(f"  Average ratings per user: {df_with_counts['user_ratings_count'].mean():.2f}")
            logger.info(f"  Average comments per user: {df_with_counts['user_comments_count'].mean():.2f}")
            logger.info(f"  Average views per user: {df_with_counts['user_views_count'].mean():.2f}")
            logger.info(f"  Total unique users: {df_with_counts['user_id_encoded'].nunique()}")
        
        return df_with_counts

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
            # Handle features carefully to preserve structure
            if df_clean['features'].dtype == 'object':
                # Check if it's already a string or needs conversion
                sample_feature = df_clean['features'].iloc[0] if len(df_clean) > 0 else None
                if hasattr(sample_feature, '__iter__') and not isinstance(sample_feature, str):
                    # Convert numpy arrays to clean strings
                    df_clean['features_clean'] = df_clean['features'].apply(self._process_features)
                else:
                    df_clean['features_clean'] = df_clean['features'].astype(str)
            else:
                df_clean['features_clean'] = df_clean['features'].astype(str)
        
        if 'details' in df_clean.columns:
            df_clean['details_parsed'] = df_clean['details'].apply(self._parse_details)
        
        # 7. Remove duplicate rows
        initial_rows = len(df_clean)
        
        # Convert numpy arrays to strings to make them hashable for drop_duplicates
        # BUT preserve media columns (images_y, videos) to maintain their structure
        media_columns = ['images_y', 'videos']
        for col in df_clean.columns:
            if col in media_columns:
                # Skip media columns - preserve their structure
                continue
            if df_clean[col].dtype == 'object':
                # Check if column contains numpy arrays
                sample_val = df_clean[col].iloc[0] if len(df_clean) > 0 else None
                if hasattr(sample_val, '__iter__') and not isinstance(sample_val, str):
                    df_clean[col] = df_clean[col].astype(str)
        
        # Handle duplicates differently for media columns
        # First, create a copy without media columns for duplicate detection
        df_for_duplicates = df_clean.drop(columns=media_columns, errors='ignore')
        df_for_duplicates = df_for_duplicates.drop_duplicates()
        
        # Then merge back the media columns
        if media_columns:
            media_data = df_clean[['asin'] + media_columns].drop_duplicates(subset=['asin'])
            df_clean = df_for_duplicates.merge(media_data, on='asin', how='left')
        else:
            df_clean = df_for_duplicates
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
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample laptop data for testing when external data is not available.
        
        Returns:
            pd.DataFrame: Sample dataset
        """
        import random
        
        # Sample laptop data
        laptops = [
            {
                'asin': 'B08N5WRWNW',
                'title_y': 'Dell XPS 13 9310 Laptop, 13.4-inch FHD+ Display, Intel Core i7-1165G7, 16GB RAM, 512GB SSD',
                'brand': 'Dell',
                'price': '$1,299.99',
                'average_rating': 4.5,
                'rating_number': 1250,
                'features': 'Intel Core i7, 16GB RAM, 512GB SSD, 13.4-inch FHD+ Display',
                'images_y': ['https://example.com/dell-xps-13-1.jpg', 'https://example.com/dell-xps-13-2.jpg'],
                'videos': ['https://example.com/dell-xps-13-demo.mp4']
            },
            {
                'asin': 'B08N5WRWNW',
                'title_y': 'MacBook Pro 13-inch, Apple M1 Chip, 8GB RAM, 256GB SSD',
                'brand': 'Apple',
                'price': '$1,299.00',
                'average_rating': 4.8,
                'rating_number': 2100,
                'features': 'Apple M1 Chip, 8GB RAM, 256GB SSD, 13-inch Retina Display',
                'images_y': ['https://example.com/macbook-pro-1.jpg', 'https://example.com/macbook-pro-2.jpg'],
                'videos': ['https://example.com/macbook-pro-review.mp4']
            },
            {
                'asin': 'B08N5WRWNW',
                'title_y': 'HP Spectre x360 13.3-inch 4K OLED Touch-Screen Laptop, Intel Core i7-1165G7',
                'brand': 'HP',
                'price': '$1,399.99',
                'average_rating': 4.3,
                'rating_number': 890,
                'features': 'Intel Core i7, 16GB RAM, 512GB SSD, 13.3-inch 4K OLED Display',
                'images_y': ['https://example.com/hp-spectre-1.jpg', 'https://example.com/hp-spectre-2.jpg'],
                'videos': ['https://example.com/hp-spectre-unboxing.mp4']
            },
            {
                'asin': 'B08N5WRWNW',
                'title_y': 'Lenovo ThinkPad X1 Carbon 9th Gen, 14-inch FHD Display, Intel Core i7-1165G7',
                'brand': 'Lenovo',
                'price': '$1,599.99',
                'average_rating': 4.6,
                'rating_number': 1560,
                'features': 'Intel Core i7, 16GB RAM, 1TB SSD, 14-inch FHD Display',
                'images_y': ['https://example.com/thinkpad-x1-1.jpg', 'https://example.com/thinkpad-x1-2.jpg'],
                'videos': ['https://example.com/thinkpad-x1-review.mp4']
            },
            {
                'asin': 'B08N5WRWNW',
                'title_y': 'ASUS ROG Zephyrus G14, 14-inch QHD Display, AMD Ryzen 9 5900HS, RTX 3060',
                'brand': 'ASUS',
                'price': '$1,449.99',
                'average_rating': 4.4,
                'rating_number': 1120,
                'features': 'AMD Ryzen 9, 16GB RAM, 1TB SSD, 14-inch QHD Display, RTX 3060',
                'images_y': ['https://example.com/asus-rog-1.jpg', 'https://example.com/asus-rog-2.jpg'],
                'videos': ['https://example.com/asus-rog-gaming.mp4']
            }
        ]
        
        # Create sample data with reviews
        sample_data = []
        for laptop in laptops:
            # Create multiple reviews for each laptop
            for i in range(random.randint(50, 200)):
                sample_data.append({
                    'asin': laptop['asin'],
                    'parent_asin': laptop['asin'],
                    'title_x': laptop['title_y'],
                    'title_y': laptop['title_y'],
                    'brand': laptop['brand'],
                    'price': laptop['price'],
                    'average_rating': laptop['average_rating'],
                    'rating_number': laptop['rating_number'],
                    'features': laptop['features'],
                    'images_y': laptop['images_y'],
                    'videos': laptop['videos'],
                    'rating': random.uniform(3.5, 5.0),
                    'user_id': f'user_{random.randint(1000, 9999)}',
                    'timestamp': f'2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}',
                    'text': f'Great laptop with {laptop["features"].split(",")[0].lower()}. Highly recommended!',
                    'helpful_vote': random.randint(0, 50),
                    'verified_purchase': random.choice([True, False]),
                    'os': random.choice(['Windows 10', 'Windows 11', 'macOS']),
                    'color': random.choice(['Silver', 'Black', 'Space Gray']),
                    'store': 'Amazon'
                })
        
        df = pd.DataFrame(sample_data)
        logger.info(f"Created sample data with {len(df)} records")
        return df

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML tags and special characters.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == 'nan':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation and letters
        text = re.sub(r'[^\w\s.,!?()-]', '', text)
        
        # Clean up common text issues
        text = text.replace('  ', ' ')  # Double spaces
        text = text.replace(' ,', ',')  # Space before comma
        text = text.replace(' .', '.')  # Space before period
        
        # Limit length to prevent weird truncation
        if len(text) > 200:
            # Try to cut at a word boundary
            words = text[:200].split()
            if len(words) > 1:
                text = ' '.join(words[:-1]) + '...'
            else:
                text = text[:200] + '...'
        
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
        
        # 3. Create text length features
        df_enhanced['review_length'] = df_enhanced['text'].str.len()
        df_enhanced['title_length'] = df_enhanced['title_x'].str.len()
        
        # 4. Create helpfulness ratio
        if 'helpful_vote' in df_enhanced.columns and 'num_reviews' in df_enhanced.columns:
            df_enhanced['helpfulness_ratio'] = (
                df_enhanced['helpful_vote'] / df_enhanced['num_reviews'].replace(0, 1)
            ).fillna(0)
        
        # 5. Create brand popularity
        if 'brand' in df_enhanced.columns:
            brand_counts = df_enhanced['brand'].value_counts()
            df_enhanced['brand_popularity'] = df_enhanced['brand'].map(brand_counts)
        
        logger.info("Derived features added successfully")
        return df_enhanced
    
    
    def _extract_screen_size_from_text(self, text: str) -> Optional[float]:
        """
        Extract screen size from text using comprehensive regex patterns.
        This method searches for screen size information in laptop titles and descriptions.
        
        Args:
            text (str): Text containing screen size information
            
        Returns:
            Optional[float]: Screen size in inches, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # Comprehensive screen size patterns to catch various formats
        screen_patterns = [
            # Standard formats with hyphen (e.g., "13-inch", "14-inch")
            r'(\d+(?:\.\d+)?)-inch',  # 13-inch, 14-inch, 15.6-inch
            r'(\d+(?:\.\d+)?)\s*inch',  # 15.6 inch
            r'(\d+(?:\.\d+)?)\s*"',  # 15.6"
            r'(\d+(?:\.\d+)?)\s*in',  # 15.6 in
            r'(\d+(?:\.\d+)?)\s*inches',  # 15.6 inches
            
            # Formats with display type (e.g., "15.6 Full HD", "15.6 144hz")
            r'(\d+(?:\.\d+)?)\s+(?:full\s+hd|fhd|qhd|uhd|4k|144hz|120hz|60hz|ips|oled|led)',  # 15.6 Full HD
            r'(\d+(?:\.\d+)?)\s+(?:display|screen|monitor)',  # 15.6 Display
            
            # Formats with resolution (e.g., "15.6 1920x1080")
            r'(\d+(?:\.\d+)?)\s+\d+x\d+',  # 15.6 1920x1080
            
            # Formats with refresh rate (e.g., "15.6 144Hz")
            r'(\d+(?:\.\d+)?)\s+\d+hz',  # 15.6 144Hz
            
            # Formats with panel type (e.g., "15.6 IPS", "15.6 OLED")
            r'(\d+(?:\.\d+)?)\s+(?:ips|oled|led|tn|va)',  # 15.6 IPS
            
            # Edge case: just the number followed by space and any word (common in titles)
            r'(\d+(?:\.\d+)?)\s+\w+',  # 15.6 Gaming, 15.6 Touch, etc.
        ]
        
        # Try each pattern and return the first match
        for pattern in screen_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                screen_size = float(matches[0])
                # Validate that it's a reasonable screen size (between 10 and 20 inches)
                if 10.0 <= screen_size <= 20.0:
                    return screen_size
        
        return None
    
    def _extract_ram_from_text(self, text: str) -> Optional[float]:
        """
        Extract RAM capacity from text using regex patterns.
        
        Args:
            text (str): Text containing RAM information
            
        Returns:
            Optional[float]: RAM capacity in GB, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # RAM patterns with various formats
        ram_patterns = [
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:ddr\d*|ram|memory)',  # 8GB DDR4, 16GB RAM
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:ddr\d*)',  # 8GB DDR4
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:ram)',  # 8GB RAM
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:memory)',  # 8GB Memory
            r'(\d+(?:\.\d+)?)\s*gb',  # 8GB (fallback)
            r'(\d+(?:\.\d+)?)\s*tb\s*(?:ddr\d*|ram|memory)',  # 1TB DDR4
            r'(\d+(?:\.\d+)?)\s*tb',  # 1TB (fallback)
        ]
        
        for pattern in ram_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                value = float(matches[0])
                # Convert TB to GB
                if 'tb' in text_str and 'gb' not in text_str:
                    value *= 1024
                return value
        
        return None
    
    def _extract_storage_from_text(self, text: str) -> Optional[float]:
        """
        Extract storage capacity from text using regex patterns.
        
        Args:
            text (str): Text containing storage information
            
        Returns:
            Optional[float]: Storage capacity in GB, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # Storage patterns with various formats - prioritize storage-specific terms
        storage_patterns = [
            # High priority: explicit storage terms with capacity
            r'(\d+(?:\.\d+)?)\s*tb\s*(?:ssd|hdd|hard\s*drive|storage|flash\s*storage|nvme|pcie)',  # 1TB SSD, 2TB NVMe
            r'(\d+(?:\.\d+)?)\s*tb\s*(?:ssd|hdd|nvme|pcie)',  # 1TB SSD, 2TB NVMe
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:ssd|hdd|hard\s*drive|storage|flash\s*storage|nvme|pcie)',  # 512GB SSD, 1TB HDD
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:ssd|hdd|nvme|pcie)',  # 512GB SSD
            
            # Medium priority: storage with less specific terms
            r'(\d+(?:\.\d+)?)\s*tb\s*(?:hard\s*drive|storage)',  # 1TB hard drive
            r'(\d+(?:\.\d+)?)\s*gb\s*(?:hard\s*drive|storage)',  # 512GB hard drive
            
            # Lower priority: just numbers with storage context
            r'(\d+(?:\.\d+)?)\s*tb',  # 1TB (fallback, but check context)
            r'(\d+(?:\.\d+)?)\s*gb',  # 512GB (fallback, but check context)
        ]
        
        # First try high-priority patterns
        for i, pattern in enumerate(storage_patterns):
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                value = float(matches[0])
                
                # Check if this pattern matched TB or GB
                # Look for the pattern in the original text to see the unit
                pattern_match = re.search(pattern, text_str, re.IGNORECASE)
                if pattern_match:
                    matched_text = pattern_match.group(0).lower()
                    # If the pattern contains TB, convert to GB
                    if 'tb' in matched_text and 'gb' not in matched_text[:matched_text.find('tb')]:
                        value *= 1024
                
                # For lower priority patterns, verify it's actually storage
                if i >= 6:  # Lower priority patterns
                    # Check if this might be RAM instead of storage
                    if any(ram_term in text_str for ram_term in ['ram', 'memory', 'ddr']):
                        continue  # Skip this match, it's likely RAM
                
                return value
        
        return None
    
    def _extract_processor_name_from_text(self, text: str) -> Optional[str]:
        """
        Extract processor model from text using regex patterns.
        
        Args:
            text (str): Text containing processor information
            
        Returns:
            Optional[str]: Processor model, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # Processor patterns - comprehensive patterns to capture complete processor names
        processor_patterns = [
            # Intel Core i series - various formats
            r'(intel\s+core\s+i[3579]-\d+[a-z]*\d*)',  # Intel Core i7-5950HQ, i5-1135G7
            r'(intel\s+core\s+i[3579]\s+\d+[a-z]*\d*)',  # Intel Core i7 5950HQ
            r'(core\s+i[3579]-\d+[a-z]*\d*)',  # Core i7-5950HQ
            r'(i[3579]-\d+[a-z]*\d*)',  # i7-5950HQ
            
            # Intel Pentium and Celeron
            r'(intel\s+pentium\s+\w+)',  # Intel Pentium Gold
            r'(intel\s+celeron\s+\w+)',  # Intel Celeron N4020
            r'(pentium\s+\w+)',  # Pentium Gold
            r'(celeron\s+\w+)',  # Celeron N4020
            
            # AMD Ryzen series
            r'(amd\s+ryzen\s+[3579]\s+\d+[a-z]*\d*)',  # AMD Ryzen 5 5500U
            r'(ryzen\s+[3579]\s+\d+[a-z]*\d*)',  # Ryzen 5 5500U
            r'(amd\s+ryzen\s+\d+[a-z]*\d*)',  # AMD Ryzen 5500U
            r'(ryzen\s+\d+[a-z]*\d*)',  # Ryzen 5500U
            
            # AMD Athlon and other AMD processors
            r'(amd\s+athlon\s+\w+)',  # AMD Athlon Silver
            r'(athlon\s+\w+)',  # Athlon Silver
            
            # Apple processors
            r'(apple\s+m\d+)',  # Apple M1, Apple M2
            r'(m\d+)',  # M1, M2
            
            # Generic patterns
            r'(intel\s+\w+)',  # Intel something
            r'(amd\s+\w+)',  # AMD something
        ]
        
        for pattern in processor_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                # Return the first match, cleaned up
                processor = matches[0].strip()
                return processor.title()  # Capitalize properly
        
        return None
    
    def _extract_gpu_name_from_text(self, text: str) -> Optional[str]:
        """
        Extract GPU model from text using regex patterns.
        
        Args:
            text (str): Text containing GPU information
            
        Returns:
            Optional[str]: GPU model, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # GPU patterns - comprehensive patterns to capture GPU names
        gpu_patterns = [
            # NVIDIA RTX series
            r'(nvidia\s+geforce\s+rtx\s+\d+\s*(?:ti|super)?)',  # NVIDIA GeForce RTX 3060 Ti
            r'(geforce\s+rtx\s+\d+\s*(?:ti|super)?)',  # GeForce RTX 3060 Ti
            r'(rtx\s+\d+\s*(?:ti|super)?)',  # RTX 3060 Ti
            
            # NVIDIA GTX series
            r'(nvidia\s+geforce\s+gtx\s+\d+\s*(?:ti|super)?)',  # NVIDIA GeForce GTX 1660 Ti
            r'(geforce\s+gtx\s+\d+\s*(?:ti|super)?)',  # GeForce GTX 1660 Ti
            r'(gtx\s+\d+\s*(?:ti|super)?)',  # GTX 1660 Ti
            
            # AMD Radeon series
            r'(amd\s+radeon\s+rx\s+\d+\s*(?:xt|xtx)?)',  # AMD Radeon RX 6600 XT
            r'(radeon\s+rx\s+\d+\s*(?:xt|xtx)?)',  # Radeon RX 6600 XT
            r'(rx\s+\d+\s*(?:xt|xtx)?)',  # RX 6600 XT
            
            # Intel Arc series
            r'(intel\s+arc\s+a\d+)',  # Intel Arc A770
            r'(arc\s+a\d+)',  # Arc A770
            
            # Integrated graphics
            r'(intel\s+iris\s+xe)',  # Intel Iris Xe
            r'(iris\s+xe)',  # Iris Xe
            r'(intel\s+uhd\s+graphics)',  # Intel UHD Graphics
            r'(uhd\s+graphics)',  # UHD Graphics
            r'(amd\s+radeon\s+graphics)',  # AMD Radeon Graphics
            r'(radeon\s+graphics)',  # Radeon Graphics
            
            # Generic patterns
            r'(nvidia\s+\w+)',  # NVIDIA something
            r'(amd\s+\w+)',  # AMD something
            r'(intel\s+\w+)',  # Intel something
        ]
        
        for pattern in gpu_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                # Return the first match, cleaned up
                gpu = matches[0].strip()
                return gpu.title()  # Capitalize properly
        
        return None
    
    def _extract_storage_type_from_text(self, text: str) -> Optional[str]:
        """
        Extract storage type from text using regex patterns.
        
        Args:
            text (str): Text containing storage type information
            
        Returns:
            Optional[str]: Storage type, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # Storage type patterns
        storage_types = ['nvme', 'ssd', 'hdd', 'pcie', 'emmc']
        
        for storage_type in storage_types:
            if storage_type in text_str:
                return storage_type.upper()
        
        return None
    
    def _extract_ram_type_from_text(self, text: str) -> Optional[str]:
        """
        Extract RAM type from text using regex patterns.
        
        Args:
            text (str): Text containing RAM type information
            
        Returns:
            Optional[str]: RAM type, None if not found
        """
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # RAM type patterns
        ram_types = ['ddr5', 'ddr4', 'ddr3', 'lpddr5', 'lpddr4', 'lpddr3']
        
        for ram_type in ram_types:
            if ram_type in text_str:
                return ram_type.upper()
        
        return None

    def save_cached_data(self, df_laptop: pd.DataFrame, df_rating: pd.DataFrame, 
                        cache_dir: str = "data/cache") -> None:
        """
        Save preprocessed data to cache files for faster loading.
        
        Args:
            df_laptop: Processed laptop dataframe
            df_rating: Processed rating dataframe
            cache_dir: Directory to save cache files
        """
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Save dataframes as parquet files (more efficient than CSV)
            laptop_cache_path = os.path.join(cache_dir, "laptop_data.parquet")
            rating_cache_path = os.path.join(cache_dir, "rating_data.parquet")
            metadata_path = os.path.join(cache_dir, "cache_metadata.json")
            
            # Save dataframes (no need for data type conversion since we use numeric values)
            df_laptop.to_parquet(laptop_cache_path, index=False)
            df_rating.to_parquet(rating_cache_path, index=False)
            
            # Save metadata with timestamp
            metadata = {
                "created_at": datetime.now().isoformat(),
                "laptop_records": len(df_laptop),
                "rating_records": len(df_rating),
                "laptop_columns": list(df_laptop.columns),
                "rating_columns": list(df_rating.columns),
                "version": "1.0"
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Cached data saved to {cache_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save cached data: {e}")
    
    def load_cached_data(self, cache_dir: str = "data/cache", 
                        max_age_hours: int = 24) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Load preprocessed data from cache files if they exist and are fresh.
        
        Args:
            cache_dir: Directory containing cache files
            max_age_hours: Maximum age of cache in hours before considering stale
            
        Returns:
            Tuple of (df_laptop, df_rating) if cache is valid, None otherwise
        """
        try:
            laptop_cache_path = os.path.join(cache_dir, "laptop_data.parquet")
            rating_cache_path = os.path.join(cache_dir, "rating_data.parquet")
            metadata_path = os.path.join(cache_dir, "cache_metadata.json")
            
            # Check if all cache files exist
            if not all(os.path.exists(path) for path in [laptop_cache_path, rating_cache_path, metadata_path]):
                logger.info("Cache files not found, will need to preprocess")
                return None
            
            # Check cache age
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            created_at = datetime.fromisoformat(metadata['created_at'])
            age_hours = (datetime.now() - created_at).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                logger.info(f"Cache is {age_hours:.1f} hours old (max: {max_age_hours}), will reprocess")
                return None
            
            # Load cached data
            df_laptop = pd.read_parquet(laptop_cache_path)
            df_rating = pd.read_parquet(rating_cache_path)
            
            # Set the data in the preprocessor
            self.df_laptop = df_laptop
            self.df_rating = df_rating
            
            logger.info(f"Loaded cached data: {len(df_laptop)} laptops, {len(df_rating)} ratings")
            return df_laptop, df_rating
            
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
            return None
    
    def clear_cache(self, cache_dir: str = "data/cache") -> None:
        """
        Clear all cached data files.
        
        Args:
            cache_dir: Directory containing cache files
        """
        try:
            import shutil
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                logger.info(f"Cache cleared: {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def preprocess_separated_pipeline(self, force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete preprocessing pipeline with separated dataframes.
        Checks for cached data first to avoid reprocessing.
        
        Args:
            force_reprocess: If True, force reprocessing even if cached data exists
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_laptop, df_rating)
        """
        # Check for cached data first
        if not force_reprocess:
            cached_data = self.load_cached_data()
            if cached_data is not None:
                logger.info("Using cached preprocessed data")
                return cached_data
        
        logger.info("Starting separated preprocessing pipeline...")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Clean data
        cleaned_data = self.clean_data()
        
        # Step 3: Separate dataframes
        df_laptop, df_rating = self.separate_dataframes(cleaned_data)
        
        # Step 4: Add price conversion
        df_laptop = self.add_price_conversion(df_laptop)
        
        # Step 5: Add benchmark scores
        df_laptop = self.add_benchmark_scores(df_laptop)

        # Step 6: Add specifications
        df_laptop = self.add_specifications(df_laptop)
        
        # Step 7: Normalize data
        df_laptop_normalized = self.normalize_laptop_data(df_laptop)
        df_rating_normalized = self.normalize_rating_data(df_rating)
        
        self.df_laptop = df_laptop_normalized
        self.df_rating = df_rating_normalized
        
        
        # Save processed data to cache
        self.save_cached_data(df_laptop_normalized, df_rating_normalized)
        
        logger.info("Separated preprocessing pipeline completed successfully")
        
        return df_laptop_normalized, df_rating_normalized

    def add_benchmark_scores(self, df_laptop: pd.DataFrame) -> pd.DataFrame:
        """
        Add CPU and GPU benchmark scores using the benchmark scraper.
        
        Args:
            df_laptop (pd.DataFrame): Laptop dataframe
            
        Returns:
            pd.DataFrame: Laptop dataframe with added benchmark scores
        """
        try:
            from benchmark_scraper import BenchmarkScraper
            
            logger.info("Initializing benchmark scraper for CPU/GPU benchmark extraction...")
            scraper = BenchmarkScraper(preprocessor=self)
            
            # Extract only CPU and GPU benchmark scores
            # Note: This can take several minutes for large datasets
            logger.info("Starting CPU/GPU benchmark score extraction (this may take 2-5 minutes)...")
            df_with_benchmarks = scraper.add_benchmark_scores(df_laptop)
            
            # Replace benchmark scores of 0 with default values for unmatched CPU/GPU
            if 'cpu_benchmark_score' in df_with_benchmarks.columns:
                df_with_benchmarks['cpu_benchmark_score'] = df_with_benchmarks['cpu_benchmark_score'].replace(0, 3000)
            if 'gpu_benchmark_score' in df_with_benchmarks.columns:
                df_with_benchmarks['gpu_benchmark_score'] = df_with_benchmarks['gpu_benchmark_score'].replace(0, 500)
            if 'total_benchmark_score' in df_with_benchmarks.columns:
                # For total score, replace 0 with default value only if both CPU and GPU are 0
                mask = (df_with_benchmarks['cpu_benchmark_score'] == 0) & (df_with_benchmarks['gpu_benchmark_score'] == 0)
                df_with_benchmarks.loc[mask, 'total_benchmark_score'] = 3000 * 0.7 + 500 * 0.3  # Weighted combination
            
            logger.info("CPU/GPU benchmark scores extracted successfully using benchmark scraper")
            return df_with_benchmarks
            
        except ImportError as e:
            logger.warning(f"Could not import benchmark scraper: {e}")
            logger.info("Falling back to basic benchmark extraction...")
            return self._add_basic_benchmarks(df_laptop)
        except Exception as e:
            logger.error(f"Error in benchmark scraper: {e}")
            logger.info("Falling back to basic benchmark extraction...")
            return self._add_basic_benchmarks(df_laptop)
        except KeyboardInterrupt:
            logger.warning("Benchmark processing interrupted by user")
            logger.info("Falling back to basic benchmark extraction...")
            return self._add_basic_benchmarks(df_laptop)
    
    def add_specifications(self, df_laptop: pd.DataFrame) -> pd.DataFrame:
        """
        Add laptop specifications (RAM, storage, screen size, processor, GPU, etc.).
        
        Args:
            df_laptop (pd.DataFrame): Laptop dataframe
            
        Returns:
            pd.DataFrame: Laptop dataframe with added specifications
        """
        logger.info("Adding laptop specifications (RAM, storage, screen size, processor, GPU)...")
        
        df_specs = df_laptop.copy()
        
        # Combine text from all relevant columns for each row
        def combine_text_columns(row):
            text_parts = []
            
            # Add title_y if it exists
            if 'title_y' in row.index and pd.notna(row['title_y']):
                text_parts.append(str(row['title_y']))
            
            # Add features if it exists
            if 'features' in row.index and pd.notna(row['features']):
                text_parts.append(str(row['features']))
            
            # Add details if it exists
            if 'details' in row.index and pd.notna(row['details']):
                text_parts.append(str(row['details']))
            
            # Add details_parsed if it exists (processed details)
            if 'details_parsed' in row.index and pd.notna(row['details_parsed']):
                details_text = str(row['details_parsed'])
                text_parts.append(details_text)
            
            return ' '.join(text_parts)
        
        # Extract specifications for each row
        logger.info("Extracting RAM specifications...")
        df_specs['ram_gb'] = df_specs.apply(
            lambda row: self._extract_ram_from_text(combine_text_columns(row)), axis=1
        )
        
        logger.info("Extracting storage specifications...")
        df_specs['storage_gb'] = df_specs.apply(
            lambda row: self._extract_storage_from_text(combine_text_columns(row)), axis=1
        )
        
        logger.info("Extracting screen size specifications...")
        df_specs['screen_size_inches'] = df_specs.apply(
            lambda row: self._extract_screen_size_from_text(combine_text_columns(row)), axis=1
        )
        
        logger.info("Extracting processor specifications...")
        df_specs['processor_model'] = df_specs.apply(
            lambda row: self._extract_processor_name_from_text(combine_text_columns(row)), axis=1
        )
        
        logger.info("Extracting GPU specifications...")
        df_specs['gpu_model'] = df_specs.apply(
            lambda row: self._extract_gpu_name_from_text(combine_text_columns(row)), axis=1
        )
        
        logger.info("Extracting storage type specifications...")
        df_specs['storage_type'] = df_specs.apply(
            lambda row: self._extract_storage_type_from_text(combine_text_columns(row)), axis=1
        )
        
        logger.info("Extracting RAM type specifications...")
        df_specs['ram_type'] = df_specs.apply(
            lambda row: self._extract_ram_type_from_text(combine_text_columns(row)), axis=1
        )
        
        # Add performance tiers and gaming capability based on benchmark scores
        if 'cpu_benchmark_score' in df_specs.columns and 'gpu_benchmark_score' in df_specs.columns:
            logger.info("Adding performance tiers and gaming capability...")
            df_specs['performance_tier'] = df_specs.apply(self._calculate_performance_tier, axis=1)
            df_specs['gaming_capability'] = df_specs.apply(self._calculate_gaming_capability, axis=1)
        
        # Log extraction results
        logger.info("Specification extraction completed:")
        logger.info(f"  RAM found: {df_specs['ram_gb'].notna().sum()}/{len(df_specs)} rows")
        logger.info(f"  Storage found: {df_specs['storage_gb'].notna().sum()}/{len(df_specs)} rows")
        logger.info(f"  Screen size found: {df_specs['screen_size_inches'].notna().sum()}/{len(df_specs)} rows")
        logger.info(f"  Processor found: {df_specs['processor_model'].notna().sum()}/{len(df_specs)} rows")
        logger.info(f"  GPU found: {df_specs['gpu_model'].notna().sum()}/{len(df_specs)} rows")
        logger.info(f"  Storage type found: {df_specs['storage_type'].notna().sum()}/{len(df_specs)} rows")
        logger.info(f"  RAM type found: {df_specs['ram_type'].notna().sum()}/{len(df_specs)} rows")
        
        return df_specs
    
    def _calculate_performance_tier(self, row) -> str:
        """Calculate performance tier based on CPU and GPU benchmark scores."""
        cpu_score = row.get('cpu_benchmark_score', 0)
        gpu_score = row.get('gpu_benchmark_score', 0)
        
        if cpu_score >= 20000 and gpu_score >= 15000:
            return 'Ultra High'
        elif cpu_score >= 15000 and gpu_score >= 10000:
            return 'High'
        elif cpu_score >= 10000 and gpu_score >= 5000:
            return 'Medium-High'
        elif cpu_score >= 5000 and gpu_score >= 2000:
            return 'Medium'
        elif cpu_score >= 3000 and gpu_score >= 500:
            return 'Low-Medium'
        else:
            return 'Low'
    
    def _calculate_gaming_capability(self, row) -> str:
        """Calculate gaming capability based on GPU benchmark score."""
        gpu_score = row.get('gpu_benchmark_score', 0)
        
        if gpu_score >= 15000:
            return 'High-End Gaming'
        elif gpu_score >= 10000:
            return 'Mid-Range Gaming'
        elif gpu_score >= 5000:
            return 'Casual Gaming'
        elif gpu_score >= 2000:
            return 'Light Gaming'
        else:
            return 'Basic Graphics'
    
    def _add_basic_benchmarks(self, df_laptop: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic CPU/GPU benchmark scores using built-in methods.
        
        Args:
            df_laptop (pd.DataFrame): Laptop dataframe
            
        Returns:
            pd.DataFrame: Laptop dataframe with added benchmark scores
        """
        logger.info("Adding basic CPU/GPU benchmark scores using built-in methods...")
        
        df_benchmarks = df_laptop.copy()
        
        # Add default benchmark scores
        df_benchmarks['cpu_benchmark_score'] = 3000  # Default CPU score
        df_benchmarks['gpu_benchmark_score'] = 500   # Default GPU score
        df_benchmarks['total_benchmark_score'] = 3000 * 0.7 + 500 * 0.3  # Weighted combination
        
        logger.info("Basic benchmark scores added successfully")
        return df_benchmarks


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
