"""
Hugging Face SQL Client for Laptop Recommender System

This module provides direct SQL query access to the Hugging Face dataset
for efficient user search and filtering operations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datasets import load_dataset
import sqlite3
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceSQLClient:
    """
    Client for executing SQL queries directly on Hugging Face datasets.
    Provides efficient user search and filtering capabilities.
    """
    
    def __init__(self, dataset_name: str = "naga-jay/amazon-laptop-reviews-enriched", 
                 cache_dir: str = "data/hf_cache"):
        """
        Initialize the Hugging Face SQL client.
        
        Args:
            dataset_name: Name of the Hugging Face dataset
            cache_dir: Directory to cache the dataset locally
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.dataset = None
        self.df_rating = None
        self.df_laptop = None
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        """Initialize and load the Hugging Face dataset."""
        try:
            logger.info(f"Loading dataset {self.dataset_name} from Hugging Face...")
            
            # Load dataset with caching
            self.dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir
            )
            
            # Convert to pandas DataFrame
            self.df_rating = self.dataset['train'].to_pandas()
            
            # Separate laptop and rating data
            self._separate_dataframes()
            
            logger.info(f"Dataset loaded successfully. Rating data: {self.df_rating.shape}")
            logger.info(f"Rating columns: {list(self.df_rating.columns)}")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _separate_dataframes(self):
        """Separate the combined dataset into laptop and rating dataframes."""
        try:
            # Get unique laptops (products)
            laptop_columns = [
                'asin', 'title_y', 'brand', 'price', 'average_rating', 
                'features', 'images_y', 'videos', 'details', 'num_reviews',
                'os', 'color', 'store'
            ]
            
            # Filter columns that exist in the dataset
            available_laptop_columns = [col for col in laptop_columns if col in self.df_rating.columns]
            
            # Create laptop dataframe with unique products
            self.df_laptop = self.df_rating[available_laptop_columns].drop_duplicates(subset=['asin'])
            
            # Add laptop_id for easier reference
            self.df_laptop['laptop_id'] = range(len(self.df_laptop))
            
            # Create rating dataframe with all reviews
            rating_columns = [
                'asin', 'user_id', 'rating', 'title_x', 'text', 
                'helpful_vote', 'timestamp'
            ]
            
            # Filter columns that exist in the dataset
            available_rating_columns = [col for col in rating_columns if col in self.df_rating.columns]
            
            self.df_rating = self.df_rating[available_rating_columns]
            
            # Merge laptop_id into rating dataframe
            laptop_mapping = self.df_laptop[['asin', 'laptop_id']].set_index('asin')
            self.df_rating = self.df_rating.merge(
                laptop_mapping, left_on='asin', right_index=True, how='left'
            )
            
            logger.info(f"Data separated. Laptops: {self.df_laptop.shape}, Ratings: {self.df_rating.shape}")
            
        except Exception as e:
            logger.error(f"Error separating dataframes: {e}")
            raise
    
    def execute_sql_query(self, query: str, params: Optional[List] = None) -> pd.DataFrame:
        """
        Execute a SQL query on the rating dataset.
        
        Args:
            query: SQL query string
            params: Optional parameters for the query
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            # Create temporary SQLite database
            temp_db = ":memory:"
            conn = sqlite3.connect(temp_db)
            
            # Clean data before loading into SQLite
            # Convert any complex objects to strings
            df_rating_clean = self.df_rating.copy()
            df_laptop_clean = self.df_laptop.copy()
            
            # Convert any non-serializable columns to strings
            for col in df_rating_clean.columns:
                if df_rating_clean[col].dtype == 'object':
                    df_rating_clean[col] = df_rating_clean[col].astype(str)
            
            for col in df_laptop_clean.columns:
                if df_laptop_clean[col].dtype == 'object':
                    df_laptop_clean[col] = df_laptop_clean[col].astype(str)
            
            # Load data into SQLite
            df_rating_clean.to_sql('ratings', conn, index=False, if_exists='replace')
            df_laptop_clean.to_sql('laptops', conn, index=False, if_exists='replace')
            
            # Execute query
            if params:
                result = pd.read_sql_query(query, conn, params=params)
            else:
                result = pd.read_sql_query(query, conn)
            
            conn.close()
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            raise
    
    def search_users_with_filters(self, 
                                 search_term: Optional[str] = None,
                                 min_rating_count: Optional[int] = None,
                                 max_rating_count: Optional[int] = None,
                                 min_avg_rating: Optional[float] = None,
                                 max_avg_rating: Optional[float] = None,
                                 limit: int = 100) -> List[Dict]:
        """
        Search users with advanced filtering options.
        
        Args:
            search_term: Search term for user ID
            min_rating_count: Minimum number of ratings
            max_rating_count: Maximum number of ratings
            min_avg_rating: Minimum average rating given
            max_avg_rating: Maximum average rating given
            limit: Maximum number of results
            
        Returns:
            List[Dict]: List of user information dictionaries
        """
        try:
            # Build SQL query with proper parameter binding
            query_parts = [
                "SELECT",
                "    user_id,",
                "    COUNT(*) as total_ratings,",
                "    AVG(rating) as avg_rating,",
                "    MIN(timestamp) as first_rating,",
                "    MAX(timestamp) as last_rating",
                "FROM ratings",
                "WHERE 1=1"
            ]
            
            params = []
            
            # Add search term filter
            if search_term:
                query_parts.append("AND user_id LIKE ?")
                params.append(f"%{search_term}%")
            
            # Add rating count filters
            if min_rating_count is not None:
                query_parts.append("AND user_id IN (")
                query_parts.append("    SELECT user_id FROM ratings")
                query_parts.append("    GROUP BY user_id")
                query_parts.append("    HAVING COUNT(*) >= ?")
                query_parts.append(")")
                params.append(min_rating_count)
            
            if max_rating_count is not None:
                query_parts.append("AND user_id IN (")
                query_parts.append("    SELECT user_id FROM ratings")
                query_parts.append("    GROUP BY user_id")
                query_parts.append("    HAVING COUNT(*) <= ?")
                query_parts.append(")")
                params.append(max_rating_count)
            
            # Add average rating filters
            if min_avg_rating is not None:
                query_parts.append("AND user_id IN (")
                query_parts.append("    SELECT user_id FROM ratings")
                query_parts.append("    GROUP BY user_id")
                query_parts.append("    HAVING AVG(rating) >= ?")
                query_parts.append(")")
                params.append(min_avg_rating)
            
            if max_avg_rating is not None:
                query_parts.append("AND user_id IN (")
                query_parts.append("    SELECT user_id FROM ratings")
                query_parts.append("    GROUP BY user_id")
                query_parts.append("    HAVING AVG(rating) <= ?")
                query_parts.append(")")
                params.append(max_avg_rating)
            
            # Add grouping and ordering
            query_parts.extend([
                "GROUP BY user_id",
                "ORDER BY total_ratings DESC, avg_rating DESC",
                f"LIMIT {limit}"
            ])
            
            query = "\n".join(query_parts)
            
            # Execute query
            result_df = self.execute_sql_query(query, params)
            
            # Convert to list of dictionaries
            users = []
            for _, row in result_df.iterrows():
                user_info = {
                    'user_id_encoded': str(row['user_id']),
                    'username': f"User_{str(row['user_id'])[:8]}...",
                    'total_ratings': int(row['total_ratings']),
                    'avg_rating': float(row['avg_rating']) if pd.notna(row['avg_rating']) else 0.0,
                    'first_rating': str(row['first_rating']) if pd.notna(row['first_rating']) else None,
                    'last_rating': str(row['last_rating']) if pd.notna(row['last_rating']) else None
                }
                users.append(user_info)
            
            logger.info(f"Found {len(users)} users matching criteria")
            return users
            
        except Exception as e:
            logger.error(f"Error searching users with filters: {e}")
            return []
    
    def get_rating_count_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of rating counts across users.
        
        Returns:
            Dict[str, int]: Distribution of rating counts
        """
        try:
            query = """
                SELECT 
                    CASE 
                        WHEN rating_count = 1 THEN '1'
                        WHEN rating_count BETWEEN 2 AND 5 THEN '2-5'
                        WHEN rating_count BETWEEN 6 AND 10 THEN '6-10'
                        WHEN rating_count BETWEEN 11 AND 20 THEN '11-20'
                        WHEN rating_count BETWEEN 21 AND 50 THEN '21-50'
                        WHEN rating_count BETWEEN 51 AND 100 THEN '51-100'
                        ELSE '100+'
                    END as rating_range,
                    COUNT(*) as user_count
                FROM (
                    SELECT user_id, COUNT(*) as rating_count
                    FROM ratings
                    GROUP BY user_id
                ) user_counts
                GROUP BY rating_range
                ORDER BY 
                    CASE rating_range
                        WHEN '1' THEN 1
                        WHEN '2-5' THEN 2
                        WHEN '6-10' THEN 3
                        WHEN '11-20' THEN 4
                        WHEN '21-50' THEN 5
                        WHEN '51-100' THEN 6
                        WHEN '100+' THEN 7
                    END
            """
            
            result_df = self.execute_sql_query(query, [])
            
            distribution = {}
            for _, row in result_df.iterrows():
                distribution[row['rating_range']] = int(row['user_count'])
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error getting rating count distribution: {e}")
            return {}
    
    def get_user_detailed_stats(self, user_id_encoded: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific user.
        
        Args:
            user_id_encoded: The encoded user ID
            
        Returns:
            Dict[str, Any]: Detailed user statistics
        """
        try:
            # Get basic user stats
            basic_query = """
                SELECT 
                    COUNT(*) as total_ratings,
                    AVG(rating) as avg_rating,
                    MIN(rating) as min_rating,
                    MAX(rating) as max_rating,
                    MIN(timestamp) as first_rating,
                    MAX(timestamp) as last_rating
                FROM ratings
                WHERE user_id = ?
            """
            
            basic_result = self.execute_sql_query(basic_query, [user_id_encoded])
            
            if basic_result.empty:
                return {}
            
            basic_stats = basic_result.iloc[0]
            
            # Get rating distribution
            distribution_query = """
                SELECT rating, COUNT(*) as count
                FROM ratings
                WHERE user_id = ?
                GROUP BY rating
                ORDER BY rating
            """
            
            distribution_result = self.execute_sql_query(distribution_query, [user_id_encoded])
            rating_distribution = {str(row['rating']): int(row['count']) for _, row in distribution_result.iterrows()}
            
            # Get top rated laptops
            top_laptops_query = """
                SELECT r.asin, r.rating, l.title_y, l.brand, l.price
                FROM ratings r
                JOIN laptops l ON r.asin = l.asin
                WHERE r.user_id = ?
                ORDER BY r.rating DESC, r.timestamp DESC
                LIMIT 10
            """
            
            top_laptops_result = self.execute_sql_query(top_laptops_query, [user_id_encoded])
            top_laptops = []
            for _, row in top_laptops_result.iterrows():
                top_laptops.append({
                    'asin': row['asin'],
                    'rating': float(row['rating']),
                    'title': row['title_y'],
                    'brand': row['brand'],
                    'price': row['price']
                })
            
            # Count comments (ratings with non-empty text)
            comments_query = """
                SELECT COUNT(*) as total_comments
                FROM ratings
                WHERE user_id = ? AND text IS NOT NULL AND text != ''
            """
            
            comments_result = self.execute_sql_query(comments_query, [user_id_encoded])
            total_comments = int(comments_result.iloc[0]['total_comments']) if not comments_result.empty else 0
            
            return {
                'user_id_encoded': user_id_encoded,
                'total_ratings': int(basic_stats['total_ratings']),
                'total_comments': total_comments,
                'avg_rating': float(basic_stats['avg_rating']) if pd.notna(basic_stats['avg_rating']) else 0.0,
                'min_rating': float(basic_stats['min_rating']) if pd.notna(basic_stats['min_rating']) else 0.0,
                'max_rating': float(basic_stats['max_rating']) if pd.notna(basic_stats['max_rating']) else 0.0,
                'first_rating': str(basic_stats['first_rating']) if pd.notna(basic_stats['first_rating']) else None,
                'last_rating': str(basic_stats['last_rating']) if pd.notna(basic_stats['last_rating']) else None,
                'rating_distribution': rating_distribution,
                'top_rated_laptops': top_laptops
            }
            
        except Exception as e:
            logger.error(f"Error getting user detailed stats: {e}")
            return {}
    
    def get_user_ratings(self, user_id_encoded: str, limit: int = 50) -> List[Dict]:
        """
        Get user's rating history from Amazon dataset.
        
        Args:
            user_id_encoded: The encoded user ID
            limit: Maximum number of ratings to return
            
        Returns:
            List[Dict]: List of user ratings
        """
        try:
            query = """
                SELECT r.asin, r.rating, r.text as comment, r.timestamp, l.title_y as laptop_title, l.brand
                FROM ratings r
                JOIN laptops l ON r.asin = l.asin
                WHERE r.user_id = ?
                ORDER BY r.timestamp DESC
                LIMIT ?
            """
            
            result = self.execute_sql_query(query, [user_id_encoded, limit])
            
            ratings = []
            for _, row in result.iterrows():
                ratings.append({
                    'laptop_id': row['asin'],  # Using asin as laptop_id for now
                    'rating': float(row['rating']),
                    'comment': row['comment'] if pd.notna(row['comment']) else '',
                    'timestamp': str(row['timestamp']),
                    'laptop_title': row['laptop_title'],
                    'brand': row['brand']
                })
            
            return ratings
            
        except Exception as e:
            logger.error(f"Error getting user ratings: {e}")
            return []
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the dataset.
        
        Returns:
            Dict[str, Any]: Dataset summary statistics
        """
        try:
            summary_query = """
                SELECT 
                    COUNT(DISTINCT user_id) as total_users,
                    COUNT(DISTINCT asin) as total_laptops,
                    COUNT(*) as total_ratings,
                    AVG(rating) as avg_rating,
                    MIN(rating) as min_rating,
                    MAX(rating) as max_rating
                FROM ratings
            """
            
            result = self.execute_sql_query(summary_query, [])
            summary = result.iloc[0]
            
            return {
                'total_users': int(summary['total_users']),
                'total_laptops': int(summary['total_laptops']),
                'total_ratings': int(summary['total_ratings']),
                'avg_rating': float(summary['avg_rating']) if pd.notna(summary['avg_rating']) else 0.0,
                'min_rating': float(summary['min_rating']) if pd.notna(summary['min_rating']) else 0.0,
                'max_rating': float(summary['max_rating']) if pd.notna(summary['max_rating']) else 0.0,
                'rating_count_distribution': self.get_rating_count_distribution()
            }
            
        except Exception as e:
            logger.error(f"Error getting dataset summary: {e}")
            return {}


def create_hf_sql_client() -> HuggingFaceSQLClient:
    """Factory function to create a HuggingFaceSQLClient instance."""
    return HuggingFaceSQLClient()


if __name__ == "__main__":
    # Test the SQL client
    client = create_hf_sql_client()
    
    # Get dataset summary
    summary = client.get_dataset_summary()
    print("Dataset Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test user search
    users = client.search_users_with_filters(limit=5)
    print(f"\nFound {len(users)} users:")
    for user in users:
        print(f"  {user['username']}: {user['total_ratings']} ratings, avg: {user['avg_rating']:.2f}")
