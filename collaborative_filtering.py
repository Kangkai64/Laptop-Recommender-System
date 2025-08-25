"""
Collaborative Filtering Algorithm for Laptop Recommendation System

This module implements collaborative filtering approaches including user-based,
item-based, and matrix factorization methods for laptop recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class CollaborativeFiltering:
    """
    Collaborative Filtering algorithm for laptop recommendations.
    
    Implements user-based, item-based, and matrix factorization approaches.
    """
    
    def __init__(self, df_laptop: pd.DataFrame, df_rating: pd.DataFrame, 
                 config: Optional[Dict] = None):
        """Initialize the Collaborative Filtering system."""
        self.df_laptop = df_laptop.copy()
        self.df_rating = df_rating.copy()
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.user_factors = None
        self.item_factors = None
        
        # Default configuration
        self.config = {
            'matrix_factorization': {
                'n_components': 50,
                'random_state': 42,
                'max_iter': 200,
                'alpha': 0.1
            },
            'similarity_methods': {
                'min_common_items': 2,
                'min_common_users': 2,
                'similarity_threshold': 0.1
            },
            'recommendation_options': {
                'min_rating_threshold': 3.0,
                'max_recommendations': 10,
                'diversity_weight': 0.3
            }
        }
        
        if config:
            self._update_config(config)
        
        logger.info("CollaborativeFiltering initialized successfully")
    
    def _update_config(self, config: Dict) -> None:
        """Update configuration with custom parameters."""
        for section, params in config.items():
            if section in self.config:
                self.config[section].update(params)
            else:
                self.config[section] = params
    
    def create_user_item_matrix(self) -> pd.DataFrame:
        """Create user-item rating matrix from rating data."""
        logger.info("Creating user-item rating matrix...")
        
        try:
            # Ensure required columns exist
            required_cols = ['user_id_encoded', 'asin', 'rating']
            if not all(col in self.df_rating.columns for col in required_cols):
                if 'user_id' in self.df_rating.columns:
                    self.df_rating['user_id_encoded'] = self.df_rating['user_id']
                else:
                    raise ValueError("Required columns not found in rating data")
            
            # Create user-item matrix
            self.user_item_matrix = self.df_rating.pivot_table(
                index='user_id_encoded',
                columns='asin',
                values='rating',
                fill_value=0
            )
            
            # Remove users and items with too few ratings
            min_ratings = self.config['similarity_methods']['min_common_items']
            min_users = self.config['similarity_methods']['min_common_users']
            
            user_rating_counts = (self.user_item_matrix > 0).sum(axis=1)
            valid_users = user_rating_counts >= min_ratings
            self.user_item_matrix = self.user_item_matrix[valid_users]
            
            item_rating_counts = (self.user_item_matrix > 0).sum(axis=0)
            valid_items = item_rating_counts >= min_users
            self.user_item_matrix = self.user_item_matrix[valid_items]
            
            logger.info(f"User-item matrix created with shape: {self.user_item_matrix.shape}")
            return self.user_item_matrix
            
        except Exception as e:
            logger.error(f"Error creating user-item matrix: {str(e)}")
            raise
    
    def compute_user_similarity_matrix(self, method: str = 'cosine') -> np.ndarray:
        """Compute similarity matrix between users."""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        logger.info(f"Computing user similarity matrix using {method} method...")
        
        try:
            if method == 'cosine':
                self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
            elif method == 'pearson':
                self.user_similarity_matrix = self.user_item_matrix.T.corr().fillna(0).values
            else:
                raise ValueError(f"Unsupported similarity method: {method}")
            
            np.fill_diagonal(self.user_similarity_matrix, 0)
            logger.info(f"User similarity matrix computed with shape: {self.user_similarity_matrix.shape}")
            return self.user_similarity_matrix
            
        except Exception as e:
            logger.error(f"Error computing user similarity matrix: {str(e)}")
            raise
    
    def compute_item_similarity_matrix(self, method: str = 'cosine') -> np.ndarray:
        """Compute similarity matrix between items."""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        logger.info(f"Computing item similarity matrix using {method} method...")
        
        try:
            if method == 'cosine':
                self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            elif method == 'pearson':
                self.item_similarity_matrix = self.user_item_matrix.corr().fillna(0).values
            else:
                raise ValueError(f"Unsupported similarity method: {method}")
            
            np.fill_diagonal(self.item_similarity_matrix, 0)
            logger.info(f"Item similarity matrix computed with shape: {self.item_similarity_matrix.shape}")
            return self.item_similarity_matrix
            
        except Exception as e:
            logger.error(f"Error computing item similarity matrix: {str(e)}")
            raise
    
    def fit_matrix_factorization(self, method: str = 'nmf') -> Tuple[np.ndarray, np.ndarray]:
        """Fit matrix factorization model to decompose user-item matrix."""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        logger.info(f"Fitting matrix factorization using {method} method...")
        
        try:
            n_components = self.config['matrix_factorization']['n_components']
            
            if method == 'nmf':
                model = NMF(
                    n_components=n_components,
                    random_state=self.config['matrix_factorization']['random_state'],
                    max_iter=self.config['matrix_factorization']['max_iter'],
                    alpha=self.config['matrix_factorization']['alpha']
                )
                self.user_factors = model.fit_transform(self.user_item_matrix)
                self.item_factors = model.components_.T
                
            elif method == 'svd':
                model = TruncatedSVD(
                    n_components=n_components,
                    random_state=self.config['matrix_factorization']['random_state']
                )
                self.user_factors = model.fit_transform(self.user_item_matrix)
                self.item_factors = model.components_.T
                
            else:
                raise ValueError(f"Unsupported factorization method: {method}")
            
            logger.info(f"Matrix factorization completed. User factors: {self.user_factors.shape}, Item factors: {self.item_factors.shape}")
            return self.user_factors, self.item_factors
            
        except Exception as e:
            logger.error(f"Error fitting matrix factorization: {str(e)}")
            raise
    
    def get_user_based_recommendations(self, user_id: int, n_recommendations: int = 5,
                                     min_similarity: float = 0.1) -> List[Dict]:
        """Get recommendations based on similar users' preferences."""
        if self.user_similarity_matrix is None:
            self.compute_user_similarity_matrix()
        
        try:
            if user_id not in self.user_item_matrix.index:
                raise ValueError(f"User {user_id} not found in the system")
            
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            user_ratings = self.user_item_matrix.iloc[user_idx]
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            
            user_similarities = self.user_similarity_matrix[user_idx]
            similar_users = np.where(user_similarities >= min_similarity)[0]
            
            if len(similar_users) == 0:
                logger.warning(f"No similar users found for user {user_id}")
                return []
            
            recommendations = {}
            for sim_user_idx in similar_users:
                sim_user_id = self.user_item_matrix.index[sim_user_idx]
                sim_user_ratings = self.user_item_matrix.iloc[sim_user_idx]
                
                high_rated_items = sim_user_ratings[
                    (sim_user_ratings >= self.config['recommendation_options']['min_rating_threshold']) &
                    (~sim_user_ratings.index.isin(rated_items))
                ]
                
                for item_id, rating in high_rated_items.items():
                    if item_id not in recommendations:
                        recommendations[item_id] = {
                            'score': 0,
                            'ratings': [],
                            'similarities': []
                        }
                    
                    recommendations[item_id]['ratings'].append(rating)
                    recommendations[item_id]['similarities'].append(user_similarities[sim_user_idx])
            
            # Calculate final scores
            for item_id, item_data in recommendations.items():
                weighted_sum = sum(r * s for r, s in zip(item_data['ratings'], item_data['similarities']))
                similarity_sum = sum(item_data['similarities'])
                item_data['score'] = weighted_sum / similarity_sum if similarity_sum > 0 else 0
            
            # Sort by score and get top recommendations
            sorted_items = sorted(recommendations.items(), key=lambda x: x[1]['score'], reverse=True)
            top_items = sorted_items[:n_recommendations]
            
            # Format recommendations
            formatted_recommendations = []
            for item_id, item_data in top_items:
                laptop_data = self._get_laptop_details(item_id)
                if laptop_data:
                    formatted_recommendations.append({
                        'asin': item_id,
                        'title': laptop_data.get('title', 'Unknown'),
                        'brand': laptop_data.get('brand', 'Unknown'),
                        'price_myr': laptop_data.get('price_myr', 0),
                        'rating': laptop_data.get('average_rating', 0),
                        'recommendation_score': item_data['score'],
                        'method': 'user_based_cf',
                        'explanation': f"Recommended based on {len(item_data['ratings'])} similar users"
                    })
            
            logger.info(f"Generated {len(formatted_recommendations)} user-based recommendations for user {user_id}")
            return formatted_recommendations
            
        except Exception as e:
            logger.error(f"Error getting user-based recommendations: {str(e)}")
            raise
    
    def get_item_based_recommendations(self, user_id: int, n_recommendations: int = 5,
                                     min_similarity: float = 0.1) -> List[Dict]:
        """Get recommendations based on item similarities."""
        if self.item_similarity_matrix is None:
            self.compute_item_similarity_matrix()
        
        try:
            if user_id not in self.user_item_matrix.index:
                raise ValueError(f"User {user_id} not found in the system")
            
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            user_ratings = self.user_item_matrix.iloc[user_idx]
            rated_items = user_ratings[user_ratings > 0]
            
            if len(rated_items) == 0:
                logger.warning(f"User {user_id} has no ratings")
                return []
            
            recommendations = {}
            for rated_item_id, rating in rated_items.items():
                if rated_item_id not in self.user_item_matrix.columns:
                    continue
                
                item_idx = self.user_item_matrix.columns.get_loc(rated_item_id)
                item_similarities = self.item_similarity_matrix[item_idx]
                
                similar_items = np.where(item_similarities >= min_similarity)[0]
                
                for sim_item_idx in similar_items:
                    sim_item_id = self.user_item_matrix.columns[sim_item_idx]
                    
                    if sim_item_id in rated_items.index:
                        continue
                    
                    if sim_item_id not in recommendations:
                        recommendations[sim_item_id] = {
                            'score': 0,
                            'contributions': []
                        }
                    
                    similarity = item_similarities[sim_item_idx]
                    contribution = rating * similarity
                    recommendations[sim_item_id]['contributions'].append(contribution)
            
            # Calculate final scores
            for item_id, item_data in recommendations.items():
                item_data['score'] = np.mean(item_data['contributions'])
            
            # Sort by score and get top recommendations
            sorted_items = sorted(recommendations.items(), key=lambda x: x[1]['score'], reverse=True)
            top_items = sorted_items[:n_recommendations]
            
            # Format recommendations
            formatted_recommendations = []
            for item_id, item_data in top_items:
                laptop_data = self._get_laptop_details(item_id)
                if laptop_data:
                    formatted_recommendations.append({
                        'asin': item_id,
                        'title': laptop_data.get('title', 'Unknown'),
                        'brand': laptop_data.get('brand', 'Unknown'),
                        'price_myr': laptop_data.get('price_myr', 0),
                        'rating': laptop_data.get('average_rating', 0),
                        'recommendation_score': item_data['score'],
                        'method': 'item_based_cf',
                        'explanation': f"Recommended based on {len(item_data['contributions'])} similar items"
                    })
            
            logger.info(f"Generated {len(formatted_recommendations)} item-based recommendations for user {user_id}")
            return formatted_recommendations
            
        except Exception as e:
            logger.error(f"Error getting item-based recommendations: {str(e)}")
            raise
    
    def get_matrix_factorization_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Get recommendations using matrix factorization."""
        if self.user_factors is None or self.item_factors is None:
            self.fit_matrix_factorization()
        
        try:
            if user_id not in self.user_item_matrix.index:
                raise ValueError(f"User {user_id} not found in the system")
            
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            user_ratings = self.user_item_matrix.iloc[user_idx]
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            
            user_factors = self.user_factors[user_idx]
            predicted_ratings = np.dot(user_factors, self.item_factors.T)
            
            # Create item-score pairs, excluding already rated items
            item_scores = []
            for item_idx, item_id in enumerate(self.user_item_matrix.columns):
                if item_id not in rated_items:
                    item_scores.append((item_id, predicted_ratings[item_idx]))
            
            # Sort by predicted rating and get top recommendations
            item_scores.sort(key=lambda x: x[1], reverse=True)
            top_items = item_scores[:n_recommendations]
            
            # Format recommendations
            formatted_recommendations = []
            for item_id, predicted_rating in top_items:
                laptop_data = self._get_laptop_details(item_id)
                if laptop_data:
                    formatted_recommendations.append({
                        'asin': item_id,
                        'title': laptop_data.get('title', 'Unknown'),
                        'brand': laptop_data.get('brand', 'Unknown'),
                        'price_myr': laptop_data.get('price_myr', 0),
                        'rating': laptop_data.get('average_rating', 0),
                        'recommendation_score': predicted_rating,
                        'method': 'matrix_factorization',
                        'explanation': f"Predicted rating: {predicted_rating:.2f}"
                    })
            
            logger.info(f"Generated {len(formatted_recommendations)} matrix factorization recommendations for user {user_id}")
            return formatted_recommendations
            
        except Exception as e:
            logger.error(f"Error getting matrix factorization recommendations: {str(e)}")
            raise
    
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5,
                                 weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """Get hybrid recommendations combining multiple collaborative filtering methods."""
        if weights is None:
            weights = {
                'user_based': 0.4,
                'item_based': 0.3,
                'matrix_factorization': 0.3
            }
        
        try:
            # Get recommendations from all methods
            user_based_recs = self.get_user_based_recommendations(user_id, n_recommendations * 2)
            item_based_recs = self.get_item_based_recommendations(user_id, n_recommendations * 2)
            mf_recs = self.get_matrix_factorization_recommendations(user_id, n_recommendations * 2)
            
            # Combine recommendations
            combined_recs = {}
            
            # Process user-based recommendations
            for rec in user_based_recs:
                asin = rec['asin']
                if asin not in combined_recs:
                    combined_recs[asin] = {
                        'asin': asin,
                        'title': rec['title'],
                        'brand': rec['brand'],
                        'price_myr': rec['price_myr'],
                        'rating': rec['rating'],
                        'combined_score': 0,
                        'methods': [],
                        'explanations': []
                    }
                
                combined_recs[asin]['combined_score'] += weights['user_based'] * rec['recommendation_score']
                combined_recs[asin]['methods'].append('user_based')
                combined_recs[asin]['explanations'].append(rec['explanation'])
            
            # Process item-based recommendations
            for rec in item_based_recs:
                asin = rec['asin']
                if asin not in combined_recs:
                    combined_recs[asin] = {
                        'asin': asin,
                        'title': rec['title'],
                        'brand': rec['brand'],
                        'price_myr': rec['price_myr'],
                        'rating': rec['rating'],
                        'combined_score': 0,
                        'methods': [],
                        'explanations': []
                    }
                
                combined_recs[asin]['combined_score'] += weights['item_based'] * rec['recommendation_score']
                combined_recs[asin]['methods'].append('item_based')
                combined_recs[asin]['explanations'].append(rec['explanation'])
            
            # Process matrix factorization recommendations
            for rec in mf_recs:
                asin = rec['asin']
                if asin not in combined_recs:
                    combined_recs[asin] = {
                        'asin': asin,
                        'title': rec['title'],
                        'brand': rec['brand'],
                        'price_myr': rec['price_myr'],
                        'rating': rec['rating'],
                        'combined_score': 0,
                        'methods': [],
                        'explanations': []
                    }
                
                combined_recs[asin]['combined_score'] += weights['matrix_factorization'] * rec['recommendation_score']
                combined_recs[asin]['methods'].append('matrix_factorization')
                combined_recs[asin]['explanations'].append(rec['explanation'])
            
            # Sort by combined score and get top recommendations
            sorted_recs = sorted(combined_recs.values(), key=lambda x: x['combined_score'], reverse=True)
            top_recs = sorted_recs[:n_recommendations]
            
            # Format final recommendations
            formatted_recommendations = []
            for rec in top_recs:
                formatted_rec = {
                    'asin': rec['asin'],
                    'title': rec['title'],
                    'brand': rec['brand'],
                    'price_myr': rec['price_myr'],
                    'rating': rec['rating'],
                    'recommendation_score': rec['combined_score'],
                    'method': 'hybrid_cf',
                    'methods_used': rec['methods'],
                    'explanation': f"Combined from {len(rec['methods'])} methods: {', '.join(rec['methods'])}"
                }
                formatted_recommendations.append(formatted_rec)
            
            logger.info(f"Generated {len(formatted_recommendations)} hybrid recommendations for user {user_id}")
            return formatted_recommendations
            
        except Exception as e:
            logger.error(f"Error getting hybrid recommendations: {str(e)}")
            raise
    
    def _get_laptop_details(self, asin: str) -> Optional[Dict]:
        """Get laptop details from the laptop dataset."""
        try:
            laptop_mask = self.df_laptop['asin'] == asin
            if laptop_mask.any():
                laptop_data = self.df_laptop[laptop_mask].iloc[0]
                return {
                    'title': laptop_data.get('title_y_clean', 'Unknown'),
                    'brand': laptop_data.get('brand', 'Unknown'),
                    'price_myr': laptop_data.get('price_myr', 0),
                    'average_rating': laptop_data.get('average_rating', 0)
                }
            return None
        except Exception:
            return None
    
    def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Get user profile and preferences."""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        try:
            if user_id not in self.user_item_matrix.index:
                raise ValueError(f"User {user_id} not found in the system")
            
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            user_ratings = self.user_item_matrix.iloc[user_idx]
            rated_items = user_ratings[user_ratings > 0]
            
            profile = {
                'user_id': user_id,
                'total_ratings': len(rated_items),
                'average_rating': rated_items.mean() if len(rated_items) > 0 else 0,
                'rating_distribution': rated_items.value_counts().to_dict(),
                'preferred_brands': [],
                'preferred_price_range': None
            }
            
            # Get preferred brands
            if len(rated_items) > 0:
                brand_ratings = {}
                for item_id, rating in rated_items.items():
                    laptop_data = self._get_laptop_details(item_id)
                    if laptop_data and laptop_data['brand'] != 'Unknown':
                        brand = laptop_data['brand']
                        if brand not in brand_ratings:
                            brand_ratings[brand] = []
                        brand_ratings[brand].append(rating)
                
                # Calculate average rating per brand
                brand_avg_ratings = {brand: np.mean(ratings) for brand, ratings in brand_ratings.items()}
                profile['preferred_brands'] = sorted(brand_avg_ratings.items(), key=lambda x: x[1], reverse=True)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting user profile: {str(e)}")
            raise


def create_collaborative_filtering(df_laptop: pd.DataFrame, 
                                 df_rating: pd.DataFrame,
                                 config: Optional[Dict] = None) -> CollaborativeFiltering:
    """Factory function to create and configure CollaborativeFiltering instance."""
    return CollaborativeFiltering(df_laptop, df_rating, config)


if __name__ == "__main__":
    print("Collaborative Filtering Module")
    print("=" * 40)
    print("This module provides collaborative filtering for laptop recommendations.")
    print("Import and use the CollaborativeFiltering class in your code.")
