"""
Collaborative Filtering Algorithm for Laptop Recommendation System

This module implements collaborative filtering approaches including user-based,
item-based, and matrix factorization methods for laptop recommendations.
Enhanced to analyze comprehensive user behavior including view history, ratings,
activity patterns, and preferences.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
from user_behavior_analyzer import UserBehaviorAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class CollaborativeFiltering:
    """Collaborative Filtering algorithm for laptop recommendations."""
    
    def __init__(self, df_laptop: pd.DataFrame, df_rating: pd.DataFrame, 
                 config: Optional[Dict] = None, db_path: str = "data/user_data.db"):
        """Initialize the Collaborative Filtering system."""
        self.df_laptop = df_laptop.copy()
        self.df_rating = df_rating.copy()
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.user_factors = None
        self.item_factors = None
        
        # Initialize behavior analyzer for enhanced user profiling
        self.behavior_analyzer = UserBehaviorAnalyzer(db_path)
        self.enhanced_user_profiles = {}
        
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
                'max_recommendations': 50,
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
    
    def create_enhanced_user_item_matrix(self) -> pd.DataFrame:
        """Create enhanced user-item matrix including implicit feedback from behavior data."""
        logger.info("Creating enhanced user-item matrix with behavior data...")
        
        # Start with basic rating matrix
        self.create_user_item_matrix()
        
        if self.user_item_matrix is None or self.user_item_matrix.empty:
            return self.user_item_matrix
        
        try:
            # Get all users in the system
            all_users = set(self.user_item_matrix.index.tolist())
            
            # Add behavior-based implicit feedback
            enhanced_matrix = self.user_item_matrix.copy()
            
            for user_id in all_users:
                try:
                    # Get enhanced user profile
                    enhanced_profile = self.get_enhanced_user_profile(user_id)
                    
                    if not enhanced_profile:
                        continue
                    
                    # Add implicit feedback from view history
                    view_insights = enhanced_profile.get('view_insights', {})
                    if view_insights.get('views'):
                        for view in view_insights['views']:
                            laptop_id = view['laptop_id']
                            if laptop_id in enhanced_matrix.columns:
                                # Add implicit rating based on view behavior
                                implicit_rating = self._calculate_implicit_rating(view, enhanced_profile)
                                current_rating = enhanced_matrix.loc[user_id, laptop_id]
                                
                                # Only add implicit rating if no explicit rating exists
                                if current_rating == 0:
                                    enhanced_matrix.loc[user_id, laptop_id] = implicit_rating
                    
                    # Add implicit feedback from activity patterns
                    activity_insights = enhanced_profile.get('activity_insights', {})
                    if activity_insights.get('activity_counts'):
                        # Boost ratings for users with high engagement
                        engagement_boost = activity_insights.get('engagement_score', 0)
                        if engagement_boost > 0.5:
                            # Slightly boost existing ratings for engaged users
                            user_ratings = enhanced_matrix.loc[user_id]
                            non_zero_ratings = user_ratings[user_ratings > 0]
                            if len(non_zero_ratings) > 0:
                                boost_factor = 1 + (engagement_boost - 0.5) * 0.1  # Max 5% boost
                                enhanced_matrix.loc[user_id, non_zero_ratings.index] *= boost_factor
                
                except Exception as e:
                    logger.warning(f"Error processing user {user_id} for enhanced matrix: {e}")
                    continue
            
            self.user_item_matrix = enhanced_matrix
            logger.info(f"Enhanced user-item matrix created with {enhanced_matrix.shape[0]} users and {enhanced_matrix.shape[1]} items")
            return enhanced_matrix
            
        except Exception as e:
            logger.error(f"Error creating enhanced user-item matrix: {e}")
            return self.user_item_matrix
    
    def _calculate_implicit_rating(self, view_data: Dict, user_profile: Dict) -> float:
        """Calculate implicit rating from view behavior."""
        base_rating = 2.0  # Base implicit rating
        
        # Adjust based on view duration
        duration = view_data.get('duration', 0)
        if duration > 120:  # 2+ minutes
            base_rating += 1.0
        elif duration > 60:  # 1+ minute
            base_rating += 0.5
        
        # Adjust based on user's rating patterns
        rating_insights = user_profile.get('rating_insights', {})
        avg_rating = rating_insights.get('average_rating', 3.0)
        if avg_rating > 4.0:
            base_rating += 0.5
        elif avg_rating < 3.0:
            base_rating -= 0.5
        
        # Adjust based on user engagement
        activity_insights = user_profile.get('activity_insights', {})
        engagement = activity_insights.get('engagement_score', 0.5)
        base_rating += (engagement - 0.5) * 0.5
        
        # Ensure rating is within valid range
        return max(1.0, min(5.0, base_rating))
    
    def create_user_item_matrix(self) -> pd.DataFrame:
        """Create user-item rating matrix from rating data."""
        logger.info("Creating user-item rating matrix...")
        
        # Reset any existing matrix to ensure clean state
        self.user_item_matrix = None
        
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
            
            # Ensure we have a valid matrix
            if self.user_item_matrix.empty:
                logger.warning("User-item matrix is empty, creating minimal matrix")
                # Create a minimal matrix to prevent errors
                self.user_item_matrix = pd.DataFrame(index=[0], columns=['dummy_item'], data=[[0]])
                return self.user_item_matrix
            
            # Remove users and items with too few ratings
            min_ratings = self.config['similarity_methods']['min_common_items']
            min_users = self.config['similarity_methods']['min_common_users']
            
            # Filter users first
            user_rating_counts = (self.user_item_matrix > 0).sum(axis=1)
            valid_users = user_rating_counts >= min_ratings
            
            # Only filter if we have valid users
            if valid_users.any():
                # Use .loc to ensure proper index alignment
                self.user_item_matrix = self.user_item_matrix.loc[valid_users]
            else:
                logger.warning("No users meet minimum rating requirements, keeping all users")
            
            # Filter items second (after user filtering)
            item_rating_counts = (self.user_item_matrix > 0).sum(axis=0)
            valid_items = item_rating_counts >= min_users
            
            # Only filter if we have valid items
            if valid_items.any():
                # Use .loc to ensure proper index alignment
                self.user_item_matrix = self.user_item_matrix.loc[:, valid_items]
            else:
                logger.warning("No items meet minimum rating requirements, keeping all items")
            
            logger.info(f"User-item matrix created with shape: {self.user_item_matrix.shape}")
            return self.user_item_matrix
            
        except Exception as e:
            logger.error(f"Error creating user-item matrix: {str(e)}")
            # Create a minimal fallback matrix to prevent complete failure
            logger.warning("Creating minimal fallback matrix")
            self.user_item_matrix = pd.DataFrame(index=[0], columns=['dummy_item'], data=[[0]])
            return self.user_item_matrix
    
    def is_initialized(self) -> bool:
        """Check if the collaborative filtering system is properly initialized."""
        try:
            if self.user_item_matrix is None:
                return False
            return not self.user_item_matrix.empty and self.user_item_matrix.shape[0] > 0
        except Exception:
            return False
    
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
    
    def get_user_based_recommendations(self, user_id: str, n_recommendations: int = 5,
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
    
    def get_item_based_recommendations(self, user_id: str, n_recommendations: int = 5,
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
    
    def get_matrix_factorization_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Dict]:
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
    
    def get_hybrid_recommendations(self, user_id: str, n_recommendations: int = 5,
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
                # Get laptop_id from asin
                laptop_row = self.df_laptop[self.df_laptop['asin'] == rec['asin']]
                laptop_id = laptop_row['laptop_id'].iloc[0] if not laptop_row.empty else None
                
                formatted_rec = {
                    'laptop_id': laptop_id,
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
    
    def get_enhanced_recommendations(self, user_id: str, preferences: Optional[Dict] = None,
                                   n_recommendations: int = 10) -> List[Dict]:
        """Get enhanced recommendations using comprehensive user profile analysis."""
        try:
            # Get enhanced user profile
            enhanced_profile = self.get_enhanced_user_profile(user_id)
            
            if not enhanced_profile:
                logger.warning(f"No enhanced profile available for user {user_id}, falling back to basic recommendations")
                return self.get_popular_recommendations(preferences, n_recommendations)
            
            # Create enhanced user-item matrix
            if self.user_item_matrix is None:
                self.create_enhanced_user_item_matrix()
            
            # Get user's behavior insights
            rating_insights = enhanced_profile.get('rating_insights', {})
            view_insights = enhanced_profile.get('view_insights', {})
            brand_preferences = enhanced_profile.get('brand_preferences', {})
            price_preferences = enhanced_profile.get('price_preferences', {})
            feature_preferences = enhanced_profile.get('feature_preferences', {})
            
            # Calculate personalized scores for each item
            item_scores = {}
            
            for item_id in self.user_item_matrix.columns:
                score = 0.0
                laptop_data = self._get_laptop_details(item_id)
                
                if not laptop_data:
                    continue
                
                # Base popularity score
                popularity_score = self._calculate_popularity_score(item_id)
                score += popularity_score * 0.3
                
                # Brand preference score
                brand_score = self._calculate_brand_preference_score(
                    laptop_data, brand_preferences
                )
                score += brand_score * 0.2
                
                # Price preference score
                price_score = self._calculate_price_preference_score(
                    laptop_data, price_preferences
                )
                score += price_score * 0.2
                
                # Feature preference score
                feature_score = self._calculate_feature_preference_score(
                    laptop_data, feature_preferences
                )
                score += feature_score * 0.2
                
                # User engagement score
                engagement_score = self._calculate_engagement_score(
                    item_id, user_id, enhanced_profile
                )
                score += engagement_score * 0.1
                
                item_scores[item_id] = {
                    'total_score': score,
                    'popularity_score': popularity_score,
                    'brand_score': brand_score,
                    'price_score': price_score,
                    'feature_score': feature_score,
                    'engagement_score': engagement_score,
                    'laptop_data': laptop_data
                }
            
            # Sort by total score
            sorted_items = sorted(item_scores.items(), 
                                key=lambda x: x[1]['total_score'], reverse=True)
            
            # Apply additional filtering based on preferences
            if preferences:
                sorted_items = self._apply_preference_filtering(sorted_items, preferences)
            
            # Get top recommendations
            recommendations = []
            for item_id, scores in sorted_items[:n_recommendations]:
                laptop_data = scores['laptop_data'].copy()
                laptop_data['recommendation_score'] = scores['total_score']
                laptop_data['method'] = 'enhanced_collaborative'
                laptop_data['score_breakdown'] = {
                    'popularity': scores['popularity_score'],
                    'brand': scores['brand_score'],
                    'price': scores['price_score'],
                    'features': scores['feature_score'],
                    'engagement': scores['engagement_score']
                }
                recommendations.append(laptop_data)
            
            logger.info(f"Generated {len(recommendations)} enhanced recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting enhanced recommendations for user {user_id}: {str(e)}")
            return self.get_popular_recommendations(preferences, n_recommendations)
    
    def _calculate_popularity_score(self, item_id: str) -> float:
        """Calculate popularity score for an item."""
        if self.user_item_matrix is None or item_id not in self.user_item_matrix.columns:
            return 0.0
        
        ratings = self.user_item_matrix[item_id]
        non_zero_ratings = ratings[ratings > 0]
        
        if len(non_zero_ratings) == 0:
            return 0.0
        
        avg_rating = non_zero_ratings.mean()
        rating_count = len(non_zero_ratings)
        # Popularity score combines average rating and number of ratings
        return avg_rating * np.log(1 + rating_count)
    
    def _calculate_brand_preference_score(self, laptop_data: Dict, brand_preferences: Dict) -> float:
        """Calculate brand preference score."""
        laptop_brand = laptop_data.get('brand', 'Unknown')
        if laptop_brand == 'Unknown' or not brand_preferences.get('preferred_brands'):
            return 0.5  # Neutral score
        
        brand_scores = dict(brand_preferences.get('preferred_brands', []))
        return brand_scores.get(laptop_brand, 0.5)
    
    def _calculate_price_preference_score(self, laptop_data: Dict, price_preferences: Dict) -> float:
        """Calculate price preference score."""
        laptop_price = laptop_data.get('price_myr', 0)
        if laptop_price <= 0 or not price_preferences.get('preferred_price_range'):
            return 0.5  # Neutral score
        
        min_price, max_price = price_preferences['preferred_price_range']
        weighted_avg = price_preferences.get('weighted_average_price', (min_price + max_price) / 2)
        
        # Score based on how close the price is to user's preferred range
        if min_price <= laptop_price <= max_price:
            return 1.0
        else:
            # Calculate distance from preferred range
            if laptop_price < min_price:
                distance = (min_price - laptop_price) / min_price
            else:
                distance = (laptop_price - max_price) / max_price
            
            return max(0.0, 1.0 - distance)
    
    def _calculate_feature_preference_score(self, laptop_data: Dict, feature_preferences: Dict) -> float:
        """Calculate feature preference score."""
        if not feature_preferences:
            return 0.5  # Neutral score
        
        total_score = 0.0
        feature_count = 0
        
        for feature, preferences in feature_preferences.items():
            if feature in laptop_data:
                laptop_value = laptop_data[feature]
                preferred_value = preferences.get('preferred_value', 0)
                
                if preferred_value > 0:
                    # Calculate similarity score
                    if isinstance(laptop_value, (int, float)) and isinstance(preferred_value, (int, float)):
                        # For numeric features
                        if preferred_value > 0:
                            similarity = 1.0 - abs(laptop_value - preferred_value) / max(laptop_value, preferred_value)
                            total_score += max(0.0, similarity)
                            feature_count += 1
                    elif isinstance(laptop_value, str) and isinstance(preferred_value, str):
                        # For string features (exact match)
                        if laptop_value.lower() == preferred_value.lower():
                            total_score += 1.0
                            feature_count += 1
        
        return total_score / max(feature_count, 1) if feature_count > 0 else 0.5
    
    def _calculate_engagement_score(self, item_id: str, user_id: str, enhanced_profile: Dict) -> float:
        """Calculate engagement score based on user's interaction with similar items."""
        view_insights = enhanced_profile.get('view_insights', {})
        activity_insights = enhanced_profile.get('activity_insights', {})
        
        # Base engagement score
        engagement_score = activity_insights.get('engagement_score', 0.5)
        
        # Check if user has viewed this specific item
        views = view_insights.get('views', [])
        for view in views:
            if view['laptop_id'] == item_id:
                # Boost score if user has viewed this item
                duration = view.get('duration', 0)
                if duration > 60:  # 1+ minute view
                    engagement_score += 0.2
                else:
                    engagement_score += 0.1
                break
        
        return min(1.0, engagement_score)
    
    def _apply_preference_filtering(self, sorted_items: List, preferences: Dict) -> List:
        """Apply additional filtering based on user preferences."""
        filtered_items = []
        
        for item_id, scores in sorted_items:
            laptop_data = scores['laptop_data']
            
            # Budget filtering
            if 'budget_range' in preferences and preferences['budget_range']:
                budget_min, budget_max = preferences['budget_range']
                laptop_price = laptop_data.get('price_myr', 0)
                if not (budget_min <= laptop_price <= budget_max):
                    continue
            
            # Brand filtering
            if 'brand_preference' in preferences and preferences['brand_preference']:
                preferred_brand = preferences['brand_preference']
                laptop_brand = laptop_data.get('brand', '')
                if preferred_brand.lower() != laptop_brand.lower():
                    continue
            
            # RAM filtering
            if 'min_ram' in preferences:
                min_ram = preferences['min_ram']
                laptop_ram = laptop_data.get('ram_gb', 0)
                if laptop_ram < min_ram:
                    continue
            
            filtered_items.append((item_id, scores))
        
        return filtered_items

    def get_popular_recommendations(self, preferences: Dict = None, 
                                  n_recommendations: int = 10) -> List[Dict]:
        """Get popular recommendations based on overall user behavior patterns."""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        try:
            logger.info("Generating popular recommendations...")
            
            # Calculate popularity scores
            item_stats = {}
            for item_id in self.user_item_matrix.columns:
                item_ratings = self.user_item_matrix[item_id]
                non_zero_ratings = item_ratings[item_ratings > 0]
                
                if len(non_zero_ratings) > 0:
                    frequency_score = len(non_zero_ratings) / len(self.user_item_matrix)
                    avg_rating = non_zero_ratings.mean()
                    popularity_score = (0.6 * frequency_score) + (0.4 * avg_rating / 5.0)
                    
                    item_stats[item_id] = {
                        'popularity_score': popularity_score,
                        'frequency': len(non_zero_ratings),
                        'avg_rating': avg_rating,
                        'total_ratings': len(non_zero_ratings)
                    }
            
            return self._format_recommendations(item_stats, 'popularity_score', 
                                              preferences, n_recommendations, 'popular_collaborative',
                                              lambda stats: f"Popular choice: {stats['total_ratings']} ratings, avg {stats['avg_rating']:.1f} stars")
            
        except Exception as e:
            logger.error(f"Error getting popular recommendations: {str(e)}")
            raise
    
    def get_trending_recommendations(self, preferences: Dict = None,
                                   n_recommendations: int = 10) -> List[Dict]:
        """Get trending recommendations based on recent user behavior patterns."""
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        try:
            logger.info("Generating trending recommendations...")
            
            # Calculate trending scores
            trending_scores = {}
            for item_id in self.user_item_matrix.columns:
                item_ratings = self.user_item_matrix[item_id]
                non_zero_ratings = item_ratings[item_ratings > 0]
                
                if len(non_zero_ratings) >= 3:  # Minimum ratings for trending
                    high_ratings = non_zero_ratings[non_zero_ratings >= 4.0]
                    trending_score = len(high_ratings) / len(non_zero_ratings)
                    frequency_boost = min(len(non_zero_ratings) / 50, 1.0)
                    trending_score = trending_score * (1 + frequency_boost)
                    
                    trending_scores[item_id] = {
                        'trending_score': trending_score,
                        'high_rating_ratio': len(high_ratings) / len(non_zero_ratings),
                        'total_ratings': len(non_zero_ratings)
                    }
            
            return self._format_recommendations(trending_scores, 'trending_score',
                                              preferences, n_recommendations, 'trending_collaborative',
                                              lambda stats: f"Trending: {stats['high_rating_ratio']:.1%} high ratings")
            
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {str(e)}")
            raise
    
    def _format_recommendations(self, item_stats: Dict, score_key: str, 
                               preferences: Dict, n_recommendations: int, 
                               method: str, explanation_func) -> List[Dict]:
        """Helper function to format recommendations and reduce code duplication."""
        # Sort by score
        sorted_items = sorted(item_stats.items(), 
                            key=lambda x: x[1][score_key], reverse=True)
        
        # Apply preference filtering if provided
        if preferences:
            filtered_items = self._filter_by_preferences(sorted_items, preferences)
            if len(filtered_items) > 0:
                sorted_items = filtered_items
        
        # Get top recommendations
        top_items = sorted_items[:n_recommendations]
        
        # Format recommendations
        recommendations = []
        for item_id, stats in top_items:
            laptop_data = self._get_laptop_details(item_id)
            if laptop_data:
                recommendations.append({
                    'asin': item_id,
                    'title': laptop_data.get('title', 'Unknown'),
                    'brand': laptop_data.get('brand', 'Unknown'),
                    'price_myr': laptop_data.get('price_myr', 0),
                    'rating': laptop_data.get('average_rating', 0),
                    'recommendation_score': stats[score_key],
                    'method': method,
                    'explanation': explanation_func(stats)
                })
        
        logger.info(f"Generated {len(recommendations)} {method} recommendations")
        return recommendations

    def _filter_by_preferences(self, items: List, preferences: Dict) -> List:
        """Filter items based on user preferences."""
        if not preferences:
            return items
        
        filtered_items = []
        
        for item_id, stats in items:
            laptop_data = self._get_laptop_details(item_id)
            if not laptop_data:
                continue
            
            # Budget filtering
            if 'budget_range' in preferences and preferences['budget_range']:
                budget_min, budget_max = preferences['budget_range']
                price = laptop_data.get('price_myr', 0)
                if price < budget_min or price > budget_max:
                    continue
            
            # Brand preference (soft filtering - boost preferred brand but don't exclude others)
            brand_boost = 1.0
            if 'brand_preference' in preferences and preferences['brand_preference']:
                brand = laptop_data.get('brand', '')
                if preferences['brand_preference'].lower() in brand.lower():
                    brand_boost = 1.2  # 20% boost for preferred brand
            
            # RAM filtering
            if 'min_ram' in preferences:
                # This would need RAM data in laptop details
                pass
            
            # Apply brand boost to the stats
            if brand_boost != 1.0:
                # Create a copy of stats and apply the boost
                boosted_stats = stats.copy()
                for key in boosted_stats:
                    if isinstance(boosted_stats[key], (int, float)):
                        boosted_stats[key] *= brand_boost
                filtered_items.append((item_id, boosted_stats))
            else:
                filtered_items.append((item_id, stats))
        
        return filtered_items

    def get_enhanced_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user profile including behavior analysis."""
        if user_id in self.enhanced_user_profiles:
            return self.enhanced_user_profiles[user_id]
        
        try:
            # Create enhanced profile using behavior analyzer
            enhanced_profile = self.behavior_analyzer.create_enhanced_user_profile(
                user_id, self.df_laptop
            )
            
            # Cache the profile
            self.enhanced_user_profiles[user_id] = enhanced_profile
            
            logger.info(f"Created enhanced profile for user {user_id}")
            return enhanced_profile
            
        except Exception as e:
            logger.error(f"Error creating enhanced user profile for {user_id}: {e}")
            # Fallback to basic profile
            return self.get_user_profile(user_id)
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get basic user profile and preferences (fallback method)."""
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
