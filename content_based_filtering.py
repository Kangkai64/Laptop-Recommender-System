"""
Content-Based Filtering Algorithm for Laptop Recommendation System

This module implements a comprehensive content-based filtering approach that recommends
laptops based on the similarity of their features and specifications to user preferences.
The algorithm combines text, numerical, and categorical features to create unified
feature vectors and computes similarities using various metrics.

Author: Laptop Recommender System Team
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class ContentBasedFiltering:
    """
    Content-Based Filtering algorithm for laptop recommendations.
    
    This class implements a comprehensive content-based filtering approach that:
    1. Creates feature vectors from text, numerical, and categorical features
    2. Computes similarity matrices using various metrics
    3. Generates recommendations based on laptop similarity and user preferences
    4. Provides explainable recommendations with feature importance analysis
    
    Attributes:
        df_laptop (pd.DataFrame): Laptop dataset with features
        df_rating (pd.DataFrame): Rating dataset with user reviews
        feature_matrix (np.ndarray): Combined feature matrix
        similarity_matrix (np.ndarray): Pre-computed similarity matrix
        feature_names (List[str]): Names of features in the matrix
        tfidf_vectorizer (TfidfVectorizer): TF-IDF vectorizer for text features
        scaler (MinMaxScaler): Scaler for numerical features
        config (Dict): Configuration parameters
    """
    
    def __init__(self, df_laptop: pd.DataFrame, df_rating: pd.DataFrame, 
                 config: Optional[Dict] = None):
        """
        Initialize the Content-Based Filtering system.
        
        Args:
            df_laptop: DataFrame containing laptop features
            df_rating: DataFrame containing rating data
            config: Optional configuration dictionary
        """
        self.df_laptop = df_laptop.copy()
        self.df_rating = df_rating.copy()
        self.feature_matrix = None
        self.similarity_matrix = None
        self.feature_names = None
        self.tfidf_vectorizer = None
        self.scaler = None
        
        # Set default configuration
        self.config = {
            'tfidf_params': {
                'max_features': 1000,
                'stop_words': 'english',
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.95,
                'use_idf': True,
                'smooth_idf': True
            },
            'similarity_methods': {
                'text_weight': 0.6,
                'numerical_weight': 0.3,
                'categorical_weight': 0.1
            },
            'filtering_options': {
                'min_similarity_threshold': 0.1,
                'max_price_difference': 0.5,
                'brand_diversity': True,
                'price_range_coverage': True
            }
        }
        
        # Update with custom configuration if provided
        if config:
            self._update_config(config)
        
        logger.info("ContentBasedFiltering initialized successfully")
    
    def _update_config(self, config: Dict) -> None:
        """Update configuration with custom parameters."""
        for section, params in config.items():
            if section in self.config:
                self.config[section].update(params)
            else:
                self.config[section] = params
    
    def create_feature_matrix(self) -> np.ndarray:
        """
        Create comprehensive feature matrix combining all laptop features.
        
        This method:
        1. Processes text features using TF-IDF vectorization
        2. Normalizes numerical features using Min-Max scaling
        3. Encodes categorical features
        4. Combines all features into a unified matrix
        
        Returns:
            np.ndarray: Combined feature matrix
        """
        logger.info("Creating feature matrix...")
        
        try:
            # Get available text features for TF-IDF
            available_text_features = []
            text_feature_names = []
            
            # Check which text columns are available
            if 'title_y_clean' in self.df_laptop.columns:
                available_text_features.append(self.df_laptop['title_y_clean'].fillna(''))
                text_feature_names.append('title_y')
            
            if 'features_clean' in self.df_laptop.columns:
                available_text_features.append(self.df_laptop['features_clean'].fillna(''))
                text_feature_names.append('features')
            
            # Combine available text features
            if available_text_features:
                # Convert Series to strings and combine
                text_features = available_text_features[0].astype(str)
                for feature in available_text_features[1:]:
                    text_features = text_features + ' ' + feature.astype(str)
                
                # TF-IDF vectorization for text
                self.tfidf_vectorizer = TfidfVectorizer(**self.config['tfidf_params'])
                text_vectors = self.tfidf_vectorizer.fit_transform(text_features)
            else:
                # If no text features, create empty text vectors
                text_vectors = np.zeros((len(self.df_laptop), 1))
                self.tfidf_vectorizer = None
            
            # Get available numerical features
            numerical_features_list = []
            numerical_feature_names = []
            
            if 'price_myr' in self.df_laptop.columns:
                numerical_features_list.append(self.df_laptop['price_myr'].fillna(0))
                numerical_feature_names.append('price_myr')
            
            if 'average_rating' in self.df_laptop.columns:
                numerical_features_list.append(self.df_laptop['average_rating'].fillna(0))
                numerical_feature_names.append('average_rating')
            
            # Scale numerical features if available
            if numerical_features_list:
                numerical_features = pd.concat(numerical_features_list, axis=1)
                self.scaler = MinMaxScaler()
                numerical_scaled = self.scaler.fit_transform(numerical_features)
            else:
                numerical_scaled = np.zeros((len(self.df_laptop), 0))
                self.scaler = None
            
            # Get available categorical features (encoded)
            categorical_features_list = []
            categorical_feature_names = []
            
            for col in ['brand_encoded', 'os_encoded', 'color_encoded', 'store_encoded']:
                if col in self.df_laptop.columns:
                    categorical_features_list.append(self.df_laptop[col].fillna(0))
                    categorical_feature_names.append(col.replace('_encoded', ''))
            
            # Combine all features
            feature_arrays = []
            
            if text_vectors is not None and text_vectors.shape[1] > 0:
                feature_arrays.append(text_vectors.toarray())
            
            if numerical_scaled.shape[1] > 0:
                feature_arrays.append(numerical_scaled)
            
            if categorical_features_list:
                categorical_features = pd.concat(categorical_features_list, axis=1)
                feature_arrays.append(categorical_features.values)
            
            if feature_arrays:
                self.feature_matrix = np.hstack(feature_arrays)
            else:
                # Fallback: create a simple feature matrix with just basic info
                self.feature_matrix = np.zeros((len(self.df_laptop), 1))
            
            # Create feature names
            self.feature_names = []
            
            if text_vectors is not None and text_vectors.shape[1] > 0:
                self.feature_names.extend([f'text_{i}' for i in range(text_vectors.shape[1])])
            
            self.feature_names.extend(numerical_feature_names)
            self.feature_names.extend(categorical_feature_names)
            
            logger.info(f"Feature matrix created with shape: {self.feature_matrix.shape}")
            logger.info(f"Features used: {self.feature_names}")
            return self.feature_matrix
            
        except Exception as e:
            logger.error(f"Error creating feature matrix: {str(e)}")
            raise
    
    def compute_similarity_matrix(self, method: str = 'cosine') -> np.ndarray:
        """
        Compute similarity matrix between all laptops.
        
        Args:
            method: Similarity method ('cosine' or 'euclidean')
            
        Returns:
            np.ndarray: Similarity matrix
        """
        if self.feature_matrix is None:
            self.create_feature_matrix()
        
        logger.info(f"Computing similarity matrix using {method} method...")
        
        try:
            if method == 'cosine':
                self.similarity_matrix = cosine_similarity(self.feature_matrix)
            elif method == 'euclidean':
                distances = euclidean_distances(self.feature_matrix)
                # Convert distances to similarities (1 / (1 + distance))
                self.similarity_matrix = 1 / (1 + distances)
            else:
                raise ValueError(f"Unsupported similarity method: {method}")
            
            logger.info(f"Similarity matrix computed with shape: {self.similarity_matrix.shape}")
            return self.similarity_matrix
            
        except Exception as e:
            logger.error(f"Error computing similarity matrix: {str(e)}")
            raise
    
    def get_recommendations(self, laptop_id: str, n_recommendations: int = 5,
                          exclude_self: bool = True, min_similarity: float = 0.1) -> List[Dict]:
        """
        Get top-N similar laptops for a given laptop.
        
        Args:
            laptop_id: ASIN of the source laptop
            n_recommendations: Number of recommendations to return
            exclude_self: Whether to exclude the source laptop
            min_similarity: Minimum similarity threshold
            
        Returns:
            List[Dict]: List of recommended laptops with details
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        try:
            # Find laptop index
            laptop_mask = self.df_laptop['asin'] == laptop_id
            if not laptop_mask.any():
                raise ValueError(f"Laptop with ASIN {laptop_id} not found")
            
            laptop_idx = laptop_mask.idxmax()
            
            # Get similarity scores for this laptop
            similarities = self.similarity_matrix[laptop_idx]
            
            # Get top similar laptops
            if exclude_self:
                similarities[laptop_idx] = -1  # Exclude self
            
            # Filter by minimum similarity
            valid_indices = np.where(similarities >= min_similarity)[0]
            if len(valid_indices) == 0:
                logger.warning(f"No laptops found with similarity >= {min_similarity}")
                return []
            
            top_indices = np.argsort(similarities[valid_indices])[::-1][:n_recommendations]
            
            recommendations = []
            for idx in top_indices:
                original_idx = valid_indices[idx]
                laptop_data = self.df_laptop.iloc[original_idx]
                recommendations.append({
                    'asin': laptop_data['asin'],
                    'title': laptop_data['title_y_clean'],
                    'brand': laptop_data['brand_encoded'],
                    'price_myr': laptop_data['price_myr'],
                    'rating': laptop_data['average_rating'],
                    'similarity_score': similarities[original_idx],
                    'features': laptop_data['features_clean']
                })
            
            logger.info(f"Generated {len(recommendations)} recommendations for laptop {laptop_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise
    
    def get_recommendations_by_preferences(self, preferences: Dict, 
                                         n_recommendations: int = 5) -> List[Dict]:
        """
        Get recommendations based on user preferences.
        
        Args:
            preferences: Dictionary containing user preferences
            n_recommendations: Number of recommendations to return
            
        Returns:
            List[Dict]: List of recommended laptops
        """
        if self.feature_matrix is None:
            self.create_feature_matrix()
        
        try:
            # Create preference vector
            preference_vector = self._create_preference_vector(preferences)
            
            # Calculate similarity to preference vector
            similarities = cosine_similarity([preference_vector], self.feature_matrix)[0]
            
            # Get top recommendations
            top_indices = np.argsort(similarities)[::-1][:n_recommendations]
            
            recommendations = []
            for idx in top_indices:
                laptop_data = self.df_laptop.iloc[idx]
                recommendations.append({
                    'asin': laptop_data['asin'],
                    'title': laptop_data['title_y_clean'],
                    'brand': laptop_data['brand_encoded'],
                    'price_myr': laptop_data['price_myr'],
                    'rating': laptop_data['average_rating'],
                    'similarity_score': similarities[idx],
                    'features': laptop_data['features_clean']
                })
            
            logger.info(f"Generated {len(recommendations)} recommendations based on preferences")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations by preferences: {str(e)}")
            raise
    
    def _create_preference_vector(self, preferences: Dict) -> np.ndarray:
        """
        Create a feature vector based on user preferences.
        
        Args:
            preferences: Dictionary containing user preferences
            
        Returns:
            np.ndarray: Preference feature vector
        """
        # Initialize preference vector with zeros
        preference_vector = np.zeros(self.feature_matrix.shape[1])
        
        try:
            # Handle text preferences (search terms)
            if 'search_terms' in preferences and preferences['search_terms']:
                search_text = ' '.join(preferences['search_terms'])
                # Use TF-IDF to vectorize search terms
                search_tfidf = TfidfVectorizer(
                    max_features=self.config['tfidf_params']['max_features'],
                    stop_words='english'
                )
                search_vector = search_tfidf.fit_transform([search_text]).toarray()[0]
                
                # Map to text features in our matrix
                text_feature_count = self.config['tfidf_params']['max_features']
                preference_vector[:text_feature_count] = search_vector[:text_feature_count]
            
            # Handle numerical preferences
            if 'max_price' in preferences:
                # Normalize price preference
                max_price_normalized = min(preferences['max_price'] / 50000, 1.0)
                preference_vector[-6] = max_price_normalized  # Price feature index
            
            if 'min_rating' in preferences:
                # Normalize rating preference
                min_rating_normalized = preferences['min_rating'] / 5.0
                preference_vector[-5] = min_rating_normalized  # Rating feature index
            
            # Handle categorical preferences
            if 'brand_preference' in preferences:
                # Find brand encoding
                brand_mask = self.df_laptop['brand_encoded'] == preferences['brand_preference']
                if brand_mask.any():
                    preference_vector[-4] = 1.0  # Brand feature index
            
            return preference_vector
            
        except Exception as e:
            logger.error(f"Error creating preference vector: {str(e)}")
            raise
    
    def get_feature_importance(self, laptop_id: str) -> Dict[str, float]:
        """
        Get feature importance for a specific laptop.
        
        Args:
            laptop_id: ASIN of the laptop
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.feature_matrix is None:
            self.create_feature_matrix()
        
        try:
            laptop_mask = self.df_laptop['asin'] == laptop_id
            if not laptop_mask.any():
                raise ValueError(f"Laptop with ASIN {laptop_id} not found")
            
            laptop_idx = laptop_mask.idxmax()
            laptop_features = self.feature_matrix[laptop_idx]
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_importance[feature_name] = float(laptop_features[i])
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
            
            return dict(sorted_features[:20])  # Top 20 features
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise
    
    def explain_recommendation(self, source_laptop_id: str, 
                             target_laptop_id: str) -> Dict[str, Any]:
        """
        Explain why a laptop was recommended.
        
        Args:
            source_laptop_id: ASIN of the source laptop
            target_laptop_id: ASIN of the recommended laptop
            
        Returns:
            Dict: Explanation with feature similarities and overall similarity
        """
        if self.feature_matrix is None:
            self.create_feature_matrix()
        
        try:
            source_mask = self.df_laptop['asin'] == source_laptop_id
            target_mask = self.df_laptop['asin'] == target_laptop_id
            
            if not source_mask.any():
                raise ValueError(f"Source laptop with ASIN {source_laptop_id} not found")
            if not target_mask.any():
                raise ValueError(f"Target laptop with ASIN {target_laptop_id} not found")
            
            source_idx = source_mask.idxmax()
            target_idx = target_mask.idxmax()
            
            source_features = self.feature_matrix[source_idx]
            target_features = self.feature_matrix[target_idx]
            
            # Calculate feature-wise similarity
            feature_similarities = []
            for i, feature_name in enumerate(self.feature_names):
                similarity = 1 - abs(source_features[i] - target_features[i])
                feature_similarities.append((feature_name, similarity))
            
            # Sort by similarity
            feature_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Generate explanation
            top_similar_features = [f[0] for f in feature_similarities[:5]]
            explanation = (f"Laptop {target_laptop_id} is recommended because it shares "
                         f"similar characteristics in: {', '.join(top_similar_features)}")
            
            return {
                'explanation': explanation,
                'feature_similarities': dict(feature_similarities[:10]),
                'overall_similarity': self.similarity_matrix[source_idx][target_idx]
            }
            
        except Exception as e:
            logger.error(f"Error explaining recommendation: {str(e)}")
            raise
    
    def get_diverse_recommendations(self, laptop_id: str, n_recommendations: int = 5,
                                  diversity_weight: float = 0.3) -> List[Dict]:
        """
        Get diverse recommendations by considering feature diversity.
        
        Args:
            laptop_id: ASIN of the source laptop
            n_recommendations: Number of recommendations to return
            diversity_weight: Weight for diversity vs similarity
            
        Returns:
            List[Dict]: Diverse list of recommended laptops
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        try:
            # Get initial recommendations
            initial_recs = self.get_recommendations(
                laptop_id, n_recommendations * 2, exclude_self=True
            )
            
            if len(initial_recs) <= n_recommendations:
                return initial_recs[:n_recommendations]
            
            # Calculate diversity scores
            diverse_recs = [initial_recs[0]]  # Start with most similar
            
            for _ in range(n_recommendations - 1):
                best_score = -1
                best_rec = None
                
                for rec in initial_recs:
                    if rec in diverse_recs:
                        continue
                    
                    # Calculate diversity score
                    diversity_score = self._calculate_diversity_score(rec, diverse_recs)
                    similarity_score = rec['similarity_score']
                    
                    # Combined score
                    combined_score = (1 - diversity_weight) * similarity_score + diversity_weight * diversity_score
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_rec = rec
                
                if best_rec:
                    diverse_recs.append(best_rec)
            
            return diverse_recs
            
        except Exception as e:
            logger.error(f"Error getting diverse recommendations: {str(e)}")
            raise
    
    def _calculate_diversity_score(self, candidate: Dict, selected: List[Dict]) -> float:
        """Calculate diversity score for a candidate recommendation."""
        if not selected:
            return 1.0
        
        # Calculate average similarity to already selected items
        similarities = []
        for selected_rec in selected:
            # Find similarity between candidate and selected
            candidate_idx = self.df_laptop[self.df_laptop['asin'] == candidate['asin']].index[0]
            selected_idx = self.df_laptop[self.df_laptop['asin'] == selected_rec['asin']].index[0]
            similarity = self.similarity_matrix[candidate_idx][selected_idx]
            similarities.append(similarity)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and parameters."""
        import pickle
        
        try:
            model_data = {
                'feature_matrix': self.feature_matrix,
                'similarity_matrix': self.similarity_matrix,
                'feature_names': self.feature_names,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'scaler': self.scaler,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a previously saved model."""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.feature_matrix = model_data['feature_matrix']
            self.similarity_matrix = model_data['similarity_matrix']
            self.feature_names = model_data['feature_names']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.scaler = model_data['scaler']
            self.config = model_data['config']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


def create_content_based_filtering(df_laptop: pd.DataFrame, 
                                 df_rating: pd.DataFrame,
                                 config: Optional[Dict] = None) -> ContentBasedFiltering:
    """
    Factory function to create and configure ContentBasedFiltering instance.
    
    Args:
        df_laptop: Laptop dataset
        df_rating: Rating dataset
        config: Optional configuration
        
    Returns:
        ContentBasedFiltering: Configured instance
    """
    return ContentBasedFiltering(df_laptop, df_rating, config)


if __name__ == "__main__":
    # Example usage
    print("Content-Based Filtering Module")
    print("=" * 40)
    print("This module provides content-based filtering for laptop recommendations.")
    print("Import and use the ContentBasedFiltering class in your code.")
