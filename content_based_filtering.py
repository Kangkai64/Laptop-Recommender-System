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
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class ContentBasedFiltering:
    """Content-Based Filtering algorithm for laptop recommendations."""
    
    def __init__(self, df_laptop: pd.DataFrame, df_rating: pd.DataFrame, 
                 config: Optional[Dict] = None):
        """Initialize the Content-Based Filtering system."""
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
            },
            'similarity_improvements': {
                'enable_price_penalty': True,
                'enable_diversity_bonus': True,
                'log_scaling_power': 1.2,
                'similarity_range_min': 0.2,
                'similarity_range_max': 0.9
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
        """Create comprehensive feature matrix combining all laptop features."""
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
                
                # TF-IDF vectorization for text with more restrictive parameters
                tfidf_params = self.config['tfidf_params'].copy()
                tfidf_params.update({
                    'max_features': 500,  # Reduce features to avoid overfitting
                    'min_df': 3,  # Increase minimum document frequency
                    'max_df': 0.8,  # Reduce maximum document frequency
                    'ngram_range': (1, 1)  # Use only unigrams for better diversity
                })
                
                self.tfidf_vectorizer = TfidfVectorizer(**tfidf_params)
                text_vectors = self.tfidf_vectorizer.fit_transform(text_features)
            else:
                # If no text features, create empty text vectors
                text_vectors = np.zeros((len(self.df_laptop), 1))
                self.tfidf_vectorizer = None
            
            # Get available numerical features (technical specifications only)
            numerical_features_list = []
            numerical_feature_names = []
            
            # Technical specifications for similarity calculation
            spec_columns = ['ram_gb', 'storage_gb', 'screen_size_inches', 'cpu_benchmark_score', 'gpu_benchmark_score', 'total_benchmark_score']
            for col in spec_columns:
                if col in self.df_laptop.columns:
                    # Apply logarithmic scaling for better differentiation
                    if col in ['ram_gb', 'storage_gb', 'cpu_benchmark_score', 'gpu_benchmark_score', 'total_benchmark_score']:
                        # Use log scaling for these features to reduce dominance of high values
                        values = self.df_laptop[col].fillna(0)
                        # Add 1 to avoid log(0) and use log10 for better scaling
                        log_values = np.log10(values + 1)
                        numerical_features_list.append(log_values)
                        numerical_feature_names.append(f'{col}_log')
                    else:
                        numerical_features_list.append(self.df_laptop[col].fillna(0))
                        numerical_feature_names.append(col)
            
            # Add price as a differentiating feature (but with lower weight)
            if 'price_myr' in self.df_laptop.columns:
                # Use log scaling for price to reduce extreme value dominance
                price_values = self.df_laptop['price_myr'].fillna(0)
                log_price = np.log10(price_values + 1)
                numerical_features_list.append(log_price)
                numerical_feature_names.append('price_log')
            
            # Add price categories for better differentiation
            if 'price_myr' in self.df_laptop.columns:
                price_data = self.df_laptop['price_myr'].fillna(0)
                price_categories = pd.cut(price_data, bins=5, labels=False, include_lowest=True)
                numerical_features_list.append(price_categories.fillna(0))
                numerical_feature_names.append('price_category')
            
            if 'average_rating' in self.df_laptop.columns:
                rating_data = self.df_laptop['average_rating'].fillna(0)
                numerical_features_list.append(rating_data)
                numerical_feature_names.append('average_rating')
                
                # Add rating categories
                rating_categories = pd.cut(rating_data, bins=5, labels=False, include_lowest=True)
                numerical_features_list.append(rating_categories.fillna(0))
                numerical_feature_names.append('rating_category')
            
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
            
            # Basic categorical features
            basic_categorical = ['brand_encoded', 'os_encoded', 'color_encoded', 'store_encoded']
            for col in basic_categorical:
                # Handle brand encoding specially - create one-hot encoded features
                if 'brand_encoded' in self.df_laptop.columns:
                    brand_encoded = self.df_laptop['brand_encoded'].fillna(0)
                    unique_brands = brand_encoded.unique()
                    
                    # Create one-hot encoded features for each brand
                    for brand_val in unique_brands:
                        if brand_val != 0:  # Skip unknown/empty brands
                            brand_feature = (brand_encoded == brand_val).astype(int)
                            categorical_features_list.append(brand_feature)
                            categorical_feature_names.append(f'brand_{int(brand_val)}')
                    
                    logger.info(f"Created {len(unique_brands)-1} brand features")
            
            # Handle other categorical features normally
            for col in ['os_encoded', 'color_encoded', 'store_encoded']:
                if col in self.df_laptop.columns:
                    categorical_features_list.append(self.df_laptop[col].fillna(0))
                    categorical_feature_names.append(col.replace('_encoded', ''))
            
            # Technical specification categorical features
            spec_categorical = ['storage_type', 'ram_type', 'processor_model', 'gpu_model']
            for col in spec_categorical:
                if col in self.df_laptop.columns:
                    # Encode these categorical features if not already encoded
                    if f'{col}_encoded' not in self.df_laptop.columns:
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        self.df_laptop[f'{col}_encoded'] = le.fit_transform(self.df_laptop[col].fillna('Unknown'))
                    
                    categorical_features_list.append(self.df_laptop[f'{col}_encoded'].fillna(0))
                    categorical_feature_names.append(col)
            
            # Add performance tier and gaming capability as important differentiating features
            performance_features = ['performance_tier', 'gaming_capability']
            for col in performance_features:
                if col in self.df_laptop.columns:
                    # Encode these categorical features if not already encoded
                    if f'{col}_encoded' not in self.df_laptop.columns:
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        self.df_laptop[f'{col}_encoded'] = le.fit_transform(self.df_laptop[col].fillna('Unknown'))
                    
                    categorical_features_list.append(self.df_laptop[f'{col}_encoded'].fillna(0))
                    categorical_feature_names.append(col)
            
            # Combine all features
            feature_arrays = []
            
            if text_vectors is not None and text_vectors.shape[1] > 0:
                # Handle both sparse matrices and numpy arrays
                if hasattr(text_vectors, 'toarray'):
                    feature_arrays.append(text_vectors.toarray())
                else:
                    feature_arrays.append(text_vectors)
            
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
        Compute similarity matrix between all laptops with improved scoring.
        
        Args:
            method: Similarity method ('cosine' or 'euclidean')
            
        Returns:
            np.ndarray: Similarity matrix with more realistic scores
        """
        if self.feature_matrix is None:
            self.create_feature_matrix()
        
        logger.info(f"Computing similarity matrix using {method} method...")
        
        try:
            if method == 'cosine':
                # Compute base cosine similarity
                base_similarity = cosine_similarity(self.feature_matrix)
                
                # Apply feature weighting and scaling for more realistic scores
                self.similarity_matrix = self._apply_similarity_improvements(base_similarity)
                
                # Use cosine similarity but apply additional normalization
                raw_similarities = cosine_similarity(self.feature_matrix)
                
                # Apply normalization to make similarities more realistic
                # Remove perfect self-similarity (diagonal) and normalize
                np.fill_diagonal(raw_similarities, 0)
                
                # This prevents all laptops from having 100% similarity
                min_sim = np.min(raw_similarities[raw_similarities > 0])
                max_sim = np.max(raw_similarities)
                
                if max_sim > min_sim:
                    # Normalize to 0.6-0.95 range
                    normalized_similarities = 0.6 + 0.35 * (raw_similarities - min_sim) / (max_sim - min_sim)
                else:
                    normalized_similarities = np.full_like(raw_similarities, 0.7)
                
                # Restore diagonal (self-similarity)
                np.fill_diagonal(normalized_similarities, 1.0)
                
                self.similarity_matrix = normalized_similarities
                
            elif method == 'euclidean':
                distances = euclidean_distances(self.feature_matrix)
                # Convert distances to similarities (1 / (1 + distance))
                base_similarity = 1 / (1 + distances)
                self.similarity_matrix = self._apply_similarity_improvements(base_similarity)
                # Convert distances to similarities with better scaling
                max_distance = np.max(distances)
                self.similarity_matrix = 1 / (1 + distances / max_distance)
            else:
                raise ValueError(f"Unsupported similarity method: {method}")
            
            logger.info(f"Similarity matrix computed with shape: {self.similarity_matrix.shape}")
            logger.info(f"Similarity score range: {self.similarity_matrix.min():.3f} - {self.similarity_matrix.max():.3f}")
            logger.info(f"Similarity range: {np.min(self.similarity_matrix):.3f} - {np.max(self.similarity_matrix):.3f}")
            return self.similarity_matrix
            
        except Exception as e:
            logger.error(f"Error computing similarity matrix: {str(e)}")
            raise
    
    def compute_specification_similarity_matrix(self) -> np.ndarray:
        """
        Compute similarity matrix focused on technical specifications and benchmarks.
        This method gives higher weight to performance-related features.
        
        Returns:
            np.ndarray: Specification-focused similarity matrix
        """
        logger.info("Computing specification-focused similarity matrix...")
        
        try:
            # Create a specification-focused feature matrix
            spec_features_list = []
            spec_feature_names = []
            
            # High-priority specification features with weights
            spec_weights = {
                'cpu_benchmark_score': 0.25,
                'gpu_benchmark_score': 0.25,
                'total_benchmark_score': 0.20,
                'ram_gb': 0.15,
                'storage_gb': 0.10,
                'screen_size_inches': 0.05
            }
            
            # Add weighted specification features
            for col, weight in spec_weights.items():
                if col in self.df_laptop.columns:
                    values = self.df_laptop[col].fillna(0)
                    # Apply log scaling for better differentiation
                    if col in ['ram_gb', 'storage_gb', 'cpu_benchmark_score', 'gpu_benchmark_score', 'total_benchmark_score']:
                        log_values = np.log10(values + 1)
                        weighted_values = log_values * weight
                    else:
                        weighted_values = values * weight
                    
                    spec_features_list.append(weighted_values)
                    spec_feature_names.append(f'{col}_weighted')
            
            # Add processor and GPU model similarity (categorical)
            if 'processor_model' in self.df_laptop.columns:
                processor_encoded = pd.get_dummies(self.df_laptop['processor_model'].fillna('Unknown'))
                spec_features_list.append(processor_encoded.values * 0.15)  # 15% weight for processor model
                spec_feature_names.extend([f'processor_{col}' for col in processor_encoded.columns])
            
            if 'gpu_model' in self.df_laptop.columns:
                gpu_encoded = pd.get_dummies(self.df_laptop['gpu_model'].fillna('Unknown'))
                spec_features_list.append(gpu_encoded.values * 0.15)  # 15% weight for GPU model
                spec_feature_names.extend([f'gpu_{col}' for col in gpu_encoded.columns])
            
            # Combine all specification features
            if spec_features_list:
                spec_matrix = np.column_stack(spec_features_list)
                
                # Normalize the specification matrix
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                spec_matrix_normalized = scaler.fit_transform(spec_matrix)
                
                # Compute cosine similarity for specifications
                spec_similarity = cosine_similarity(spec_matrix_normalized)
                
                # Apply additional weighting for benchmark scores
                if 'cpu_benchmark_score' in self.df_laptop.columns and 'gpu_benchmark_score' in self.df_laptop.columns:
                    # Boost similarity for laptops with similar benchmark performance
                    cpu_scores = self.df_laptop['cpu_benchmark_score'].fillna(0).values
                    gpu_scores = self.df_laptop['gpu_benchmark_score'].fillna(0).values
                    
                    # Create benchmark similarity boost
                    cpu_similarity = 1 - np.abs(cpu_scores[:, np.newaxis] - cpu_scores[np.newaxis, :]) / (cpu_scores.max() + 1)
                    gpu_similarity = 1 - np.abs(gpu_scores[:, np.newaxis] - gpu_scores[np.newaxis, :]) / (gpu_scores.max() + 1)
                    
                    # Combine with base similarity (85% base, 7.5% CPU, 7.5% GPU) - reduced boost
                    spec_similarity = 0.85 * spec_similarity + 0.075 * cpu_similarity + 0.075 * gpu_similarity
                
                # Apply similarity improvements (including price penalty)
                spec_similarity = self._apply_similarity_improvements(spec_similarity)
                
                logger.info(f"Specification similarity matrix computed with shape: {spec_similarity.shape}")
                logger.info(f"Specification similarity range: {spec_similarity.min():.3f} - {spec_similarity.max():.3f}")
                
                return spec_similarity
            else:
                logger.warning("No specification features available, falling back to standard similarity")
                return self.compute_similarity_matrix()
                
        except Exception as e:
            logger.error(f"Error computing specification similarity matrix: {str(e)}")
            # Fall back to standard similarity
            return self.compute_similarity_matrix()
    
    def _apply_similarity_improvements(self, base_similarity: np.ndarray) -> np.ndarray:
        """
        Apply improvements to similarity scores for more realistic distribution.
        
        Args:
            base_similarity: Base similarity matrix
            
        Returns:
            np.ndarray: Improved similarity matrix with more realistic scores
        """
        try:
            # Create a copy to avoid modifying the original
            improved_similarity = base_similarity.copy()
            
            # 1. Apply logarithmic scaling to reduce high similarity scores
            # This helps differentiate between very similar items
            log_power = self.config['similarity_improvements']['log_scaling_power']
            improved_similarity = np.power(improved_similarity, log_power)
            
            # 2. Add price-based penalty for better discrimination
            if (self.config['similarity_improvements']['enable_price_penalty'] and 
                'price_myr' in self.df_laptop.columns):
                price_penalty = self._compute_price_penalty()
                improved_similarity = improved_similarity * price_penalty
            
            # 3. Apply feature diversity bonus
            if self.config['similarity_improvements']['enable_diversity_bonus']:
                diversity_bonus = self._compute_diversity_bonus()
                improved_similarity = improved_similarity * diversity_bonus
            
            # 4. Apply gentle scaling to ensure reasonable range
            min_score = improved_similarity.min()
            max_score = improved_similarity.max()
            
            # Only scale if the range is too narrow or too wide
            if max_score - min_score < 0.1:  # Too narrow
                improved_similarity = 0.3 + (improved_similarity - min_score) / (max_score - min_score + 1e-8) * 0.6
            elif max_score - min_score > 0.8:  # Too wide
                improved_similarity = 0.2 + (improved_similarity - min_score) / (max_score - min_score) * 0.7
            
            # Handle any NaN values and ensure diagonal elements are 1.0
            improved_similarity = np.nan_to_num(improved_similarity, nan=0.5, posinf=1.0, neginf=0.0)
            np.fill_diagonal(improved_similarity, 1.0)
            
            logger.info(f"Applied similarity improvements - new range: {improved_similarity.min():.3f} - {improved_similarity.max():.3f}")
            return improved_similarity
            
        except Exception as e:
            logger.error(f"Error applying similarity improvements: {str(e)}")
            # Return original similarity if improvements fail
            return base_similarity
    
    def _compute_price_penalty(self) -> np.ndarray:
        """
        Compute price-based penalty matrix to reduce similarity for laptops with large price differences.
        
        Returns:
            np.ndarray: Price penalty matrix
        """
        try:
            prices = self.df_laptop['price_myr'].fillna(0).values
            
            # Create price difference matrix
            price_diff = np.abs(prices[:, np.newaxis] - prices[np.newaxis, :])
            
            # Normalize price differences (assuming max price difference of 10000)
            max_price_diff = 10000
            normalized_diff = np.minimum(price_diff / max_price_diff, 1.0)
            
            # Convert to penalty (higher price difference = lower similarity)
            # Penalty ranges from 1.0 (same price) to 0.5 (very different prices)
            penalty = 1.0 - (normalized_diff * 0.5)
            
            # Ensure no NaN or invalid values
            penalty = np.nan_to_num(penalty, nan=1.0, posinf=1.0, neginf=0.5)
            
            return penalty
            
        except Exception as e:
            logger.warning(f"Error computing price penalty: {str(e)}")
            # Return neutral penalty if computation fails
            return np.ones((len(self.df_laptop), len(self.df_laptop)))
    
    def _compute_diversity_bonus(self) -> np.ndarray:
        """
        Compute diversity bonus to reward laptops with different feature combinations.
        
        Returns:
            np.ndarray: Diversity bonus matrix
        """
        try:
            # Get key differentiating features
            diversity_features = []
            
            # Add brand diversity
            if 'brand_encoded' in self.df_laptop.columns:
                brand_diff = (self.df_laptop['brand_encoded'].values[:, np.newaxis] != 
                             self.df_laptop['brand_encoded'].values[np.newaxis, :])
                diversity_features.append(brand_diff.astype(float))
            
            # Add processor diversity
            if 'processor_model' in self.df_laptop.columns:
                processor_diff = (self.df_laptop['processor_model'].values[:, np.newaxis] != 
                                 self.df_laptop['processor_model'].values[np.newaxis, :])
                diversity_features.append(processor_diff.astype(float))
            
            # Add GPU diversity
            if 'gpu_model' in self.df_laptop.columns:
                gpu_diff = (self.df_laptop['gpu_model'].values[:, np.newaxis] != 
                           self.df_laptop['gpu_model'].values[np.newaxis, :])
                diversity_features.append(gpu_diff.astype(float))
            
            if diversity_features:
                # Combine diversity features
                total_diversity = np.sum(diversity_features, axis=0)
                max_diversity = len(diversity_features)
                
                # Normalize and convert to bonus (0.9 to 1.1 range)
                normalized_diversity = total_diversity / max_diversity
                bonus = 0.9 + (normalized_diversity * 0.2)
                
                return bonus
            else:
                # Return neutral bonus if no diversity features available
                return np.ones((len(self.df_laptop), len(self.df_laptop)))
                
        except Exception as e:
            logger.warning(f"Error computing diversity bonus: {str(e)}")
            # Return neutral bonus if computation fails
            return np.ones((len(self.df_laptop), len(self.df_laptop)))
    
    def get_recommendations(self, laptop_id: int, n_recommendations: int = 5,
                          exclude_self: bool = True, min_similarity: float = 0.1,
                          use_spec_similarity: bool = True) -> List[Dict]:
        """
        Get top-N similar laptops for a given laptop.
        
        Args:
            laptop_id: laptop_id of the source laptop
            n_recommendations: Number of recommendations to return
            exclude_self: Whether to exclude the source laptop
            min_similarity: Minimum similarity threshold
            use_spec_similarity: Whether to use specification-focused similarity
            
        Returns:
            List[Dict]: List of recommended laptops with details
        """
        # Use specification-focused similarity if requested and available
        if use_spec_similarity:
            try:
                spec_similarity_matrix = self.compute_specification_similarity_matrix()
                similarity_matrix = spec_similarity_matrix
                logger.info("Using specification-focused similarity matrix")
            except Exception as e:
                logger.warning(f"Failed to compute specification similarity: {e}, falling back to standard similarity")
                if self.similarity_matrix is None:
                    self.compute_similarity_matrix()
                similarity_matrix = self.similarity_matrix
        else:
            if self.similarity_matrix is None:
                self.compute_similarity_matrix()
            similarity_matrix = self.similarity_matrix
        
        try:
            # Find laptop index using laptop_id
            laptop_mask = self.df_laptop['laptop_id'] == laptop_id
            if not laptop_mask.any():
                raise ValueError(f"Laptop with ID {laptop_id} not found")
            
            laptop_idx = laptop_mask.idxmax()
            
            # Get similarity scores for this laptop
            similarities = similarity_matrix[laptop_idx]
            
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
                
                # Use the already normalized similarity score
                normalized_similarity = similarities[original_idx]
                
                recommendations.append({
                    'laptop_id': laptop_data['laptop_id'],
                    'asin': laptop_data['asin'],
                    'title_y': laptop_data['title_y_clean'],  # Fix: use title_y instead of title
                    'brand': laptop_data.get('brand', f"Brand_{laptop_data['brand_encoded']}"),  # Use actual brand name if available
                    'price_myr': laptop_data['price_myr'],
                    'average_rating': laptop_data['average_rating'],  # Fix: use average_rating instead of rating
                    'rating_number': laptop_data.get('rating_number', 0),  # Include rating count
                    'similarity_score': normalized_similarity,
                    'features': laptop_data['features_clean'],
                    'images_y': laptop_data.get('images_y'),  # Include media columns
                    'videos': laptop_data.get('videos')
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
            # Apply filtering first
            filtered_df = self.df_laptop.copy()
            
            # Handle budget range filtering
            if 'budget_range' in preferences and preferences['budget_range']:
                budget_min, budget_max = preferences['budget_range']
                if budget_min is not None and budget_max is not None:
                    filtered_df = filtered_df[
                        (filtered_df['price_myr'] >= budget_min) & 
                        (filtered_df['price_myr'] <= budget_max)
                    ]
                    logger.info(f"Budget filtering applied: RM {budget_min} - RM {budget_max}, {len(filtered_df)} laptops remaining")
            
            # If no laptops match budget, return empty list
            if len(filtered_df) == 0:
                logger.warning("No laptops match the specified budget range")
                return []
            
            # Create preference vector
            preference_vector = self._create_preference_vector(preferences)
            
            # Calculate similarity to preference vector for filtered dataset
            filtered_indices = filtered_df.index
            filtered_feature_matrix = self.feature_matrix[filtered_indices]
            
            similarities = cosine_similarity([preference_vector], filtered_feature_matrix)[0]
            
            # Get top recommendations from filtered dataset
            top_indices = np.argsort(similarities)[::-1][:n_recommendations]
            
            recommendations = []
            for idx in top_indices:
                laptop_data = filtered_df.iloc[idx]
                
                # Use the similarity score directly (already normalized in matrix)
                normalized_similarity = similarities[idx]
                
                recommendations.append({
                    'laptop_id': laptop_data['laptop_id'],  # Add laptop_id as primary key
                    'laptop_id': laptop_data.get('laptop_id'),
                    'asin': laptop_data['asin'],
                    'title_y': laptop_data['title_y_clean'],  # Changed from 'title' to 'title_y'
                    'brand': laptop_data.get('brand', f"Brand_{laptop_data['brand_encoded']}"),  # Use actual brand name if available
                    'price_myr': laptop_data['price_myr'],
                    'average_rating': laptop_data['average_rating'],  # Changed from 'rating' to 'average_rating'
                    'rating_number': laptop_data.get('rating_number', 0),  # Include rating count
                    'similarity_score': normalized_similarity,
                    'features': laptop_data['features_clean'],
                    'images_y': laptop_data.get('images_y'),  # Include media columns
                    'videos': laptop_data.get('videos')
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
            if 'search_terms' in preferences and preferences['search_terms'] and self.tfidf_vectorizer is not None:
                search_text = ' '.join(preferences['search_terms'])
                # Use the same TF-IDF vectorizer that was used to create the feature matrix
                search_vector = self.tfidf_vectorizer.transform([search_text]).toarray()[0]
                
                # Map to text features in our matrix - use the actual number of text features
                text_feature_count = search_vector.shape[0]
                if text_feature_count <= preference_vector.shape[0]:
                    preference_vector[:text_feature_count] = search_vector
                else:
                    # If search vector is larger, truncate it
                    preference_vector[:preference_vector.shape[0]] = search_vector[:preference_vector.shape[0]]
            
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
            if 'brand_preference' in preferences and preferences['brand_preference']:
                # Find brand encoding - handle both string brand names and encoded values
                brand_pref = preferences['brand_preference']
                
                # First try to match by original brand name if available
                if 'brand' in self.df_laptop.columns:
                    brand_mask = self.df_laptop['brand'].str.lower() == brand_pref.lower()
                    if brand_mask.any():
                        # Get the encoded value for this brand
                        brand_encoded_value = self.df_laptop.loc[brand_mask, 'brand_encoded'].iloc[0]
                        
                        # Only proceed if the brand is not unknown (encoded value != 0)
                        if brand_encoded_value != 0:
                            # Set the corresponding brand feature in the preference vector
                            brand_feature_idx = self._get_brand_feature_index(brand_encoded_value)
                            if brand_feature_idx is not None:
                                preference_vector[brand_feature_idx] = 1.0
                                logger.info(f"Brand preference '{brand_pref}' mapped to encoded value {brand_encoded_value}")
                            else:
                                logger.warning(f"Could not find feature index for brand '{brand_pref}' (encoded value {brand_encoded_value})")
                        else:
                            logger.warning(f"Brand '{brand_pref}' is marked as unknown (encoded value 0) - skipping brand preference")
                else:
                    # Fallback: try to match encoded value directly
                    try:
                        brand_encoded_value = int(brand_pref)
                        
                        # Only proceed if the brand is not unknown (encoded value != 0)
                        if brand_encoded_value != 0:
                            brand_mask = self.df_laptop['brand_encoded'] == brand_encoded_value
                            if brand_mask.any():
                                brand_feature_idx = self._get_brand_feature_index(brand_encoded_value)
                                if brand_feature_idx is not None:
                                    preference_vector[brand_feature_idx] = 1.0
                                    logger.info(f"Brand preference '{brand_pref}' matched to encoded value {brand_encoded_value}")
                                else:
                                    logger.warning(f"Could not find feature index for brand encoded value {brand_encoded_value}")
                            else:
                                logger.warning(f"No laptops found with brand encoded value {brand_encoded_value}")
                        else:
                            logger.warning(f"Brand preference '{brand_pref}' is encoded as 0 (unknown) - skipping brand preference")
                    except ValueError:
                        logger.warning(f"Could not parse brand preference '{brand_pref}' as integer")
            
            return preference_vector
            
        except Exception as e:
            logger.error(f"Error creating preference vector: {str(e)}")
            raise
    
    def _get_brand_feature_index(self, brand_encoded_value: int) -> Optional[int]:
        """
        Find the feature index for a specific brand encoded value.
        
        Args:
            brand_encoded_value: The encoded brand value
            
        Returns:
            int: Feature index in the feature matrix, or None if not found
        """
        try:
            # Look for the brand feature in feature names
            brand_feature_name = f"brand_{brand_encoded_value}"
            
            for i, feature_name in enumerate(self.feature_names):
                if feature_name == brand_feature_name:
                    return i
            
            # If not found, provide more detailed error information
            brand_features = [name for name in self.feature_names if 'brand' in name.lower()]
            logger.warning(f"Could not find feature index for brand encoded value {brand_encoded_value}")
            logger.info(f"Available brand features: {brand_features}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding brand feature index: {str(e)}")
            return None
    
    def get_available_brands(self) -> List[str]:
        """
        Get list of available brands in the dataset (excluding unknown/empty brands).
        
        Returns:
            List[str]: List of available brand names
        """
        try:
            if 'brand' in self.df_laptop.columns:
                # Get unique brands, excluding empty/unknown values
                available_brands = self.df_laptop['brand'].dropna().unique()
                # Filter out empty strings and 'Unknown' values
                available_brands = [brand for brand in available_brands 
                                 if brand and str(brand).strip() and str(brand).lower() not in ['unknown', 'n/a', '']]
                return sorted(available_brands)
            else:
                logger.warning("Brand column not found in dataset")
                return []
                
        except Exception as e:
            logger.error(f"Error getting available brands: {str(e)}")
            return []
    
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
        """Save the trained model and parameters using joblib for optimized serialization."""
        from joblib import dump
        
        try:
            model_data = {
                'feature_matrix': self.feature_matrix,
                'similarity_matrix': self.similarity_matrix,
                'feature_names': self.feature_names,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'scaler': self.scaler,
                'config': self.config,
                'df_laptop_shape': self.df_laptop.shape if self.df_laptop is not None else None,
                'df_rating_shape': self.df_rating.shape if self.df_rating is not None else None
            }
            
            # Use joblib for efficient serialization of scikit-learn objects
            dump(model_data, filepath, compress=3)  # compress=3 for good compression/speed balance
            
            logger.info(f"Content-based model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving content-based model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a previously saved model using joblib."""
        from joblib import load
        
        try:
            model_data = load(filepath)
            
            # Validate model data structure
            required_keys = ['feature_matrix', 'similarity_matrix', 'feature_names', 'config']
            for key in required_keys:
                if key not in model_data:
                    raise ValueError(f"Missing required key '{key}' in saved model")
            
            self.feature_matrix = model_data['feature_matrix']
            self.similarity_matrix = model_data['similarity_matrix']
            self.feature_names = model_data['feature_names']
            self.tfidf_vectorizer = model_data.get('tfidf_vectorizer')
            self.scaler = model_data.get('scaler')
            self.config = model_data['config']
            
            # Log model information
            logger.info(f"Content-based model loaded from {filepath}")
            logger.info(f"Feature matrix shape: {self.feature_matrix.shape}")
            logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")
            logger.info(f"Number of features: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Error loading content-based model: {str(e)}")
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
