"""
Main Driver for Laptop Recommender System

This module combines Content-Based Filtering and Collaborative Filtering algorithms
to provide comprehensive laptop recommendations. It serves as the main interface
for users to get personalized laptop suggestions.

Author: Laptop Recommender System Team
License: MIT
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime
import random

# Import our recommendation algorithms
from content_based_filtering import create_content_based_filtering
from collaborative_filtering import create_collaborative_filtering
from data_preprocessing import LaptopDataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class LaptopRecommenderSystem:
    """Main Laptop Recommender System that combines multiple recommendation approaches."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Laptop Recommender System."""
        self.df_laptop = None
        self.df_rating = None
        self.content_based_filter = None
        self.collaborative_filter = None
        
        # Default configuration
        self.config = {
            'system': {
                'max_recommendations': 50,
                'min_similarity_threshold': 0.1,
                'enable_logging': True,
                'cache_results': True
            },
            'content_based': {
                'tfidf_params': {
                    'max_features': 2000,
                    'stop_words': 'english',
                    'ngram_range': (1, 2),
                    'min_df': 2,
                    'max_df': 0.9,
                    'use_idf': True,
                    'smooth_idf': True
                },
                'similarity_methods': {
                    'text_weight': 0.6,
                    'numerical_weight': 0.3,
                    'categorical_weight': 0.1
                }
            },
            'collaborative': {
                'matrix_factorization': {
                    'n_components': 80,
                    'random_state': 42,
                    'max_iter': 300
                },
                'similarity_methods': {
                    'min_common_items': 2,
                    'min_common_users': 2
                }
            },
            'hybrid': {
                'content_based_weight': 0.4,
                'collaborative_weight': 0.6,
                'diversity_weight': 0.2
            }
        }
        
        # Update with custom configuration if provided
        if config:
            self._update_config(config)
        
        logger.info("Laptop Recommender System initialized successfully")
    
    def _update_config(self, config: Dict) -> None:
        """Update configuration with custom parameters."""
        for section, params in config.items():
            if section in self.config:
                self.config[section].update(params)
            else:
                self.config[section] = params
    
    def load_and_preprocess_data(self, force_reload: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess the laptop dataset.
        
        Args:
            force_reload: Whether to force reload data even if already loaded
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_laptop, df_rating)
        """
        if not force_reload and self.df_laptop is not None and self.df_rating is not None:
            logger.info("Data already loaded, skipping preprocessing")
            return self.df_laptop, self.df_rating
        
        logger.info("Loading and preprocessing data...")
        # Deterministic seeding for reproducibility
        try:
            self._set_global_seed(self.config.get('collaborative', {}).get('matrix_factorization', {}).get('random_state', 42))
        except Exception:
            pass
        
        try:
            # Initialize preprocessor
            preprocessor = LaptopDataPreprocessor()
            
            # Use the separated preprocessing pipeline that includes benchmark scraper
            self.df_laptop, self.df_rating = preprocessor.preprocess_separated_pipeline()
            
            logger.info(f"Data loaded successfully. Laptop data: {self.df_laptop.shape}, Rating data: {self.df_rating.shape}")
            
            return self.df_laptop, self.df_rating
            
        except Exception as e:
            logger.error(f"Error loading and preprocessing data: {str(e)}")
            raise

    def _set_global_seed(self, seed: int = 42) -> None:
        """Set seeds for reproducibility across numpy, random, and sklearn where applicable."""
        try:
            np.random.seed(seed)
            random.seed(seed)
        except Exception:
            pass
    
    def initialize_recommendation_engines(self) -> None:
        """Initialize both content-based and collaborative filtering engines."""
        if self.df_laptop is None or self.df_rating is None:
            raise ValueError("Data must be loaded before initializing recommendation engines")
        
        logger.info("Initializing recommendation engines...")
        
        try:
            # Initialize content-based filtering
            self.content_based_filter = create_content_based_filtering(
                self.df_laptop, 
                self.df_rating,
                self.config['content_based']
            )
            
            # Initialize collaborative filtering
            self.collaborative_filter = create_collaborative_filtering(
                self.df_laptop,
                self.df_rating,
                self.config['collaborative']
            )
            
            # Automatically load pre-trained models if available
            logger.info("Loading pre-trained models...")
            
            # Load content-based model
            try:
                self.content_based_filter.create_feature_matrix()
                logger.info("✅ Content-based model loaded successfully")
                try:
                    fm_shape = getattr(self.content_based_filter, 'feature_matrix', None)
                    sm_shape = getattr(self.content_based_filter, 'similarity_matrix', None)
                    if fm_shape is not None:
                        logger.info(f"Content-based feature matrix shape: {self.content_based_filter.feature_matrix.shape}")
                    if sm_shape is not None:
                        logger.info(f"Content-based similarity matrix shape: {self.content_based_filter.similarity_matrix.shape}")
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Could not load content-based model: {e}")
            
            # Load collaborative filtering model
            try:
                self.collaborative_filter.create_user_item_matrix()
                logger.info("✅ Collaborative filtering model loaded successfully")
                try:
                    uim_shape = getattr(self.collaborative_filter, 'user_item_matrix', None)
                    if uim_shape is not None:
                        logger.info(f"Collaborative user-item matrix shape: {self.collaborative_filter.user_item_matrix.shape}")
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Could not load collaborative filtering model: {e}")
            
            logger.info("Recommendation engines initialized successfully")
            
            # Log unused/legacy model artifacts for cleanup visibility (no deletion)
            try:
                self._log_model_artifacts()
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Error initializing recommendation engines: {str(e)}")
            raise

    def _log_model_artifacts(self) -> None:
        """Log present model files and highlight likely-unused artifacts."""
        import os
        models_dir = 'models'
        if not os.path.isdir(models_dir):
            return
        try:
            files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if not files:
                return
            logger.info(f"Model artifacts found in '{models_dir}': {files}")
            expected = {'content_based_model.pkl', 'collaborative_model.pkl'}
            unused = [f for f in files if f not in expected]
            if unused:
                logger.info(f"Unused or legacy model files detected (safe to ignore): {unused}")
        except Exception as e:
            logger.debug(f"Could not scan model artifacts: {e}")
    
    def get_content_based_recommendations(self, preferences: Dict, 
                                        n_recommendations: int = None) -> List[Dict]:
        """
        Get recommendations using content-based filtering.
        
        Args:
            preferences: Dictionary containing user preferences
            n_recommendations: Number of recommendations to return
            
        Returns:
            List[Dict]: List of recommended laptops
        """
        if self.content_based_filter is None:
            raise ValueError("Content-based filtering engine not initialized")
        
        if n_recommendations is None:
            n_recommendations = self.config['system']['max_recommendations']
        
        try:
            logger.info(f"Generating content-based recommendations for preferences: {preferences}")
            
            recommendations = self.content_based_filter.get_recommendations_by_preferences(
                preferences, n_recommendations
            )
            
            # Add method information
            for rec in recommendations:
                rec['method'] = 'content_based'
                rec['algorithm'] = 'tfidf_cosine_similarity'
            
            logger.info(f"Generated {len(recommendations)} content-based recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {str(e)}")
            raise
    
    def get_collaborative_filtering_recommendations(self, user_id: str, 
                                                  method: str = 'hybrid',
                                                  n_recommendations: int = None) -> List[Dict]:
        """
        Get recommendations using collaborative filtering.
        
        Args:
            user_id: User ID for recommendations
            method: Recommendation method ('user_based', 'item_based', 'matrix_factorization', 'hybrid')
            n_recommendations: Number of recommendations to return
            
        Returns:
            List[Dict]: List of recommended laptops
        """
        if self.collaborative_filter is None:
            raise ValueError("Collaborative filtering engine not initialized")
        
        if n_recommendations is None:
            n_recommendations = self.config['system']['max_recommendations']
        
        try:
            logger.info(f"Generating collaborative filtering recommendations for user {user_id} using {method} method")
            
            if method == 'user_based':
                recommendations = self.collaborative_filter.get_user_based_recommendations(
                    user_id, n_recommendations
                )
            elif method == 'item_based':
                recommendations = self.collaborative_filter.get_item_based_recommendations(
                    user_id, n_recommendations
                )
            elif method == 'matrix_factorization':
                recommendations = self.collaborative_filter.get_matrix_factorization_recommendations(
                    user_id, n_recommendations
                )
            elif method == 'hybrid':
                recommendations = self.collaborative_filter.get_hybrid_recommendations(
                    user_id, n_recommendations
                )
            else:
                raise ValueError(f"Unsupported collaborative filtering method: {method}")
            
            logger.info(f"Generated {len(recommendations)} collaborative filtering recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting collaborative filtering recommendations: {str(e)}")
            raise
    
    def get_hybrid_recommendations(self, user_id: str, preferences: Dict,
                                 n_recommendations: int = None,
                                 weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """Get hybrid recommendations combining both approaches with dynamic weights."""
        if n_recommendations is None:
            n_recommendations = self.config['system']['max_recommendations']
        
        if weights is None:
            weights = self._get_dynamic_hybrid_weights(preferences)
        
        try:
            logger.info(f"Generating hybrid recommendations for user {user_id}")
            
            # Get recommendations from both methods
            content_based_recs = self.get_content_based_recommendations(
                preferences, n_recommendations * 2
            )
            collaborative_recs = self.get_collaborative_filtering_recommendations(
                user_id, 'hybrid', n_recommendations * 2
            )
            
            return self._combine_recommendations(content_based_recs, collaborative_recs, 
                                               weights, n_recommendations, 'hybrid')
            
        except Exception as e:
            logger.error(f"Error getting hybrid recommendations: {str(e)}")
            raise

    def _get_dynamic_hybrid_weights(self, preferences: Dict) -> Dict[str, float]:
        """Compute dynamic hybrid weights based on use_case/priority."""
        default_cb = self.config['hybrid']['content_based_weight']
        default_cf = self.config['hybrid']['collaborative_weight']
        use_case = str(preferences.get('use_case', '')).lower()
        priority = str(preferences.get('priority', '')).lower()

        cb_w, cf_w = default_cb, default_cf
        # Performance/professional queries rely more on specs/text
        if priority in ['performance'] or use_case in ['professional', 'work', 'business']:
            cb_w, cf_w = max(0.6, default_cb), min(0.4, default_cf)
        # Brand-loyal or returning users with history could lean more on CF (kept simple here)
        # else keep defaults

        # Normalize to sum to 1.0
        total = cb_w + cf_w
        if total == 0:
            return {'content_based': 0.5, 'collaborative': 0.5}
        return {'content_based': cb_w / total, 'collaborative': cf_w / total}
    
    def get_hybrid_recommendations_auto(self, preferences: Dict,
                                      n_recommendations: int = None,
                                      weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """Get automatic hybrid recommendations without requiring a specific user_id."""
        if n_recommendations is None:
            n_recommendations = self.config['system']['max_recommendations']
        
        if weights is None:
            weights = {
                'content_based': self.config['hybrid']['content_based_weight'],
                'collaborative': self.config['hybrid']['collaborative_weight']
            }
        
        try:
            logger.info("Generating automatic hybrid recommendations based on dataset analysis")
            
            # Get content-based recommendations
            content_based_recs = self.get_content_based_recommendations(
                preferences, n_recommendations * 2
            )
            
            # Get popular collaborative recommendations
            collaborative_recs = self.collaborative_filter.get_popular_recommendations(
                preferences, n_recommendations * 2
            )
            
            return self._combine_recommendations(content_based_recs, collaborative_recs, 
                                               weights, n_recommendations, 'hybrid_auto')
            
        except Exception as e:
            logger.error(f"Error getting automatic hybrid recommendations: {str(e)}")
            raise
    
    def _combine_recommendations(self, content_based_recs: List[Dict], 
                                collaborative_recs: List[Dict], weights: Dict[str, float],
                                n_recommendations: int, method: str) -> List[Dict]:
        """Helper function to combine recommendations and reduce code duplication."""
        combined_recs = {}
        
        # Process content-based recommendations
        for rec in content_based_recs:
            asin = rec['asin']
            if asin not in combined_recs:
                combined_recs[asin] = {
                    'asin': asin,
                    'title': rec.get('title_y', rec.get('title', 'Unknown')),
                    'brand': rec.get('brand', 'Unknown'),
                    'price_myr': rec.get('price_myr', 0),
                    'rating': rec.get('average_rating', rec.get('rating', 0)),
                    'combined_score': 0,
                    'methods': [],
                    'scores': {}
                }
            
            normalized_score = rec.get('similarity_score', rec.get('recommendation_score', 0))
            combined_recs[asin]['combined_score'] += weights['content_based'] * normalized_score
            combined_recs[asin]['methods'].append('content_based')
            combined_recs[asin]['scores']['content_based'] = normalized_score
        
        # Process collaborative filtering recommendations
        for rec in collaborative_recs:
            asin = rec['asin']
            if asin not in combined_recs:
                combined_recs[asin] = {
                    'asin': asin,
                    'title': rec.get('title', 'Unknown'),
                    'brand': rec.get('brand', 'Unknown'),
                    'price_myr': rec.get('price_myr', 0),
                    'rating': rec.get('rating', 0),
                    'combined_score': 0,
                    'methods': [],
                    'scores': {}
                }
            
            normalized_score = rec.get('recommendation_score', 0)
            combined_recs[asin]['combined_score'] += weights['collaborative'] * normalized_score
            combined_recs[asin]['methods'].append('collaborative')
            combined_recs[asin]['scores']['collaborative'] = normalized_score
        
        # Normalize per-algorithm scores to [0,1] to prevent scale domination
        try:
            cb_scores = [v['scores'].get('content_based', None) for v in combined_recs.values()]
            cf_scores = [v['scores'].get('collaborative', None) for v in combined_recs.values()]
            def norm(values):
                nums = [x for x in values if x is not None]
                if not nums:
                    return lambda x: 0.0
                vmin, vmax = min(nums), max(nums)
                if vmax == vmin:
                    return lambda x: 1.0 if x is not None else 0.0
                return lambda x: (x - vmin) / (vmax - vmin) if x is not None else 0.0
            n_cb = norm(cb_scores)
            n_cf = norm(cf_scores)
            for asin, rec in combined_recs.items():
                rec['combined_score'] = (
                    weights['content_based'] * n_cb(rec['scores'].get('content_based')) +
                    weights['collaborative'] * n_cf(rec['scores'].get('collaborative'))
                )
        except Exception:
            # Fallback: keep unnormalized combined_score
            pass

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
                'title_y': rec['title'],
                'brand': rec['brand'],
                'price_myr': rec['price_myr'],
                'average_rating': rec['rating'],
                'recommendation_score': rec['combined_score'],
                'method': method,
                'methods_used': rec['methods'],
                'individual_scores': rec['scores'],
                'explanation': f"Smart hybrid: matches your preferences + popular with similar users" if method == 'hybrid_auto' else f"Combined from {len(rec['methods'])} methods: {', '.join(rec['methods'])}"
            }
            formatted_recommendations.append(formatted_rec)

        # Reorder to place items similar to the top-1 after it (if similarity matrix available)
        try:
            if self.content_based_filter and getattr(self.content_based_filter, 'similarity_matrix', None) is not None:
                sim_matrix = self.content_based_filter.similarity_matrix
                # Map asin to index in df_laptop
                asin_to_idx = {row['asin']: idx for idx, row in self.df_laptop[['asin']].reset_index().to_dict('index').items()}
                # Find index of top-1
                if formatted_recommendations:
                    top_asin = formatted_recommendations[0]['asin']
                    if top_asin in asin_to_idx:
                        t_idx = asin_to_idx[top_asin]
                        # Score others by similarity to top-1
                        rest = formatted_recommendations[1:]
                        rest_sorted = sorted(rest, key=lambda r: sim_matrix[t_idx][asin_to_idx.get(r['asin'], t_idx)] if asin_to_idx.get(r['asin']) is not None else 0.0, reverse=True)
                        formatted_recommendations = [formatted_recommendations[0]] + rest_sorted
        except Exception:
            pass

        logger.info(f"Generated {len(formatted_recommendations)} {method} recommendations")
        return formatted_recommendations

    def get_recommendations_by_use_case(self, use_case: str, budget: float = None,
                                       n_recommendations: int = None) -> List[Dict]:
        """
        Get recommendations based on specific use case.
        
        Args:
            use_case: Intended use case (gaming, work, student, etc.)
            budget: Maximum budget in MYR
            n_recommendations: Number of recommendations to return
            
        Returns:
            List[Dict]: List of recommended laptops
        """
        if n_recommendations is None:
            n_recommendations = self.config['system']['max_recommendations']
        
        try:
            logger.info(f"Generating recommendations for use case: {use_case}")
            
            # Define use case preferences
            use_case_preferences = self._get_use_case_preferences(use_case)
            
            # Add budget constraint if specified
            if budget:
                use_case_preferences['max_price'] = budget
            
            # Get content-based recommendations
            recommendations = self.get_content_based_recommendations(
                use_case_preferences, n_recommendations
            )
            
            # Filter by budget if specified
            if budget:
                recommendations = [rec for rec in recommendations if rec['price_myr'] <= budget]
            
            logger.info(f"Generated {len(recommendations)} recommendations for use case: {use_case}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting use case recommendations: {str(e)}")
            raise
    
    def _get_use_case_preferences(self, use_case: str) -> Dict[str, Any]:
        """Get preferences based on use case."""
        use_case_mapping = {
            'gaming': {
                'search_terms': ['gaming', 'gpu', 'graphics', 'performance', 'high-end'],
                'min_rating': 4.0,
                'specifications': ['high_performance', 'dedicated_gpu']
            },
            'work': {
                'search_terms': ['business', 'professional', 'work', 'office', 'productivity'],
                'min_rating': 3.5,
                'specifications': ['reliable', 'business_class']
            },
            'student': {
                'search_terms': ['student', 'budget', 'affordable', 'basic', 'study'],
                'min_rating': 3.0,
                'specifications': ['budget_friendly', 'basic_specs']
            },
            'creative': {
                'search_terms': ['creative', 'design', 'video', 'photo', 'editing'],
                'min_rating': 4.0,
                'specifications': ['high_resolution', 'color_accurate']
            },
            'travel': {
                'search_terms': ['portable', 'lightweight', 'travel', 'compact', 'battery'],
                'min_rating': 3.5,
                'specifications': ['portable', 'long_battery']
            }
        }
        
        return use_case_mapping.get(use_case.lower(), {
            'search_terms': [use_case],
            'min_rating': 3.0
        })
    
    def find_similar_laptops(self, laptop_id: int, n_recommendations: int = None,
                            method: str = 'content_based', use_spec_similarity: bool = True) -> List[Dict]:
        """
        Find laptops similar to a given laptop based on specifications and benchmarks.
        
        Args:
            laptop_id: laptop_id (integer) of the reference laptop
            n_recommendations: Number of similar laptops to return
            method: Method to use ('content_based' or 'collaborative')
            use_spec_similarity: Whether to use specification-focused similarity for better results
            
        Returns:
            List[Dict]: List of similar laptops with benchmark and specification data
        """
        if n_recommendations is None:
            n_recommendations = self.config['system']['max_recommendations']
        
        try:
            logger.info(f"Finding similar laptops to {laptop_id} using {method} method (spec_similarity={use_spec_similarity})")
            
            if method == 'content_based':
                if self.content_based_filter is None:
                    raise ValueError("Content-based filtering engine not initialized")
                
                # Use specification-focused similarity for better results
                recommendations = self.content_based_filter.get_recommendations(
                    laptop_id, n_recommendations, use_spec_similarity=use_spec_similarity
                )
                
                # Add benchmark and specification information to recommendations
                for rec in recommendations:
                    # Get full laptop data for this recommendation
                    laptop_data = self.df_laptop[self.df_laptop['laptop_id'] == rec['laptop_id']]
                    if not laptop_data.empty:
                        laptop_row = laptop_data.iloc[0]
                        
                        # Add essential fields that template expects
                        rec['title_y'] = laptop_row.get('title_y_clean', laptop_row.get('title_y', 'Unknown Title'))
                        rec['features'] = laptop_row.get('features_clean', laptop_row.get('features', ''))
                        rec['average_rating'] = laptop_row.get('average_rating', 0.0)
                        rec['price_myr'] = laptop_row.get('price_myr', 0.0)
                        
                        # Add brand information
                        if 'brand_original' in laptop_row and pd.notna(laptop_row['brand_original']):
                            rec['brand'] = laptop_row['brand_original']
                        elif 'brand_encoded' in laptop_row:
                            rec['brand'] = f"Brand_{laptop_row['brand_encoded']}"
                        else:
                            rec['brand'] = 'Unknown Brand'
                        
                        # Add images and videos
                        rec['images_y'] = laptop_row.get('images_y', '')
                        rec['videos'] = laptop_row.get('videos', '')
                        
                        # Add benchmark scores if available
                        if 'cpu_benchmark_score' in self.df_laptop.columns:
                            rec['cpu_benchmark_score'] = laptop_row.get('cpu_benchmark_score', 0)
                            rec['gpu_benchmark_score'] = laptop_row.get('gpu_benchmark_score', 0)
                            rec['total_benchmark_score'] = laptop_row.get('total_benchmark_score', 0)
                            rec['performance_tier'] = laptop_row.get('performance_tier', 'Unknown')
                            rec['gaming_capability'] = laptop_row.get('gaming_capability', 'Unknown')
                            
                            # Add detailed specifications
                            rec['ram_gb'] = laptop_row.get('ram_gb', 0)
                            rec['storage_gb'] = laptop_row.get('storage_gb', 0)
                            rec['screen_size_inches'] = laptop_row.get('screen_size_inches', 0)
                            rec['processor_model'] = laptop_row.get('processor_model', 'Unknown')
                            rec['gpu_model'] = laptop_row.get('gpu_model', 'Unknown')
                            rec['storage_type'] = laptop_row.get('storage_type', 'Unknown')
                            rec['ram_type'] = laptop_row.get('ram_type', 'Unknown')
                
            elif method == 'collaborative':
                if self.collaborative_filter is None:
                    raise ValueError("Collaborative filtering engine not initialized")
                
                # For collaborative filtering, we need a user context
                # This is a simplified approach - in practice, you might want to use
                # item-based collaborative filtering or find users who liked this laptop
                recommendations = self.collaborative_filter.get_item_based_recommendations(
                    user_id=0,  # Placeholder user ID
                    n_recommendations=n_recommendations
                )
                
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            logger.info(f"Found {len(recommendations)} similar laptops")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error finding similar laptops: {str(e)}")
            raise
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of the system status and data."""
        try:
            summary = {
                'system_status': 'initialized' if self.df_laptop is not None else 'not_initialized',
                'data_info': {
                    'laptop_records': len(self.df_laptop) if self.df_laptop is not None else 0,
                    'rating_records': len(self.df_rating) if self.df_rating is not None else 0,
                    'unique_users': self.df_rating['user_id_encoded'].nunique() if self.df_rating is not None else 0,
                    'unique_laptops': self.df_laptop['asin'].nunique() if self.df_laptop is not None else 0
                },
                'engines_status': {
                    'content_based': self.content_based_filter is not None,
                    'collaborative': self.collaborative_filter is not None
                },
                'configuration': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting system summary: {str(e)}")
            raise
    
    def save_recommendations(self, recommendations: List[Dict], filepath: str) -> None:
        """Save recommendations to a file."""
        try:
            import json
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Convert recommendations
            serializable_recs = []
            for rec in recommendations:
                serializable_rec = {}
                for key, value in rec.items():
                    serializable_rec[key] = convert_numpy_types(value)
                serializable_recs.append(serializable_rec)
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_recs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Recommendations saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}")
            raise
    
    def get_laptop_by_id(self, laptop_id: int) -> Optional[Dict]:
        """
        Get laptop details by laptop_id.
        
        Args:
            laptop_id: The laptop ID to search for
            
        Returns:
            Dict with laptop details or None if not found
        """
        try:
            if self.df_laptop is None:
                logger.error("Laptop data not loaded")
                return None
            
            # Find laptop by laptop_id
            laptop_row = self.df_laptop[self.df_laptop['laptop_id'] == laptop_id]
            
            if laptop_row.empty:
                logger.warning(f"Laptop with ID {laptop_id} not found")
                return None
            
            # Convert to dictionary
            laptop_data = laptop_row.iloc[0].to_dict()
            
            # Ensure brand is available
            if 'brand' not in laptop_data and 'brand_original' in laptop_data:
                laptop_data['brand'] = laptop_data['brand_original']
            elif 'brand' not in laptop_data and 'brand_encoded' in laptop_data:
                laptop_data['brand'] = f"Brand_{laptop_data['brand_encoded']}"
            
            # Ensure title_y is available (template expects this)
            if 'title_y' not in laptop_data and 'title_y_clean' in laptop_data:
                laptop_data['title_y'] = laptop_data['title_y_clean']
            elif 'title_y' not in laptop_data and 'title' in laptop_data:
                laptop_data['title_y'] = laptop_data['title']
            
            # Ensure features is available
            if 'features' not in laptop_data and 'features_clean' in laptop_data:
                laptop_data['features'] = laptop_data['features_clean']
            
            # Ensure average_rating is available
            if 'average_rating' not in laptop_data:
                laptop_data['average_rating'] = 0.0
            
            # Ensure price_myr is available
            if 'price_myr' not in laptop_data:
                laptop_data['price_myr'] = 0.0
            
            # Ensure specifications are available (these are extracted during preprocessing)
            if 'ram_gb' not in laptop_data:
                laptop_data['ram_gb'] = None
            if 'storage_gb' not in laptop_data:
                laptop_data['storage_gb'] = None
            if 'screen_size_inches' not in laptop_data:
                laptop_data['screen_size_inches'] = None
            if 'processor_model' not in laptop_data:
                laptop_data['processor_model'] = None
            if 'gpu_model' not in laptop_data:
                laptop_data['gpu_model'] = None
            if 'storage_type' not in laptop_data:
                laptop_data['storage_type'] = None
            if 'ram_type' not in laptop_data:
                laptop_data['ram_type'] = None
            if 'os' not in laptop_data:
                laptop_data['os'] = None
            
            return laptop_data
            
        except Exception as e:
            logger.error(f"Error getting laptop by ID {laptop_id}: {str(e)}")
            return None
    
    @property
    def preprocessor(self):
        """
        Get the data preprocessor instance.
        This is a property to maintain compatibility with existing code.
        """
        if not hasattr(self, '_preprocessor'):
            self._preprocessor = LaptopDataPreprocessor()
        return self._preprocessor


def create_laptop_recommender_system(config: Optional[Dict] = None) -> LaptopRecommenderSystem:
    """
    Factory function to create and configure LaptopRecommenderSystem instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LaptopRecommenderSystem: Configured instance
    """
    return LaptopRecommenderSystem(config)


def main():
    """Main function to demonstrate the Laptop Recommender System."""
    print("Laptop Recommender System")
    print("=" * 50)
    
    try:
        # Create recommender system
        recommender = create_laptop_recommender_system()
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df_laptop, df_rating = recommender.load_and_preprocess_data()
        
        # Initialize recommendation engines
        print("Initializing recommendation engines...")
        recommender.initialize_recommendation_engines()
        
        # Get system summary
        summary = recommender.get_system_summary()
        print(f"\nSystem Summary:")
        print(f"Laptop records: {summary['data_info']['laptop_records']}")
        print(f"Rating records: {summary['data_info']['rating_records']}")
        print(f"Unique users: {summary['data_info']['unique_users']}")
        print(f"Unique laptops: {summary['data_info']['unique_laptops']}")
        
        # Example: Get recommendations for gaming use case
        print(f"\nGetting gaming laptop recommendations...")
        gaming_recs = recommender.get_recommendations_by_use_case('gaming', budget=5000)
        
        if gaming_recs:
            print(f"\nTop Gaming Laptop Recommendations:")
            for i, rec in enumerate(gaming_recs[:5], 1):
                print(f"{i}. {rec['title']}")
                print(f"   Brand: {rec['brand']}, Price: RM {rec['price_myr']:.2f}")
                print(f"   Rating: {rec['rating']:.1f}, Score: {rec['recommendation_score']:.3f}")
                print()
        
        # Example: Get collaborative filtering recommendations
        print(f"Getting collaborative filtering recommendations...")
        try:
            cf_recs = recommender.get_collaborative_filtering_recommendations(
                user_id="AHV6QCNBJNSGLATP56JAWJ3C4G2A", method='hybrid', n_recommendations=5
            )
            
            if cf_recs:
                print(f"\nTop Collaborative Filtering Recommendations:")
                for i, rec in enumerate(cf_recs[:5], 1):
                    print(f"{i}. {rec['title']}")
                    print(f"   Brand: {rec['brand']}, Price: RM {rec['price_myr']:.2f}")
                    print(f"   Rating: {rec['rating']:.1f}, Score: {rec['recommendation_score']:.3f}")
                    print()
        except Exception as e:
            print(f"Collaborative filtering not available: {e}")
        
        print("Laptop Recommender System demonstration completed successfully!")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        logger.error(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    main()
