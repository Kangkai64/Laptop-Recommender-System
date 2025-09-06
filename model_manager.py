"""
Model Manager for Laptop Recommender System

This module provides a unified interface for loading and managing trained models
for the web application. It handles model loading, caching, and provides a
unified recommendation API.

Author: Laptop Recommender System Team
License: MIT
"""

import pickle
import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np
import pandas as pd

# Import our recommendation algorithms
from content_based_filtering import ContentBasedFiltering
from collaborative_filtering import CollaborativeFiltering

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Centralized model manager for the laptop recommender system.
    
    This class handles:
    - Loading trained models from pickle files
    - Caching models in memory for fast access
    - Providing unified recommendation interface
    - Managing database mappings
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory containing trained model files
        """
        self.models_dir = models_dir
        self.models_loaded = False
        self.load_start_time = None
        
        # Model instances
        self.content_model = None
        self.collaborative_model = None
        
        # Database mappings
        self.laptop_metadata = {}
        self.user_profiles = {}
        self.brand_mapping = {}
        self.category_mapping = {}
        
        # Model metadata
        self.model_metadata = {}
        
        logger.info(f"ModelManager initialized with models directory: {models_dir}")
    
    def load_models(self) -> bool:
        """
        Load all trained models and mappings from pickle files.
        
        Returns:
            bool: True if all models loaded successfully, False otherwise
        """
        self.load_start_time = datetime.now()
        logger.info("Starting model loading process...")
        
        try:
            # Load laptop metadata
            self._load_laptop_metadata()
            
            # Load user profiles
            self._load_user_profiles()
            
            # Load brand mapping
            self._load_brand_mapping()
            
            # Load category mapping
            self._load_category_mapping()
            
            # Load content-based model
            self._load_content_model()
            
            # Load collaborative model
            self._load_collaborative_model()
            
            # Load model metadata
            self._load_model_metadata()
            
            self.models_loaded = True
            load_time = (datetime.now() - self.load_start_time).total_seconds()
            
            logger.info(f"✅ All models loaded successfully in {load_time:.2f} seconds")
            logger.info(f"   Laptop metadata: {len(self.laptop_metadata)//2} laptops")
            logger.info(f"   User profiles: {len(self.user_profiles)} users")
            logger.info(f"   Brand mappings: {len(set(self.brand_mapping.values()))} brands")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            self.models_loaded = False
            return False
    
    def _load_laptop_metadata(self):
        """Load laptop metadata mapping."""
        try:
            metadata_path = os.path.join(self.models_dir, "database_mappings_laptop_metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.laptop_metadata = pickle.load(f)
                logger.info(f"✅ Loaded laptop metadata: {len(self.laptop_metadata)//2} laptops")
            else:
                logger.warning(f"⚠️ Laptop metadata file not found: {metadata_path}")
        except Exception as e:
            logger.error(f"❌ Error loading laptop metadata: {e}")
            self.laptop_metadata = {}
    
    def _load_user_profiles(self):
        """Load user profiles mapping."""
        try:
            profiles_path = os.path.join(self.models_dir, "database_mappings_user_profiles.pkl")
            if os.path.exists(profiles_path):
                with open(profiles_path, 'rb') as f:
                    self.user_profiles = pickle.load(f)
                logger.info(f"✅ Loaded user profiles: {len(self.user_profiles)} users")
            else:
                logger.warning(f"⚠️ User profiles file not found: {profiles_path}")
        except Exception as e:
            logger.error(f"❌ Error loading user profiles: {e}")
            self.user_profiles = {}
    
    def _load_brand_mapping(self):
        """Load brand mapping."""
        try:
            brand_path = os.path.join(self.models_dir, "database_mappings_brand_mapping.pkl")
            if os.path.exists(brand_path):
                with open(brand_path, 'rb') as f:
                    self.brand_mapping = pickle.load(f)
                logger.info(f"✅ Loaded brand mapping: {len(set(self.brand_mapping.values()))} brands")
            else:
                logger.warning(f"⚠️ Brand mapping file not found: {brand_path}")
        except Exception as e:
            logger.error(f"❌ Error loading brand mapping: {e}")
            self.brand_mapping = {}
    
    def _load_category_mapping(self):
        """Load category mapping."""
        try:
            category_path = os.path.join(self.models_dir, "database_mappings_category_mapping.pkl")
            if os.path.exists(category_path):
                with open(category_path, 'rb') as f:
                    self.category_mapping = pickle.load(f)
                logger.info(f"✅ Loaded category mapping: {len(self.category_mapping)} categories")
            else:
                logger.warning(f"⚠️ Category mapping file not found: {category_path}")
        except Exception as e:
            logger.error(f"❌ Error loading category mapping: {e}")
            self.category_mapping = {}
    
    def _load_content_model(self):
        """Load content-based filtering model."""
        try:
            model_path = os.path.join(self.models_dir, "content_based_model.pkl")
            if os.path.exists(model_path):
                self.content_model = ContentBasedFiltering(None, None)
                self.content_model.load_model(model_path)
                logger.info("✅ Loaded content-based filtering model")
            else:
                logger.warning(f"⚠️ Content-based model file not found: {model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading content-based model: {e}")
            self.content_model = None
    
    def _load_collaborative_model(self):
        """Load collaborative filtering model."""
        try:
            model_path = os.path.join(self.models_dir, "collaborative_model.pkl")
            if os.path.exists(model_path):
                self.collaborative_model = CollaborativeFiltering(None, None)
                self.collaborative_model.load_model(model_path)
                logger.info("✅ Loaded collaborative filtering model")
            else:
                logger.warning(f"⚠️ Collaborative model file not found: {model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading collaborative model: {e}")
            self.collaborative_model = None
    
    def _load_model_metadata(self):
        """Load model metadata."""
        try:
            metadata_path = os.path.join(self.models_dir, "model_metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.model_metadata = pickle.load(f)
                logger.info("✅ Loaded model metadata")
            else:
                logger.warning(f"⚠️ Model metadata file not found: {metadata_path}")
        except Exception as e:
            logger.error(f"❌ Error loading model metadata: {e}")
            self.model_metadata = {}
    
    def is_ready(self) -> bool:
        """Check if all models are loaded and ready."""
        return (self.models_loaded and 
                self.content_model is not None and 
                self.collaborative_model is not None and 
                len(self.laptop_metadata) > 0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information."""
        return {
            'models_loaded': self.models_loaded,
            'content_model_ready': self.content_model is not None,
            'collaborative_model_ready': self.collaborative_model is not None,
            'laptop_metadata_count': len(self.laptop_metadata) // 2,
            'user_profiles_count': len(self.user_profiles),
            'brand_mapping_count': len(set(self.brand_mapping.values())),
            'load_time_seconds': (datetime.now() - self.load_start_time).total_seconds() if self.load_start_time else None,
            'system_ready': self.is_ready()
        }
    
    def recommend(self, 
                  user_id: Optional[int] = None,
                  algorithm: str = "content_based",
                  top_n: int = 10,
                  preferences: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Generate recommendations using the specified algorithm.
        
        Args:
            user_id: User ID for collaborative filtering (optional)
            algorithm: Algorithm to use ("content_based", "collaborative", "hybrid")
            top_n: Number of recommendations to return
            preferences: User preferences dictionary
            
        Returns:
            List of recommendation dictionaries with full laptop details
        """
        if not self.is_ready():
            logger.error("❌ Models not ready. Please load models first.")
            return []
        
        try:
            recommendations = []
            
            if algorithm == "content_based":
                recommendations = self._get_content_based_recommendations(preferences, top_n)
            
            elif algorithm == "collaborative":
                recommendations = self._get_collaborative_recommendations(user_id, preferences, top_n)
            
            elif algorithm == "hybrid":
                recommendations = self._get_hybrid_recommendations(user_id, preferences, top_n)
            
            else:
                logger.error(f"❌ Unknown algorithm: {algorithm}")
                return []
            
            # Enrich recommendations with full laptop details
            enriched_recommendations = self._enrich_recommendations(recommendations, algorithm)
            
            logger.info(f"✅ Generated {len(enriched_recommendations)} recommendations using {algorithm}")
            return enriched_recommendations[:top_n]
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return []
    
    def _get_content_based_recommendations(self, preferences: Optional[Dict], top_n: int) -> List[Dict]:
        """Get content-based recommendations."""
        if not preferences:
            preferences = {'budget_range': (0, 50000)}
        
        return self.content_model.get_recommendations_by_preferences(
            preferences, n_recommendations=top_n
        )
    
    def _get_collaborative_recommendations(self, user_id: Optional[int], preferences: Optional[Dict], top_n: int) -> List[Dict]:
        """Get collaborative filtering recommendations."""
        if user_id and user_id in self.user_profiles:
            return self.collaborative_model.get_hybrid_recommendations(
                user_id, n_recommendations=top_n
            )
        else:
            return self.collaborative_model.get_popular_recommendations(
                preferences, n_recommendations=top_n
            )
    
    def _get_hybrid_recommendations(self, user_id: Optional[int], preferences: Optional[Dict], top_n: int) -> List[Dict]:
        """Get hybrid recommendations combining content-based and collaborative filtering."""
        # Get content-based recommendations
        content_recs = self._get_content_based_recommendations(preferences, top_n // 2)
        
        # Get collaborative recommendations
        collab_recs = self._get_collaborative_recommendations(user_id, preferences, top_n // 2)
        
        # Combine and deduplicate
        all_recs = content_recs + collab_recs
        seen_asins = set()
        recommendations = []
        
        for rec in all_recs:
            asin = rec.get('asin')
            if asin and asin not in seen_asins:
                seen_asins.add(asin)
                recommendations.append(rec)
                if len(recommendations) >= top_n:
                    break
        
        return recommendations
    
    def _enrich_recommendations(self, recommendations: List[Dict], algorithm: str) -> List[Dict[str, Any]]:
        """Enrich recommendations with full laptop details from database mapping."""
        enriched_recommendations = []
        
        for rec in recommendations:
            asin = rec.get('asin')
            laptop_id = rec.get('laptop_id')
            
            # Get full laptop details from metadata
            if asin in self.laptop_metadata:
                laptop_details = self.laptop_metadata[asin].copy()
            elif laptop_id in self.laptop_metadata:
                laptop_details = self.laptop_metadata[laptop_id].copy()
            else:
                laptop_details = rec.copy()
            
            # Add recommendation metadata
            laptop_details['recommendation_score'] = rec.get('similarity_score', rec.get('recommendation_score', 0))
            laptop_details['algorithm_used'] = algorithm
            laptop_details['method'] = rec.get('method', algorithm)
            
            # Ensure all required fields are present
            laptop_details.setdefault('laptop_id', laptop_id)
            laptop_details.setdefault('asin', asin)
            laptop_details.setdefault('title', laptop_details.get('title', 'Unknown'))
            laptop_details.setdefault('brand', laptop_details.get('brand', 'Unknown'))
            laptop_details.setdefault('price_myr', laptop_details.get('price_myr', 0))
            laptop_details.setdefault('average_rating', laptop_details.get('average_rating', 0))
            
            enriched_recommendations.append(laptop_details)
        
        return enriched_recommendations
    
    def get_laptop_details(self, laptop_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get full laptop details by ID or ASIN.
        
        Args:
            laptop_id: Laptop ID or ASIN
            
        Returns:
            Dictionary with laptop details or None if not found
        """
        if laptop_id in self.laptop_metadata:
            return self.laptop_metadata[laptop_id]
        return None
    
    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user profile by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with user profile or None if not found
        """
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        return None
    
    def get_available_brands(self) -> List[str]:
        """Get list of available brands."""
        if self.brand_mapping:
            return [brand for brand in set(self.brand_mapping.values()) if isinstance(brand, str)]
        return []
    
    def get_brand_encoded(self, brand_name: str) -> Optional[int]:
        """Get encoded value for a brand name."""
        return self.brand_mapping.get(brand_name)
    
    def get_brand_name(self, brand_encoded: int) -> Optional[str]:
        """Get brand name for an encoded value."""
        return self.brand_mapping.get(brand_encoded)
    
    def get_price_categories(self) -> Dict[str, tuple]:
        """Get price category ranges."""
        return self.category_mapping.get('price_categories', {})
    
    def get_performance_categories(self) -> Dict[str, tuple]:
        """Get performance category ranges."""
        return self.category_mapping.get('performance_categories', {})
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata information."""
        return self.model_metadata.copy()
    
    def reload_models(self) -> bool:
        """Reload all models from disk."""
        logger.info("🔄 Reloading models...")
        self.models_loaded = False
        return self.load_models()


# Global model manager instance
_model_manager = None


def get_model_manager(models_dir: str = "models") -> ModelManager:
    """
    Get or create the global model manager instance.
    
    Args:
        models_dir: Directory containing trained model files
        
    Returns:
        ModelManager instance
    """
    global _model_manager
    
    if _model_manager is None:
        _model_manager = ModelManager(models_dir)
        _model_manager.load_models()
    
    return _model_manager


def initialize_models(models_dir: str = "models") -> bool:
    """
    Initialize the model manager and load all models.
    
    Args:
        models_dir: Directory containing trained model files
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global _model_manager
    
    _model_manager = ModelManager(models_dir)
    return _model_manager.load_models()


def get_recommendations(user_id: Optional[int] = None,
                       algorithm: str = "content_based",
                       top_n: int = 10,
                       preferences: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to get recommendations using the global model manager.
    
    Args:
        user_id: User ID for collaborative filtering (optional)
        algorithm: Algorithm to use ("content_based", "collaborative", "hybrid")
        top_n: Number of recommendations to return
        preferences: User preferences dictionary
        
    Returns:
        List of recommendation dictionaries
    """
    model_manager = get_model_manager()
    return model_manager.recommend(user_id, algorithm, top_n, preferences)


if __name__ == "__main__":
    # Test the model manager
    print("🧪 Testing Model Manager")
    print("=" * 30)
    
    # Initialize model manager
    model_manager = ModelManager()
    
    if model_manager.load_models():
        print("✅ Models loaded successfully")
        
        # Test system status
        status = model_manager.get_system_status()
        print(f"System Status: {status}")
        
        # Test recommendations
        test_preferences = {
            'budget_range': (2000, 5000),
            'brand_preference': 'Dell'
        }
        
        recommendations = model_manager.recommend(
            algorithm="content_based",
            top_n=3,
            preferences=test_preferences
        )
        
        print(f"Generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec.get('title', 'Unknown')} - RM {rec.get('price_myr', 0):.2f}")
    
    else:
        print("❌ Failed to load models")
