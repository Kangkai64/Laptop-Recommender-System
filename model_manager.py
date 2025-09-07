"""
Model Management Utilities for Laptop Recommendation System

This module provides utilities for managing, saving, loading, and validating
recommendation models using joblib for optimized serialization.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json

from joblib import dump, load
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Utility class for managing recommendation models."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory to store model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different model types
        self.content_based_dir = self.models_dir / "content_based"
        self.collaborative_dir = self.models_dir / "collaborative"
        self.hybrid_dir = self.models_dir / "hybrid"
        
        for dir_path in [self.content_based_dir, self.collaborative_dir, self.hybrid_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_model(self, model: Any, model_type: str, model_name: str, 
                   metadata: Optional[Dict] = None, version: Optional[str] = None) -> str:
        """
        Save a model with metadata and versioning.
        
        Args:
            model: The model object to save
            model_type: Type of model ('content_based', 'collaborative', 'hybrid')
            model_name: Name for the model
            metadata: Additional metadata to save with the model
            version: Version string (defaults to timestamp)
            
        Returns:
            str: Path to the saved model file
        """
        try:
            # Determine save directory
            if model_type == 'content_based':
                save_dir = self.content_based_dir
            elif model_type == 'collaborative':
                save_dir = self.collaborative_dir
            elif model_type == 'hybrid':
                save_dir = self.hybrid_dir
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Generate version if not provided
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            filename = f"{model_name}_v{version}.pkl"
            filepath = save_dir / filename
            
            # Prepare model data
            model_data = {
                'model': model,
                'model_type': model_type,
                'model_name': model_name,
                'version': version,
                'created_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            # Save model using joblib
            dump(model_data, filepath, compress=3)
            
            # Save metadata separately for easy access
            metadata_file = save_dir / f"{model_name}_v{version}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    'model_type': model_type,
                    'model_name': model_name,
                    'version': version,
                    'created_at': model_data['created_at'],
                    'metadata': metadata or {}
                }, f, indent=2)
            
            logger.info(f"Model saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load a model from file.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Dict containing the model and metadata
        """
        try:
            model_data = load(filepath)
            
            # Validate model data structure
            required_keys = ['model', 'model_type', 'model_name', 'version', 'created_at']
            for key in required_keys:
                if key not in model_data:
                    raise ValueError(f"Missing required key '{key}' in saved model")
            
            logger.info(f"Model loaded: {filepath}")
            logger.info(f"Model type: {model_data['model_type']}")
            logger.info(f"Model name: {model_data['model_name']}")
            logger.info(f"Version: {model_data['version']}")
            logger.info(f"Created at: {model_data['created_at']}")
            
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Args:
            model_type: Filter by model type ('content_based', 'collaborative', 'hybrid')
            
        Returns:
            List of model information dictionaries
        """
        models = []
        
        # Determine which directories to search
        if model_type:
            if model_type == 'content_based':
                search_dirs = [self.content_based_dir]
            elif model_type == 'collaborative':
                search_dirs = [self.collaborative_dir]
            elif model_type == 'hybrid':
                search_dirs = [self.hybrid_dir]
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            search_dirs = [self.content_based_dir, self.collaborative_dir, self.hybrid_dir]
        
        # Search for model files
        for search_dir in search_dirs:
            for file_path in search_dir.glob("*.pkl"):
                try:
                    # Load model metadata
                    model_data = load(file_path)
                    
                    # Get file stats
                    file_stats = file_path.stat()
                    
                    models.append({
                        'filepath': str(file_path),
                        'model_type': model_data.get('model_type', 'unknown'),
                        'model_name': model_data.get('model_name', 'unknown'),
                        'version': model_data.get('version', 'unknown'),
                        'created_at': model_data.get('created_at', 'unknown'),
                        'file_size_mb': file_stats.st_size / (1024 * 1024),
                        'modified_at': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    })
                    
                except Exception as e:
                    logger.warning(f"Could not load metadata for {file_path}: {e}")
                    continue
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x['created_at'], reverse=True)
        
        return models
    
    def get_latest_model(self, model_type: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a specific model.
        
        Args:
            model_type: Type of model ('content_based', 'collaborative', 'hybrid')
            model_name: Name of the model
            
        Returns:
            Model data dictionary or None if not found
        """
        models = self.list_models(model_type)
        
        # Filter by model name and get the latest
        matching_models = [m for m in models if m['model_name'] == model_name]
        
        if matching_models:
            return self.load_model(matching_models[0]['filepath'])
        
        return None
    
    def delete_model(self, filepath: str) -> bool:
        """
        Delete a model file and its metadata.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(filepath)
            
            # Delete the model file
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted model file: {filepath}")
            
            # Delete metadata file if it exists
            metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()
                logger.info(f"Deleted metadata file: {metadata_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            return False
    
    def validate_model(self, model: Any, model_type: str) -> Dict[str, Any]:
        """
        Validate a model and return validation results.
        
        Args:
            model: The model object to validate
            model_type: Type of model ('content_based', 'collaborative', 'hybrid')
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'model_info': {}
        }
        
        try:
            if model_type == 'content_based':
                validation_results = self._validate_content_based_model(model)
            elif model_type == 'collaborative':
                validation_results = self._validate_collaborative_model(model)
            elif model_type == 'hybrid':
                validation_results = self._validate_hybrid_model(model)
            else:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Unknown model type: {model_type}")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def _validate_content_based_model(self, model: Any) -> Dict[str, Any]:
        """Validate content-based filtering model."""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'model_info': {}
        }
        
        try:
            # Check required attributes
            required_attrs = ['feature_matrix', 'similarity_matrix', 'feature_names', 'config']
            for attr in required_attrs:
                if not hasattr(model, attr):
                    results['errors'].append(f"Missing required attribute: {attr}")
                    results['is_valid'] = False
                else:
                    value = getattr(model, attr)
                    if value is None:
                        results['warnings'].append(f"Attribute {attr} is None")
            
            # Check feature matrix
            if hasattr(model, 'feature_matrix') and model.feature_matrix is not None:
                results['model_info']['feature_matrix_shape'] = model.feature_matrix.shape
                if model.feature_matrix.shape[0] == 0:
                    results['errors'].append("Feature matrix is empty")
                    results['is_valid'] = False
            
            # Check similarity matrix
            if hasattr(model, 'similarity_matrix') and model.similarity_matrix is not None:
                results['model_info']['similarity_matrix_shape'] = model.similarity_matrix.shape
                if model.similarity_matrix.shape[0] == 0:
                    results['errors'].append("Similarity matrix is empty")
                    results['is_valid'] = False
            
            # Check feature names
            if hasattr(model, 'feature_names') and model.feature_names is not None:
                results['model_info']['num_features'] = len(model.feature_names)
                if len(model.feature_names) == 0:
                    results['warnings'].append("No feature names available")
            
        except Exception as e:
            results['errors'].append(f"Validation error: {str(e)}")
            results['is_valid'] = False
        
        return results
    
    def _validate_collaborative_model(self, model: Any) -> Dict[str, Any]:
        """Validate collaborative filtering model."""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'model_info': {}
        }
        
        try:
            # Check if model has the required methods
            required_methods = ['is_model_trained', 'get_model_info']
            for method in required_methods:
                if not hasattr(model, method):
                    results['errors'].append(f"Missing required method: {method}")
                    results['is_valid'] = False
            
            # Check if model is trained
            if hasattr(model, 'is_model_trained'):
                if not model.is_model_trained():
                    results['warnings'].append("Model appears to be untrained")
            
            # Get model info
            if hasattr(model, 'get_model_info'):
                model_info = model.get_model_info()
                results['model_info'] = model_info
                
                if not model_info.get('is_trained', False):
                    results['warnings'].append("Model is not trained")
            
        except Exception as e:
            results['errors'].append(f"Validation error: {str(e)}")
            results['is_valid'] = False
        
        return results
    
    def _validate_hybrid_model(self, model: Any) -> Dict[str, Any]:
        """Validate hybrid model (combination of content-based and collaborative)."""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'model_info': {}
        }
        
        try:
            # Check if model has both content-based and collaborative components
            if hasattr(model, 'content_based_model'):
                content_results = self._validate_content_based_model(model.content_based_model)
                results['model_info']['content_based'] = content_results['model_info']
                if not content_results['is_valid']:
                    results['errors'].extend([f"Content-based: {e}" for e in content_results['errors']])
                    results['is_valid'] = False
            
            if hasattr(model, 'collaborative_model'):
                collab_results = self._validate_collaborative_model(model.collaborative_model)
                results['model_info']['collaborative'] = collab_results['model_info']
                if not collab_results['is_valid']:
                    results['errors'].extend([f"Collaborative: {e}" for e in collab_results['errors']])
                    results['is_valid'] = False
            
        except Exception as e:
            results['errors'].append(f"Validation error: {str(e)}")
            results['is_valid'] = False
        
        return results


def create_model_manager(models_dir: str = "models") -> ModelManager:
    """
    Factory function to create a ModelManager instance.
    
    Args:
        models_dir: Directory to store model files
        
    Returns:
        ModelManager: Configured model manager instance
    """
    return ModelManager(models_dir)


if __name__ == "__main__":
    print("Model Manager Module")
    print("=" * 40)
    print("This module provides utilities for managing recommendation models.")
    print("Import and use the ModelManager class in your code.")
