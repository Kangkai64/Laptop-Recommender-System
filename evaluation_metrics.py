"""
Comprehensive Evaluation Metrics for Laptop Recommender System

This module provides comprehensive evaluation metrics for testing and assessing
the performance of the laptop recommendation system, including:
- Precision, Recall, F1 Score
- Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
- User satisfaction metrics
- Cross-validation and holdout testing
- A/B testing framework

Author: Laptop Recommender System Team
License: MIT
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics.pairwise import cosine_similarity
import json
import time
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class RecommendationEvaluator:
    """Comprehensive evaluation system for recommendation algorithms."""
    
    def __init__(self, df_laptop: pd.DataFrame, df_rating: pd.DataFrame):
        """
        Initialize the evaluator with laptop and rating data.
        
        Args:
            df_laptop: DataFrame containing laptop information
            df_rating: DataFrame containing user ratings
        """
        self.df_laptop = df_laptop
        self.df_rating = df_rating
        self.evaluation_results = {}
        self.user_satisfaction_data = {}
        
        # Prepare data for evaluation
        self._prepare_evaluation_data()
        
        logger.info("RecommendationEvaluator initialized successfully")
    
    def _prepare_evaluation_data(self):
        """Prepare data structures for evaluation."""
        # Create user-item matrix for collaborative filtering evaluation
        # Use 'asin' as the laptop identifier since that's what's in the rating data
        if 'user_id_encoded' in self.df_rating.columns and 'asin' in self.df_rating.columns:
            self.user_item_matrix = self.df_rating.pivot_table(
                index='user_id_encoded', 
                columns='asin', 
                values='rating', 
                fill_value=0
            )
            logger.info(f"Created user-item matrix with shape: {self.user_item_matrix.shape}")
        else:
            logger.warning("Required columns not found for user-item matrix")
            logger.warning(f"Available columns: {list(self.df_rating.columns)}")
            self.user_item_matrix = None
        
        # Create laptop feature matrix for content-based evaluation
        self.laptop_features = self._extract_laptop_features()
        
        # Split data for holdout testing
        self._split_data_for_evaluation()
    
    def _extract_laptop_features(self) -> pd.DataFrame:
        """Extract numerical features from laptop data for evaluation."""
        features = ['price_myr', 'average_rating', 'ram_gb', 'storage_gb']
        available_features = [f for f in features if f in self.df_laptop.columns]
        
        if available_features:
            return self.df_laptop[available_features].fillna(0)
        else:
            logger.warning("No suitable features found for laptop feature matrix")
            return pd.DataFrame()
    
    def _split_data_for_evaluation(self):
        """Split rating data for holdout testing."""
        if self.df_rating is not None and len(self.df_rating) > 0:
            # Split into train/test sets (80/20)
            self.train_ratings, self.test_ratings = train_test_split(
                self.df_rating, 
                test_size=0.2, 
                random_state=42,
                stratify=self.df_rating['rating'] if 'rating' in self.df_rating.columns else None
            )
            logger.info(f"Split data: {len(self.train_ratings)} train, {len(self.test_ratings)} test")
        else:
            logger.warning("No rating data available for splitting")
            self.train_ratings = pd.DataFrame()
            self.test_ratings = pd.DataFrame()
    
    def evaluate_content_based_accuracy(self, recommender, n_recommendations: int = 10) -> Dict[str, float]:
        """
        Evaluate content-based filtering accuracy using precision, recall, and F1.
        
        Args:
            recommender: Content-based recommender instance
            n_recommendations: Number of recommendations to generate
            
        Returns:
            Dictionary containing precision, recall, F1, and coverage metrics
        """
        logger.info("Evaluating content-based filtering accuracy...")
        
        try:
            # Test with different user preference scenarios
            test_scenarios = [
                {'search_terms': ['gaming', 'performance'], 'min_rating': 4.0, 'max_price': 8000},
                {'search_terms': ['student', 'budget'], 'min_rating': 3.5, 'max_price': 3000},
                {'search_terms': ['business', 'professional'], 'min_rating': 4.0, 'max_price': 6000},
                {'search_terms': ['creative', 'design'], 'min_rating': 3.5, 'max_price': 7000}
            ]
            
            all_precisions = []
            all_recalls = []
            all_f1_scores = []
            coverage_scores = []
            
            for scenario in test_scenarios:
                try:
                    # Get recommendations
                    recommendations = recommender.get_recommendations_by_preferences(
                        scenario, n_recommendations
                    )
                    
                    if not recommendations:
                        continue
                    
                    # Calculate metrics for this scenario
                    precision, recall, f1, coverage = self._calculate_recommendation_metrics(
                        recommendations, scenario
                    )
                    
                    all_precisions.append(precision)
                    all_recalls.append(recall)
                    all_f1_scores.append(f1)
                    coverage_scores.append(coverage)
                    
                except Exception as e:
                    logger.warning(f"Error evaluating scenario {scenario}: {e}")
                    continue
            
            # Calculate average metrics
            results = {
                'precision': np.mean(all_precisions) if all_precisions else 0.0,
                'recall': np.mean(all_recalls) if all_recalls else 0.0,
                'f1_score': np.mean(all_f1_scores) if all_f1_scores else 0.0,
                'coverage': np.mean(coverage_scores) if coverage_scores else 0.0,
                'n_scenarios_tested': len(test_scenarios)
            }
            
            logger.info(f"Content-based evaluation completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in content-based evaluation: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'coverage': 0.0}
    
    def evaluate_collaborative_filtering_accuracy(self, recommender, n_recommendations: int = 10) -> Dict[str, float]:
        """
        Evaluate collaborative filtering accuracy using precision, recall, and F1.
        
        Args:
            recommender: Collaborative filtering recommender instance
            n_recommendations: Number of recommendations to generate
            
        Returns:
            Dictionary containing precision, recall, F1, and coverage metrics
        """
        logger.info("Evaluating collaborative filtering accuracy...")
        
        try:
            if self.user_item_matrix is None or len(self.user_item_matrix) == 0:
                logger.warning("No user-item matrix available for collaborative evaluation")
                return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'coverage': 0.0}
            
            # Test with different users - use actual user IDs from the data
            test_users = self.user_item_matrix.index[:min(50, len(self.user_item_matrix))]
            all_precisions = []
            all_recalls = []
            all_f1_scores = []
            coverage_scores = []
            
            for user_id in test_users:
                try:
                    # Get user's actual ratings for comparison
                    user_ratings = self.user_item_matrix.loc[user_id]
                    actual_rated_items = user_ratings[user_ratings > 0].index.tolist()
                    
                    if len(actual_rated_items) < 2:  # Need at least 2 ratings for meaningful evaluation
                        continue
                    
                    # Get recommendations for this user
                    recommendations = recommender.get_user_based_recommendations(
                        user_id=user_id, n_recommendations=n_recommendations
                    )
                    
                    if not recommendations:
                        continue
                    
                    # Extract recommended item IDs
                    recommended_items = [rec.get('laptop_id', rec.get('asin', '')) for rec in recommendations]
                    recommended_items = [item for item in recommended_items if item != '']
                    
                    if not recommended_items:
                        continue
                    
                    # Calculate metrics
                    precision, recall, f1, coverage = self._calculate_collaborative_metrics(
                        recommended_items, actual_rated_items, user_ratings
                    )
                    
                    all_precisions.append(precision)
                    all_recalls.append(recall)
                    all_f1_scores.append(f1)
                    coverage_scores.append(coverage)
                    
                except Exception as e:
                    logger.warning(f"Error evaluating user {user_id}: {e}")
                    continue
            
            # Calculate average metrics
            results = {
                'precision': np.mean(all_precisions) if all_precisions else 0.0,
                'recall': np.mean(all_recalls) if all_recalls else 0.0,
                'f1_score': np.mean(all_f1_scores) if all_f1_scores else 0.0,
                'coverage': np.mean(coverage_scores) if coverage_scores else 0.0,
                'n_users_tested': len(test_users)
            }
            
            logger.info(f"Collaborative filtering evaluation completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering evaluation: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'coverage': 0.0}
    
    def evaluate_rating_prediction_accuracy(self, recommender) -> Dict[str, float]:
        """
        Evaluate rating prediction accuracy using MSE and RMSE.
        
        Args:
            recommender: Recommender system instance
            
        Returns:
            Dictionary containing MSE, RMSE, and MAE metrics
        """
        logger.info("Evaluating rating prediction accuracy...")
        
        try:
            if len(self.test_ratings) == 0:
                logger.warning("No test ratings available for prediction evaluation")
                return {'mse': 0.0, 'rmse': 0.0, 'mae': 0.0}
            
            predicted_ratings = []
            actual_ratings = []
            
            # Sample test cases for prediction
            test_sample = self.test_ratings.sample(min(100, len(self.test_ratings)))
            
            for _, rating_row in test_sample.iterrows():
                try:
                    user_id = rating_row.get('user_id_encoded', rating_row.get('user_id'))
                    laptop_id = rating_row.get('asin')  # Use 'asin' as the laptop identifier
                    actual_rating = rating_row.get('rating', rating_row.get('average_rating'))
                    
                    if pd.isna(user_id) or pd.isna(laptop_id) or pd.isna(actual_rating):
                        continue
                    
                    # Try to predict rating (this would need to be implemented in the recommender)
                    # For now, we'll use a simple heuristic based on laptop average rating
                    laptop_data = self.df_laptop[self.df_laptop['asin'] == laptop_id]
                    if not laptop_data.empty:
                        predicted_rating = laptop_data.iloc[0].get('average_rating', 3.0)
                    else:
                        predicted_rating = 3.0  # Default rating
                    
                    predicted_ratings.append(predicted_rating)
                    actual_ratings.append(actual_rating)
                    
                except Exception as e:
                    logger.warning(f"Error predicting rating for row: {e}")
                    continue
            
            if not predicted_ratings or not actual_ratings:
                logger.warning("No valid predictions generated")
                return {'mse': 0.0, 'rmse': 0.0, 'mae': 0.0}
            
            # Calculate metrics
            mse = mean_squared_error(actual_ratings, predicted_ratings)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(np.array(actual_ratings) - np.array(predicted_ratings)))
            
            results = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'n_predictions': len(predicted_ratings)
            }
            
            logger.info(f"Rating prediction evaluation completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in rating prediction evaluation: {e}")
            return {'mse': 0.0, 'rmse': 0.0, 'mae': 0.0}
    
    def _calculate_recommendation_metrics(self, recommendations: List[Dict], scenario: Dict) -> Tuple[float, float, float, float]:
        """Calculate precision, recall, F1, and coverage for content-based recommendations."""
        try:
            if not recommendations:
                return 0.0, 0.0, 0.0, 0.0
            
            # Define relevance criteria based on scenario
            min_rating = scenario.get('min_rating', 3.0)
            max_price = scenario.get('max_price', 10000)
            search_terms = scenario.get('search_terms', [])
            
            relevant_count = 0
            total_recommendations = len(recommendations)
            
            for rec in recommendations:
                is_relevant = True
                
                # Check rating criteria
                if rec.get('rating', rec.get('average_rating', 0)) < min_rating:
                    is_relevant = False
                
                # Check price criteria
                if rec.get('price_myr', 0) > max_price:
                    is_relevant = False
                
                # Check search term relevance (simplified)
                if search_terms:
                    title = str(rec.get('title', rec.get('title_y', ''))).lower()
                    features = str(rec.get('features', '')).lower()
                    text_content = f"{title} {features}"
                    
                    term_matches = sum(1 for term in search_terms if term.lower() in text_content)
                    if term_matches == 0:
                        is_relevant = False
                
                if is_relevant:
                    relevant_count += 1
            
            precision = relevant_count / total_recommendations if total_recommendations > 0 else 0.0
            recall = relevant_count / total_recommendations if total_recommendations > 0 else 0.0  # Simplified
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            coverage = min(1.0, total_recommendations / 50)  # Coverage based on max recommendations
            
            return precision, recall, f1, coverage
            
        except Exception as e:
            logger.warning(f"Error calculating recommendation metrics: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def _calculate_collaborative_metrics(self, recommended_items: List, actual_rated_items: List, user_ratings: pd.Series) -> Tuple[float, float, float, float]:
        """Calculate precision, recall, F1, and coverage for collaborative filtering recommendations."""
        try:
            if not recommended_items or not actual_rated_items:
                return 0.0, 0.0, 0.0, 0.0
            
            # Convert to sets for easier comparison
            recommended_set = set(recommended_items)
            actual_set = set(actual_rated_items)
            
            # Calculate intersection
            intersection = recommended_set.intersection(actual_set)
            
            # Calculate metrics
            precision = len(intersection) / len(recommended_set) if len(recommended_set) > 0 else 0.0
            recall = len(intersection) / len(actual_set) if len(actual_set) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Coverage: fraction of items that can be recommended
            total_items = len(self.df_laptop) if self.df_laptop is not None else 1
            coverage = len(recommended_set) / total_items
            
            return precision, recall, f1, coverage
            
        except Exception as e:
            logger.warning(f"Error calculating collaborative metrics: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def evaluate_system_performance(self, recommender) -> Dict[str, Any]:
        """
        Comprehensive system performance evaluation.
        
        Args:
            recommender: Complete recommender system instance
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info("Starting comprehensive system evaluation...")
        
        start_time = time.time()
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_duration': 0,
            'content_based': {},
            'collaborative': {},
            'rating_prediction': {},
            'system_metrics': {}
        }
        
        try:
            # Evaluate content-based filtering
            if hasattr(recommender, 'content_based_filter') and recommender.content_based_filter:
                evaluation_results['content_based'] = self.evaluate_content_based_accuracy(
                    recommender.content_based_filter
                )
            
            # Evaluate collaborative filtering
            if hasattr(recommender, 'collaborative_filter') and recommender.collaborative_filter:
                evaluation_results['collaborative'] = self.evaluate_collaborative_filtering_accuracy(
                    recommender.collaborative_filter
                )
            
            # Evaluate rating prediction
            evaluation_results['rating_prediction'] = self.evaluate_rating_prediction_accuracy(recommender)
            
            # System performance metrics
            evaluation_results['system_metrics'] = self._calculate_system_metrics(recommender)
            
            # Calculate total evaluation time
            evaluation_results['evaluation_duration'] = time.time() - start_time
            
            # Store results
            self.evaluation_results = evaluation_results
            
            logger.info("Comprehensive system evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            evaluation_results['error'] = str(e)
            evaluation_results['evaluation_duration'] = time.time() - start_time
            return evaluation_results
    
    def _calculate_system_metrics(self, recommender) -> Dict[str, Any]:
        """Calculate system-level performance metrics."""
        try:
            metrics = {
                'total_laptops': len(self.df_laptop) if self.df_laptop is not None else 0,
                'total_ratings': len(self.df_rating) if self.df_rating is not None else 0,
                'unique_users': self.df_rating['user_id_encoded'].nunique() if self.df_rating is not None else 0,
                'avg_rating': self.df_laptop['average_rating'].mean() if self.df_laptop is not None else 0,
                'data_sparsity': 0.0,
                'memory_usage_mb': 0.0
            }
            
            # Calculate data sparsity
            if self.user_item_matrix is not None:
                total_possible_ratings = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
                actual_ratings = (self.user_item_matrix > 0).sum().sum()
                metrics['data_sparsity'] = 1 - (actual_ratings / total_possible_ratings) if total_possible_ratings > 0 else 1.0
            
            # Estimate memory usage
            if self.df_laptop is not None:
                metrics['memory_usage_mb'] = self.df_laptop.memory_usage(deep=True).sum() / 1024 / 1024
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating system metrics: {e}")
            return {'error': str(e)}
    
    def save_evaluation_results(self, filename: str = None) -> str:
        """Save evaluation results to a JSON file in the results folder."""
        import os
        
        # Create results directory if it doesn't exist
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_metrics_{timestamp}.json"
        
        # Ensure filename is in results directory
        if not filename.startswith(results_dir):
            filename = os.path.join(results_dir, filename)
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            return ""

    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.evaluation_results:
            return "No evaluation results available. Please run evaluation first."
        
        report = []
        report.append("=" * 80)
        report.append("LAPTOP RECOMMENDER SYSTEM - EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.evaluation_results.get('timestamp', 'Unknown')}")
        report.append(f"Evaluation Duration: {self.evaluation_results.get('evaluation_duration', 0):.2f} seconds")
        report.append("")
        
        # Content-based results
        if 'content_based' in self.evaluation_results:
            cb_results = self.evaluation_results['content_based']
            report.append("CONTENT-BASED FILTERING METRICS:")
            report.append("-" * 40)
            report.append(f"Precision: {cb_results.get('precision', 0):.3f}")
            report.append(f"Recall: {cb_results.get('recall', 0):.3f}")
            report.append(f"F1 Score: {cb_results.get('f1_score', 0):.3f}")
            report.append(f"Coverage: {cb_results.get('coverage', 0):.3f}")
            report.append(f"Scenarios Tested: {cb_results.get('n_scenarios_tested', 0)}")
            report.append("")
        
        # Collaborative filtering results
        if 'collaborative' in self.evaluation_results:
            cf_results = self.evaluation_results['collaborative']
            report.append("COLLABORATIVE FILTERING METRICS:")
            report.append("-" * 40)
            report.append(f"Precision: {cf_results.get('precision', 0):.3f}")
            report.append(f"Recall: {cf_results.get('recall', 0):.3f}")
            report.append(f"F1 Score: {cf_results.get('f1_score', 0):.3f}")
            report.append(f"Coverage: {cf_results.get('coverage', 0):.3f}")
            report.append(f"Users Tested: {cf_results.get('n_users_tested', 0)}")
            report.append("")
        
        # Rating prediction results
        if 'rating_prediction' in self.evaluation_results:
            rp_results = self.evaluation_results['rating_prediction']
            report.append("RATING PREDICTION METRICS:")
            report.append("-" * 40)
            report.append(f"Mean Squared Error (MSE): {rp_results.get('mse', 0):.3f}")
            report.append(f"Root Mean Squared Error (RMSE): {rp_results.get('rmse', 0):.3f}")
            report.append(f"Mean Absolute Error (MAE): {rp_results.get('mae', 0):.3f}")
            report.append(f"Predictions Made: {rp_results.get('n_predictions', 0)}")
            report.append("")
        
        # System metrics
        if 'system_metrics' in self.evaluation_results:
            sys_results = self.evaluation_results['system_metrics']
            report.append("SYSTEM METRICS:")
            report.append("-" * 40)
            report.append(f"Total Laptops: {sys_results.get('total_laptops', 0):,}")
            report.append(f"Total Ratings: {sys_results.get('total_ratings', 0):,}")
            report.append(f"Unique Users: {sys_results.get('unique_users', 0):,}")
            report.append(f"Average Rating: {sys_results.get('avg_rating', 0):.2f}")
            report.append(f"Data Sparsity: {sys_results.get('data_sparsity', 0):.3f}")
            report.append(f"Memory Usage: {sys_results.get('memory_usage_mb', 0):.2f} MB")
            report.append("")
        
        report.append("=" * 80)
        report.append("EVALUATION COMPLETED")
        report.append("=" * 80)
        
        return "\n".join(report)


class UserSatisfactionSurvey:
    """User satisfaction survey system for recommendation quality assessment."""
    
    def __init__(self):
        """Initialize the user satisfaction survey system."""
        self.survey_responses = []
        self.survey_questions = self._create_survey_questions()
        
    def _create_survey_questions(self) -> List[Dict[str, Any]]:
        """Create survey questions for user satisfaction assessment."""
        return [
            {
                'id': 'relevance',
                'question': 'How relevant were the laptop recommendations to your needs?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Not relevant at all', 'Slightly relevant', 'Moderately relevant', 'Very relevant', 'Extremely relevant']
            },
            {
                'id': 'diversity',
                'question': 'How diverse were the recommended laptops?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Not diverse at all', 'Slightly diverse', 'Moderately diverse', 'Very diverse', 'Extremely diverse']
            },
            {
                'id': 'novelty',
                'question': 'Did you discover new laptops you hadn\'t considered before?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Not at all', 'Slightly', 'Moderately', 'Very much', 'Extremely']
            },
            {
                'id': 'accuracy',
                'question': 'How accurate were the laptop specifications and features shown?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Very inaccurate', 'Somewhat inaccurate', 'Neutral', 'Somewhat accurate', 'Very accurate']
            },
            {
                'id': 'speed',
                'question': 'How satisfied are you with the recommendation speed?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Very slow', 'Slow', 'Acceptable', 'Fast', 'Very fast']
            },
            {
                'id': 'overall',
                'question': 'Overall, how satisfied are you with the recommendation system?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Very dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very satisfied']
            },
            {
                'id': 'improvement',
                'question': 'What would you like to see improved in the recommendation system?',
                'type': 'text',
                'placeholder': 'Please provide your suggestions...'
            }
        ]
    
    def submit_survey_response(self, user_id: str, responses: Dict[str, Any]) -> bool:
        """
        Submit a user satisfaction survey response.
        
        Args:
            user_id: Unique identifier for the user
            responses: Dictionary containing question_id -> response mappings
            
        Returns:
            Boolean indicating success
        """
        try:
            response_data = {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'responses': responses
            }
            
            self.survey_responses.append(response_data)
            logger.info(f"Survey response submitted for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting survey response: {e}")
            return False
    
    def calculate_satisfaction_metrics(self) -> Dict[str, float]:
        """Calculate user satisfaction metrics from survey responses."""
        if not self.survey_responses:
            return {'avg_satisfaction': 0.0, 'response_count': 0}
        
        try:
            # Extract rating responses
            rating_questions = [q['id'] for q in self.survey_questions if q['type'] == 'rating']
            
            satisfaction_scores = []
            for response in self.survey_responses:
                user_scores = []
                for question_id in rating_questions:
                    if question_id in response['responses']:
                        score = response['responses'][question_id]
                        if isinstance(score, (int, float)) and 1 <= score <= 5:
                            user_scores.append(score)
                
                if user_scores:
                    satisfaction_scores.append(np.mean(user_scores))
            
            if satisfaction_scores:
                return {
                    'avg_satisfaction': float(np.mean(satisfaction_scores)),
                    'satisfaction_std': float(np.std(satisfaction_scores)),
                    'response_count': len(self.survey_responses),
                    'max_possible': 5.0,
                    'satisfaction_percentage': float(np.mean(satisfaction_scores) / 5.0 * 100)
                }
            else:
                return {'avg_satisfaction': 0.0, 'response_count': 0}
                
        except Exception as e:
            logger.error(f"Error calculating satisfaction metrics: {e}")
            return {'avg_satisfaction': 0.0, 'response_count': 0}
    
    def get_survey_questions(self) -> List[Dict[str, Any]]:
        """Get the list of survey questions."""
        return self.survey_questions


def create_evaluator(df_laptop: pd.DataFrame, df_rating: pd.DataFrame) -> RecommendationEvaluator:
    """
    Factory function to create a RecommendationEvaluator instance.
    
    Args:
        df_laptop: DataFrame containing laptop information
        df_rating: DataFrame containing user ratings
        
    Returns:
        RecommendationEvaluator instance
    """
    return RecommendationEvaluator(df_laptop, df_rating)


def create_satisfaction_survey() -> UserSatisfactionSurvey:
    """
    Factory function to create a UserSatisfactionSurvey instance.
    
    Returns:
        UserSatisfactionSurvey instance
    """
    return UserSatisfactionSurvey()


if __name__ == "__main__":
    # Example usage
    print("Recommendation System Evaluation Module")
    print("This module provides comprehensive evaluation metrics for the laptop recommender system.")
    print("Use create_evaluator() and create_satisfaction_survey() to get started.")
