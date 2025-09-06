"""
User Behavior Analyzer for Enhanced Collaborative Filtering

This module analyzes user behavior data including view history, ratings, 
activity patterns, and preferences to create comprehensive user profiles
for improved collaborative filtering recommendations.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserBehaviorAnalyzer:
    """Analyzes user behavior data to create comprehensive user profiles."""
    
    def __init__(self, db_path: str = "data/user_data.db"):
        """Initialize the behavior analyzer with database connection."""
        self.db_path = db_path
        
    def get_user_behavior_data(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive behavior data for a user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get user profile
                user_profile = self._get_user_profile(conn, user_id)
                
                # Get behavior history
                behavior_data = self._get_behavior_history(conn, user_id)
                
                # Get rating patterns
                rating_patterns = self._get_rating_patterns(conn, user_id)
                
                # Get view patterns
                view_patterns = self._get_view_patterns(conn, user_id)
                
                # Get activity patterns
                activity_patterns = self._get_activity_patterns(conn, user_id)
                
                return {
                    'user_profile': user_profile,
                    'behavior_data': behavior_data,
                    'rating_patterns': rating_patterns,
                    'view_patterns': view_patterns,
                    'activity_patterns': activity_patterns
                }
                
        except Exception as e:
            logger.error(f"Error getting behavior data for user {user_id}: {e}")
            return {}
    
    def _get_user_profile(self, conn, user_id: str) -> Dict[str, Any]:
        """Get basic user profile information."""
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, username, email, created_at, last_active, 
                   preferences, total_views, total_ratings, total_comments
            FROM users WHERE user_id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'user_id': row[0],
                'username': row[1],
                'email': row[2],
                'created_at': row[3],
                'last_active': row[4],
                'preferences': json.loads(row[5]) if row[5] else {},
                'total_views': row[6],
                'total_ratings': row[7],
                'total_comments': row[8]
            }
        return {}
    
    def _get_behavior_history(self, conn, user_id: str) -> List[Dict[str, Any]]:
        """Get complete behavior history for a user."""
        cursor = conn.cursor()
        cursor.execute('''
            SELECT behavior_id, laptop_id, behavior_type, timestamp, data
            FROM user_behavior 
            WHERE user_id = ?
            ORDER BY timestamp DESC
        ''', (user_id,))
        
        behaviors = []
        for row in cursor.fetchall():
            behaviors.append({
                'behavior_id': row[0],
                'laptop_id': row[1],
                'behavior_type': row[2],
                'timestamp': row[3],
                'data': json.loads(row[4]) if row[4] else {}
            })
        
        return behaviors
    
    def _get_rating_patterns(self, conn, user_id: str) -> Dict[str, Any]:
        """Analyze user's rating patterns and preferences."""
        cursor = conn.cursor()
        cursor.execute('''
            SELECT laptop_id, data, timestamp
            FROM user_behavior 
            WHERE user_id = ? AND behavior_type = 'rating'
            ORDER BY timestamp DESC
        ''', (user_id,))
        
        ratings = []
        for row in cursor.fetchall():
            data = json.loads(row[2]) if row[2] else {}
            if 'rating' in data:
                ratings.append({
                    'laptop_id': row[0],
                    'rating': data['rating'],
                    'timestamp': row[1],
                    'comment': data.get('comment', '')
                })
        
        if not ratings:
            return {'ratings': [], 'average_rating': 0, 'rating_distribution': {}}
        
        rating_values = [r['rating'] for r in ratings]
        
        return {
            'ratings': ratings,
            'average_rating': np.mean(rating_values),
            'rating_std': np.std(rating_values),
            'rating_distribution': Counter(rating_values),
            'total_ratings': len(ratings),
            'recent_ratings': ratings[:10]  # Last 10 ratings
        }
    
    def _get_view_patterns(self, conn, user_id: str) -> Dict[str, Any]:
        """Analyze user's view patterns and interests."""
        cursor = conn.cursor()
        cursor.execute('''
            SELECT laptop_id, timestamp, data
            FROM user_behavior 
            WHERE user_id = ? AND behavior_type = 'view'
            ORDER BY timestamp DESC
        ''', (user_id,))
        
        views = []
        for row in cursor.fetchall():
            data = json.loads(row[2]) if row[2] else {}
            views.append({
                'laptop_id': row[0],
                'timestamp': row[1],
                'page': data.get('page', 'unknown'),
                'duration': data.get('duration', 0)
            })
        
        if not views:
            return {'views': [], 'total_views': 0, 'view_frequency': 0}
        
        # Calculate view frequency (views per day)
        if views:
            first_view = datetime.fromisoformat(views[-1]['timestamp'])
            last_view = datetime.fromisoformat(views[0]['timestamp'])
            days = (last_view - first_view).days + 1
            view_frequency = len(views) / max(days, 1)
        else:
            view_frequency = 0
        
        return {
            'views': views,
            'total_views': len(views),
            'view_frequency': view_frequency,
            'recent_views': views[:20],  # Last 20 views
            'unique_laptops_viewed': len(set(v['laptop_id'] for v in views))
        }
    
    def _get_activity_patterns(self, conn, user_id: str) -> Dict[str, Any]:
        """Analyze user's activity patterns and engagement."""
        cursor = conn.cursor()
        cursor.execute('''
            SELECT behavior_type, COUNT(*) as count, 
                   MIN(timestamp) as first_activity,
                   MAX(timestamp) as last_activity
            FROM user_behavior 
            WHERE user_id = ?
            GROUP BY behavior_type
        ''', (user_id,))
        
        activity_counts = {}
        for row in cursor.fetchall():
            activity_counts[row[0]] = {
                'count': row[1],
                'first_activity': row[2],
                'last_activity': row[3]
            }
        
        # Calculate engagement score
        total_activities = sum(data['count'] for data in activity_counts.values())
        engagement_score = min(total_activities / 100, 1.0)  # Normalize to 0-1
        
        return {
            'activity_counts': activity_counts,
            'total_activities': total_activities,
            'engagement_score': engagement_score,
            'activity_types': list(activity_counts.keys())
        }
    
    def create_enhanced_user_profile(self, user_id: str, laptop_data: pd.DataFrame) -> Dict[str, Any]:
        """Create an enhanced user profile with behavior insights."""
        behavior_data = self.get_user_behavior_data(user_id)
        
        if not behavior_data:
            return {}
        
        user_profile = behavior_data['user_profile']
        rating_patterns = behavior_data['rating_patterns']
        view_patterns = behavior_data['view_patterns']
        activity_patterns = behavior_data['activity_patterns']
        
        # Analyze brand preferences from behavior
        brand_preferences = self._analyze_brand_preferences(
            behavior_data['behavior_data'], laptop_data
        )
        
        # Analyze price preferences
        price_preferences = self._analyze_price_preferences(
            behavior_data['behavior_data'], laptop_data
        )
        
        # Analyze feature preferences
        feature_preferences = self._analyze_feature_preferences(
            behavior_data['behavior_data'], laptop_data
        )
        
        # Calculate user similarity weights
        similarity_weights = self._calculate_similarity_weights(
            rating_patterns, view_patterns, activity_patterns
        )
        
        return {
            'user_id': user_id,
            'basic_profile': user_profile,
            'rating_insights': rating_patterns,
            'view_insights': view_patterns,
            'activity_insights': activity_patterns,
            'brand_preferences': brand_preferences,
            'price_preferences': price_preferences,
            'feature_preferences': feature_preferences,
            'similarity_weights': similarity_weights,
            'engagement_level': self._calculate_engagement_level(activity_patterns),
            'preference_confidence': self._calculate_preference_confidence(behavior_data)
        }
    
    def _analyze_brand_preferences(self, behavior_data: List[Dict], laptop_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user's brand preferences from behavior data."""
        brand_scores = defaultdict(list)
        
        for behavior in behavior_data:
            laptop_id = behavior['laptop_id']
            behavior_type = behavior['behavior_type']
            
            # Get laptop brand
            laptop_info = laptop_data[laptop_data['asin'] == laptop_id]
            if laptop_info.empty:
                continue
                
            brand = laptop_info.iloc[0].get('brand', 'Unknown')
            if brand == 'Unknown':
                continue
            
            # Weight different behaviors
            weight = self._get_behavior_weight(behavior_type, behavior.get('data', {}))
            brand_scores[brand].append(weight)
        
        # Calculate average scores
        brand_avg_scores = {}
        for brand, scores in brand_scores.items():
            brand_avg_scores[brand] = np.mean(scores)
        
        # Sort by preference
        sorted_brands = sorted(brand_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'preferred_brands': sorted_brands[:5],  # Top 5 brands
            'brand_scores': brand_avg_scores,
            'total_brands_interacted': len(brand_avg_scores)
        }
    
    def _analyze_price_preferences(self, behavior_data: List[Dict], laptop_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user's price preferences from behavior data."""
        price_interactions = []
        
        for behavior in behavior_data:
            laptop_id = behavior['laptop_id']
            behavior_type = behavior['behavior_type']
            
            # Get laptop price
            laptop_info = laptop_data[laptop_data['asin'] == laptop_id]
            if laptop_info.empty:
                continue
                
            price = laptop_info.iloc[0].get('price_myr', 0)
            if price <= 0:
                continue
            
            weight = self._get_behavior_weight(behavior_type, behavior.get('data', {}))
            price_interactions.append({
                'price': price,
                'weight': weight,
                'behavior_type': behavior_type
            })
        
        if not price_interactions:
            return {'preferred_price_range': None, 'price_sensitivity': 0}
        
        # Calculate price statistics
        prices = [p['price'] for p in price_interactions]
        weights = [p['weight'] for p in price_interactions]
        
        # Weighted average price
        weighted_avg_price = np.average(prices, weights=weights)
        
        # Price range analysis
        min_price = min(prices)
        max_price = max(prices)
        price_std = np.std(prices)
        
        # Price sensitivity (lower std = more consistent price preference)
        price_sensitivity = 1 - min(price_std / weighted_avg_price, 1) if weighted_avg_price > 0 else 0
        
        return {
            'preferred_price_range': (min_price, max_price),
            'weighted_average_price': weighted_avg_price,
            'price_sensitivity': price_sensitivity,
            'price_interactions': len(price_interactions)
        }
    
    def _analyze_feature_preferences(self, behavior_data: List[Dict], laptop_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user's feature preferences from behavior data."""
        feature_scores = defaultdict(list)
        
        for behavior in behavior_data:
            laptop_id = behavior['laptop_id']
            behavior_type = behavior['behavior_type']
            
            # Get laptop features
            laptop_info = laptop_data[laptop_data['asin'] == laptop_id]
            if laptop_info.empty:
                continue
            
            laptop = laptop_info.iloc[0]
            weight = self._get_behavior_weight(behavior_type, behavior.get('data', {}))
            
            # Analyze different features
            features = {
                'ram': laptop.get('ram_gb', 0),
                'storage': laptop.get('storage_gb', 0),
                'screen_size': laptop.get('screen_size', 0),
                'processor_type': laptop.get('processor_type', ''),
                'gpu_type': laptop.get('gpu_type', ''),
                'weight': laptop.get('weight_kg', 0)
            }
            
            for feature, value in features.items():
                if value and value != '' and value != 0:
                    feature_scores[feature].append((value, weight))
        
        # Calculate feature preferences
        feature_preferences = {}
        for feature, scores in feature_scores.items():
            if scores:
                values, weights = zip(*scores)
                weighted_avg = np.average(values, weights=weights)
                feature_preferences[feature] = {
                    'preferred_value': weighted_avg,
                    'interaction_count': len(scores),
                    'value_range': (min(values), max(values))
                }
        
        return feature_preferences
    
    def _get_behavior_weight(self, behavior_type: str, data: Dict) -> float:
        """Get weight for different types of behaviors."""
        weights = {
            'rating': 1.0,
            'comment': 0.8,
            'view': 0.3,
            'like': 0.6,
            'dislike': -0.4
        }
        
        base_weight = weights.get(behavior_type, 0.1)
        
        # Adjust weight based on additional data
        if behavior_type == 'rating' and 'rating' in data:
            rating = data['rating']
            # Higher ratings get higher weights
            base_weight *= (rating / 5.0)
        elif behavior_type == 'view' and 'duration' in data:
            duration = data['duration']
            # Longer views get higher weights
            base_weight *= min(duration / 60, 2.0)  # Cap at 2x for 2+ minutes
        
        return base_weight
    
    def _calculate_similarity_weights(self, rating_patterns: Dict, view_patterns: Dict, 
                                    activity_patterns: Dict) -> Dict[str, float]:
        """Calculate weights for different similarity factors."""
        weights = {}
        
        # Rating-based weight
        if rating_patterns['total_ratings'] > 0:
            weights['rating_similarity'] = min(rating_patterns['total_ratings'] / 10, 1.0)
        else:
            weights['rating_similarity'] = 0.0
        
        # View-based weight
        if view_patterns['total_views'] > 0:
            weights['view_similarity'] = min(view_patterns['total_views'] / 20, 1.0)
        else:
            weights['view_similarity'] = 0.0
        
        # Activity-based weight
        weights['activity_similarity'] = activity_patterns['engagement_score']
        
        # Preference-based weight (if user has strong preferences)
        preference_strength = 0.0
        if rating_patterns['total_ratings'] > 5:
            preference_strength += 0.3
        if view_patterns['total_views'] > 10:
            preference_strength += 0.2
        if activity_patterns['total_activities'] > 20:
            preference_strength += 0.5
        
        weights['preference_similarity'] = min(preference_strength, 1.0)
        
        return weights
    
    def _calculate_engagement_level(self, activity_patterns: Dict) -> str:
        """Calculate user engagement level."""
        total_activities = activity_patterns['total_activities']
        engagement_score = activity_patterns['engagement_score']
        
        if engagement_score >= 0.8:
            return 'high'
        elif engagement_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_preference_confidence(self, behavior_data: Dict) -> float:
        """Calculate confidence in user preferences based on data quality."""
        rating_patterns = behavior_data['rating_patterns']
        view_patterns = behavior_data['view_patterns']
        activity_patterns = behavior_data['activity_patterns']
        
        confidence = 0.0
        
        # Rating confidence
        if rating_patterns['total_ratings'] >= 5:
            confidence += 0.4
        elif rating_patterns['total_ratings'] >= 2:
            confidence += 0.2
        
        # View confidence
        if view_patterns['total_views'] >= 10:
            confidence += 0.3
        elif view_patterns['total_views'] >= 5:
            confidence += 0.15
        
        # Activity confidence
        if activity_patterns['total_activities'] >= 20:
            confidence += 0.3
        elif activity_patterns['total_activities'] >= 10:
            confidence += 0.15
        
        return min(confidence, 1.0)


def create_user_behavior_analyzer(db_path: str = "data/user_data.db") -> UserBehaviorAnalyzer:
    """Create a UserBehaviorAnalyzer instance."""
    return UserBehaviorAnalyzer(db_path)


if __name__ == "__main__":
    # Test the behavior analyzer
    analyzer = create_user_behavior_analyzer()
    
    # Test with a sample user
    user_id = "test_user"
    behavior_data = analyzer.get_user_behavior_data(user_id)
    print(f"Behavior data for {user_id}: {behavior_data}")
