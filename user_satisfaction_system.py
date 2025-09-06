"""
User Satisfaction System for Laptop Recommender System

This module provides a comprehensive user satisfaction tracking and analysis system
that includes:
- Interactive satisfaction surveys
- Real-time feedback collection
- Satisfaction metrics calculation
- User experience analytics
- Feedback analysis and reporting

Author: Laptop Recommender System Team
License: MIT
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SatisfactionResponse:
    """Data class for storing user satisfaction responses."""
    user_id: str
    session_id: str
    timestamp: str
    question_id: str
    response_value: Any
    response_type: str  # 'rating', 'text'
    context: Dict[str, Any]  # Additional context like laptop_id, recommendation_method


@dataclass
class SatisfactionMetrics:
    """Data class for storing calculated satisfaction metrics."""
    overall_satisfaction: float
    satisfaction_std: float
    response_count: int
    satisfaction_percentage: float
    category_scores: Dict[str, float]
    trend_data: List[Dict[str, Any]]
    recommendations: List[str]


class UserSatisfactionSystem:
    """Comprehensive user satisfaction tracking and analysis system."""
    
    def __init__(self, db_path: str = "satisfaction_data.db"):
        """
        Initialize the user satisfaction system.
        
        Args:
            db_path: Path to SQLite database for storing satisfaction data
        """
        self.db_path = db_path
        self.survey_questions = self._create_survey_questions()
        self._initialize_database()
        
        logger.info("UserSatisfactionSystem initialized successfully")
    
    def _initialize_database(self):
        """Initialize SQLite database for storing satisfaction data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create satisfaction_responses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS satisfaction_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    response_value TEXT NOT NULL,
                    response_type TEXT NOT NULL,
                    context TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create satisfaction_sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS satisfaction_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    recommendation_method TEXT,
                    laptops_viewed TEXT,
                    recommendations_received TEXT,
                    session_completed BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create satisfaction_metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS satisfaction_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calculation_date TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    category TEXT,
                    period TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _create_survey_questions(self) -> List[Dict[str, Any]]:
        """Create comprehensive survey questions for user satisfaction assessment."""
        return [
            {
                'id': 'overall_satisfaction',
                'question': 'How satisfied are you with the laptop recommendations you received?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'],
                'category': 'overall',
                'weight': 1.0
            },
            {
                'id': 'relevance',
                'question': 'How relevant were the recommended laptops to your needs?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Not relevant at all', 'Slightly relevant', 'Moderately relevant', 'Very relevant', 'Extremely relevant'],
                'category': 'quality',
                'weight': 0.9
            },
            {
                'id': 'diversity',
                'question': 'How diverse were the laptop recommendations?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Not diverse at all', 'Slightly diverse', 'Moderately diverse', 'Very diverse', 'Extremely diverse'],
                'category': 'quality',
                'weight': 0.8
            },
            {
                'id': 'novelty',
                'question': 'Did you discover new laptops you hadn\'t considered before?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Not at all', 'Slightly', 'Moderately', 'Very much', 'Extremely'],
                'category': 'discovery',
                'weight': 0.7
            },
            {
                'id': 'accuracy',
                'question': 'How accurate were the laptop specifications and features shown?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Very inaccurate', 'Somewhat inaccurate', 'Neutral', 'Somewhat accurate', 'Very accurate'],
                'category': 'quality',
                'weight': 0.9
            },
            {
                'id': 'speed',
                'question': 'How satisfied are you with the recommendation speed?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Very slow', 'Slow', 'Acceptable', 'Fast', 'Very fast'],
                'category': 'performance',
                'weight': 0.6
            },
            {
                'id': 'ease_of_use',
                'question': 'How easy was it to use the recommendation system?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Very difficult', 'Difficult', 'Neutral', 'Easy', 'Very easy'],
                'category': 'usability',
                'weight': 0.8
            },
            {
                'id': 'trust',
                'question': 'How much do you trust the recommendations provided?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['No trust at all', 'Little trust', 'Moderate trust', 'High trust', 'Complete trust'],
                'category': 'trust',
                'weight': 0.9
            },
            {
                'id': 'value',
                'question': 'How valuable were the recommendations for your laptop search?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Not valuable at all', 'Slightly valuable', 'Moderately valuable', 'Very valuable', 'Extremely valuable'],
                'category': 'value',
                'weight': 0.9
            },
            {
                'id': 'would_recommend',
                'question': 'Would you recommend this system to others?',
                'type': 'rating',
                'scale': (1, 5),
                'labels': ['Definitely not', 'Probably not', 'Neutral', 'Probably yes', 'Definitely yes'],
                'category': 'advocacy',
                'weight': 1.0
            },
            {
                'id': 'improvement_suggestions',
                'question': 'What would you like to see improved in the recommendation system?',
                'type': 'text',
                'placeholder': 'Please provide your suggestions for improvement...',
                'category': 'feedback',
                'weight': 0.5
            },
            {
                'id': 'missing_features',
                'question': 'What features are missing that would make the system more useful?',
                'type': 'text',
                'placeholder': 'Please describe any missing features...',
                'category': 'feedback',
                'weight': 0.5
            }
        ]
    
    def start_satisfaction_session(self, user_id: str, session_id: str = None, 
                                 recommendation_method: str = None) -> str:
        """
        Start a new satisfaction tracking session.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Optional session ID (generated if not provided)
            recommendation_method: Method used for recommendations
            
        Returns:
            Session ID for tracking
        """
        try:
            if not session_id:
                session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO satisfaction_sessions 
                (session_id, user_id, start_time, recommendation_method)
                VALUES (?, ?, ?, ?)
            ''', (session_id, user_id, datetime.now().isoformat(), recommendation_method))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Satisfaction session started: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting satisfaction session: {e}")
            return ""
    
    def submit_satisfaction_response(self, session_id: str, question_id: str, 
                                   response_value: Any, context: Dict[str, Any] = None) -> bool:
        """
        Submit a satisfaction response for a specific question.
        
        Args:
            session_id: Session ID for tracking
            question_id: ID of the question being answered
            response_value: User's response
            context: Additional context information
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get session information
            session_info = self._get_session_info(session_id)
            if not session_info:
                logger.warning(f"Session {session_id} not found")
                return False
            
            # Get question information
            question = next((q for q in self.survey_questions if q['id'] == question_id), None)
            if not question:
                logger.warning(f"Question {question_id} not found")
                return False
            
            # Prepare response data
            response = SatisfactionResponse(
                user_id=session_info['user_id'],
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                question_id=question_id,
                response_value=str(response_value),
                response_type=question['type'],
                context=context or {}
            )
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO satisfaction_responses 
                (user_id, session_id, timestamp, question_id, response_value, response_type, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                response.user_id,
                response.session_id,
                response.timestamp,
                response.question_id,
                response.response_value,
                response.response_type,
                json.dumps(response.context)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Satisfaction response submitted: {question_id} = {response_value}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting satisfaction response: {e}")
            return False
    
    def complete_satisfaction_session(self, session_id: str, 
                                    laptops_viewed: List[str] = None,
                                    recommendations_received: List[str] = None) -> bool:
        """
        Complete a satisfaction tracking session.
        
        Args:
            session_id: Session ID to complete
            laptops_viewed: List of laptop IDs viewed during session
            recommendations_received: List of laptop IDs recommended
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE satisfaction_sessions 
                SET end_time = ?, laptops_viewed = ?, recommendations_received = ?, session_completed = TRUE
                WHERE session_id = ?
            ''', (
                datetime.now().isoformat(),
                json.dumps(laptops_viewed or []),
                json.dumps(recommendations_received or []),
                session_id
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Satisfaction session completed: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error completing satisfaction session: {e}")
            return False
    
    def calculate_satisfaction_metrics(self, days: int = 30) -> SatisfactionMetrics:
        """
        Calculate comprehensive satisfaction metrics.
        
        Args:
            days: Number of days to include in calculation
            
        Returns:
            SatisfactionMetrics object with calculated metrics
        """
        try:
            # Get responses from the specified period
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            
            # Get all responses from the period
            responses_df = pd.read_sql_query('''
                SELECT * FROM satisfaction_responses 
                WHERE timestamp >= ? AND response_type = 'rating'
            ''', conn, params=(cutoff_date,))
            
            conn.close()
            
            if responses_df.empty:
                logger.warning("No satisfaction responses found for the specified period")
                return SatisfactionMetrics(
                    overall_satisfaction=0.0,
                    satisfaction_std=0.0,
                    response_count=0,
                    satisfaction_percentage=0.0,
                    category_scores={},
                    trend_data=[],
                    recommendations=[]
                )
            
            # Calculate overall satisfaction
            rating_responses = responses_df[responses_df['response_value'].str.isdigit()]
            if not rating_responses.empty:
                ratings = rating_responses['response_value'].astype(float)
                overall_satisfaction = ratings.mean()
                satisfaction_std = ratings.std()
                satisfaction_percentage = (overall_satisfaction / 5.0) * 100
            else:
                overall_satisfaction = 0.0
                satisfaction_std = 0.0
                satisfaction_percentage = 0.0
            
            # Calculate category scores
            category_scores = self._calculate_category_scores(responses_df)
            
            # Calculate trend data
            trend_data = self._calculate_trend_data(responses_df)
            
            # Generate recommendations
            recommendations = self._generate_satisfaction_recommendations(
                overall_satisfaction, category_scores, responses_df
            )
            
            return SatisfactionMetrics(
                overall_satisfaction=float(overall_satisfaction),
                satisfaction_std=float(satisfaction_std),
                response_count=len(responses_df),
                satisfaction_percentage=float(satisfaction_percentage),
                category_scores=category_scores,
                trend_data=trend_data,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error calculating satisfaction metrics: {e}")
            return SatisfactionMetrics(
                overall_satisfaction=0.0,
                satisfaction_std=0.0,
                response_count=0,
                satisfaction_percentage=0.0,
                category_scores={},
                trend_data=[],
                recommendations=[f"Error calculating metrics: {str(e)}"]
            )
    
    def _get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM satisfaction_sessions WHERE session_id = ?
            ''', (session_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'session_id': result[0],
                    'user_id': result[1],
                    'start_time': result[2],
                    'end_time': result[3],
                    'recommendation_method': result[4],
                    'laptops_viewed': result[5],
                    'recommendations_received': result[6],
                    'session_completed': result[7]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return None
    
    def _calculate_category_scores(self, responses_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate satisfaction scores by category."""
        try:
            category_scores = {}
            
            # Get question categories
            question_categories = {q['id']: q['category'] for q in self.survey_questions}
            
            for category in set(question_categories.values()):
                category_questions = [qid for qid, cat in question_categories.items() if cat == category]
                category_responses = responses_df[responses_df['question_id'].isin(category_questions)]
                
                if not category_responses.empty:
                    rating_responses = category_responses[category_responses['response_value'].str.isdigit()]
                    if not rating_responses.empty:
                        ratings = rating_responses['response_value'].astype(float)
                        category_scores[category] = float(ratings.mean())
                    else:
                        category_scores[category] = 0.0
                else:
                    category_scores[category] = 0.0
            
            return category_scores
            
        except Exception as e:
            logger.error(f"Error calculating category scores: {e}")
            return {}
    
    def _calculate_trend_data(self, responses_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate satisfaction trends over time."""
        try:
            # Group by date and calculate daily averages
            responses_df['date'] = pd.to_datetime(responses_df['timestamp']).dt.date
            daily_scores = responses_df.groupby('date')['response_value'].apply(
                lambda x: pd.to_numeric(x, errors='coerce').mean()
            ).reset_index()
            
            trend_data = []
            for _, row in daily_scores.iterrows():
                if not pd.isna(row['response_value']):
                    trend_data.append({
                        'date': row['date'].isoformat(),
                        'satisfaction_score': float(row['response_value']),
                        'response_count': len(responses_df[responses_df['date'] == row['date']])
                    })
            
            return trend_data
            
        except Exception as e:
            logger.error(f"Error calculating trend data: {e}")
            return []
    
    def _generate_satisfaction_recommendations(self, overall_satisfaction: float, 
                                             category_scores: Dict[str, float],
                                             responses_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on satisfaction analysis."""
        recommendations = []
        
        try:
            # Overall satisfaction recommendations
            if overall_satisfaction < 3.0:
                recommendations.append("Overall satisfaction is low - consider improving core recommendation algorithms")
            elif overall_satisfaction < 4.0:
                recommendations.append("Overall satisfaction is moderate - focus on improving user experience")
            
            # Category-specific recommendations
            for category, score in category_scores.items():
                if score < 3.0:
                    if category == 'quality':
                        recommendations.append("Improve recommendation quality by refining similarity algorithms")
                    elif category == 'performance':
                        recommendations.append("Optimize system performance to reduce response times")
                    elif category == 'usability':
                        recommendations.append("Improve user interface and user experience design")
                    elif category == 'trust':
                        recommendations.append("Increase transparency in recommendation explanations")
                    elif category == 'discovery':
                        recommendations.append("Enhance recommendation diversity and novelty")
            
            # Text feedback analysis
            text_responses = responses_df[responses_df['response_type'] == 'text']
            if not text_responses.empty:
                common_issues = self._analyze_text_feedback(text_responses)
                recommendations.extend(common_issues)
            
            # General recommendations
            if not recommendations:
                recommendations.append("Satisfaction levels are good - continue monitoring and consider advanced features")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [f"Error generating recommendations: {str(e)}"]
    
    def _analyze_text_feedback(self, text_responses: pd.DataFrame) -> List[str]:
        """Analyze text feedback to identify common issues."""
        try:
            # Simple keyword analysis (in production, use more sophisticated NLP)
            feedback_text = ' '.join(text_responses['response_value'].astype(str).str.lower())
            
            issues = []
            if 'slow' in feedback_text or 'speed' in feedback_text:
                issues.append("Users report speed issues - optimize recommendation generation")
            if 'irrelevant' in feedback_text or 'not relevant' in feedback_text:
                issues.append("Users find recommendations irrelevant - improve matching algorithms")
            if 'confusing' in feedback_text or 'difficult' in feedback_text:
                issues.append("Users find interface confusing - improve usability")
            if 'missing' in feedback_text or 'need' in feedback_text:
                issues.append("Users request additional features - consider feature requests")
            
            return issues
            
        except Exception as e:
            logger.error(f"Error analyzing text feedback: {e}")
            return []
    
    def get_survey_questions(self) -> List[Dict[str, Any]]:
        """Get the list of survey questions."""
        return self.survey_questions
    
    def get_satisfaction_dashboard_data(self, days: int = 30) -> Dict[str, Any]:
        """Get data for satisfaction dashboard."""
        try:
            metrics = self.calculate_satisfaction_metrics(days)
            
            # Get additional statistics
            conn = sqlite3.connect(self.db_path)
            
            # Total sessions
            total_sessions = pd.read_sql_query('''
                SELECT COUNT(*) as count FROM satisfaction_sessions 
                WHERE start_time >= ?
            ''', conn, params=((datetime.now() - timedelta(days=days)).isoformat(),)).iloc[0]['count']
            
            # Completed sessions
            completed_sessions = pd.read_sql_query('''
                SELECT COUNT(*) as count FROM satisfaction_sessions 
                WHERE start_time >= ? AND session_completed = TRUE
            ''', conn, params=((datetime.now() - timedelta(days=days)).isoformat(),)).iloc[0]['count']
            
            # Response rate
            response_rate = (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0
            
            conn.close()
            
            return {
                'overall_satisfaction': metrics.overall_satisfaction,
                'satisfaction_percentage': metrics.satisfaction_percentage,
                'response_count': metrics.response_count,
                'category_scores': metrics.category_scores,
                'trend_data': metrics.trend_data,
                'recommendations': metrics.recommendations,
                'total_sessions': int(total_sessions),
                'completed_sessions': int(completed_sessions),
                'response_rate': float(response_rate),
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                'overall_satisfaction': 0.0,
                'satisfaction_percentage': 0.0,
                'response_count': 0,
                'category_scores': {},
                'trend_data': [],
                'recommendations': [f"Error loading data: {str(e)}"],
                'total_sessions': 0,
                'completed_sessions': 0,
                'response_rate': 0.0,
                'period_days': days
            }
    
    def export_satisfaction_data(self, filename: str = None) -> str:
        """Export satisfaction data to CSV file."""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"satisfaction_data_{timestamp}.csv"
            
            conn = sqlite3.connect(self.db_path)
            
            # Export responses
            responses_df = pd.read_sql_query('''
                SELECT * FROM satisfaction_responses
            ''', conn)
            
            conn.close()
            
            responses_df.to_csv(filename, index=False)
            logger.info(f"Satisfaction data exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting satisfaction data: {e}")
            return ""


def create_satisfaction_system(db_path: str = "satisfaction_data.db") -> UserSatisfactionSystem:
    """
    Factory function to create a UserSatisfactionSystem instance.
    
    Args:
        db_path: Path to SQLite database for storing satisfaction data
        
    Returns:
        UserSatisfactionSystem instance
    """
    return UserSatisfactionSystem(db_path)


if __name__ == "__main__":
    # Example usage
    print("User Satisfaction System for Laptop Recommender System")
    print("This module provides comprehensive user satisfaction tracking and analysis.")
    print("Use create_satisfaction_system() to get started.")
