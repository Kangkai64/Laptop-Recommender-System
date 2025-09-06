"""
User Management System for Laptop Recommender System

This module handles user creation, selection, and behavior tracking including:
- User profile management
- View history tracking
- Rating and comment management
- User preference learning
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import sqlite3
from dataclasses import dataclass, asdict
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile data structure."""
    user_id: str
    username: str
    email: Optional[str] = None
    created_at: str = None
    last_active: str = None
    preferences: Dict = None
    total_views: int = 0
    total_ratings: int = 0
    total_comments: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_active is None:
            self.last_active = datetime.now().isoformat()
        if self.preferences is None:
            self.preferences = {}

@dataclass
class UserBehavior:
    """User behavior tracking data structure."""
    behavior_id: str
    user_id: str
    laptop_id: int
    behavior_type: str  # 'view', 'rating', 'comment', 'like', 'dislike'
    timestamp: str = None
    data: Dict = None  # Additional data like rating value, comment text, etc.
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.data is None:
            self.data = {}

class UserManager:
    """Manages user profiles and behavior tracking."""
    
    def __init__(self, db_path: str = "data/user_data.db"):
        """Initialize the user manager with database connection."""
        self.db_path = db_path
        self.ensure_database_exists()
        self._initialize_tables()
    
    def ensure_database_exists(self):
        """Ensure the database directory and file exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _initialize_tables(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    preferences TEXT,
                    total_views INTEGER DEFAULT 0,
                    total_ratings INTEGER DEFAULT 0,
                    total_comments INTEGER DEFAULT 0
                )
            ''')
            
            # User behavior table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_behavior (
                    behavior_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    laptop_id INTEGER NOT NULL,
                    behavior_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # User ratings table (for quick access)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_ratings (
                    user_id TEXT,
                    laptop_id INTEGER,
                    rating REAL NOT NULL,
                    comment TEXT,
                    timestamp TEXT NOT NULL,
                    PRIMARY KEY (user_id, laptop_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # User viewed laptops table (for quick access)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_views (
                    user_id TEXT,
                    laptop_id INTEGER,
                    view_count INTEGER DEFAULT 1,
                    first_viewed TEXT NOT NULL,
                    last_viewed TEXT NOT NULL,
                    PRIMARY KEY (user_id, laptop_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()
    
    def create_user(self, username: str, email: Optional[str] = None, 
                   preferences: Optional[Dict] = None) -> UserProfile:
        """Create a new user profile."""
        user_id = str(uuid.uuid4())
        
        if preferences is None:
            preferences = {}
        
        user_profile = UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            preferences=preferences
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (user_id, username, email, created_at, last_active, preferences)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id, username, email, user_profile.created_at, 
                user_profile.last_active, json.dumps(preferences)
            ))
            conn.commit()
        
        logger.info(f"Created new user: {username} (ID: {user_id})")
        return user_profile
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, username, email, created_at, last_active, 
                       preferences, total_views, total_ratings, total_comments
                FROM users WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            if row:
                return UserProfile(
                    user_id=row[0],
                    username=row[1],
                    email=row[2],
                    created_at=row[3],
                    last_active=row[4],
                    preferences=json.loads(row[5]) if row[5] else {},
                    total_views=row[6],
                    total_ratings=row[7],
                    total_comments=row[8]
                )
        return None
    
    def get_user_by_username(self, username: str) -> Optional[UserProfile]:
        """Get user profile by username."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, username, email, created_at, last_active, 
                       preferences, total_views, total_ratings, total_comments
                FROM users WHERE username = ?
            ''', (username,))
            
            row = cursor.fetchone()
            if row:
                return UserProfile(
                    user_id=row[0],
                    username=row[1],
                    email=row[2],
                    created_at=row[3],
                    last_active=row[4],
                    preferences=json.loads(row[5]) if row[5] else {},
                    total_views=row[6],
                    total_ratings=row[7],
                    total_comments=row[8]
                )
        return None
    
    def list_users(self, limit: int = 100) -> List[UserProfile]:
        """List all users with pagination."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, username, email, created_at, last_active, 
                       preferences, total_views, total_ratings, total_comments
                FROM users ORDER BY last_active DESC LIMIT ?
            ''', (limit,))
            
            users = []
            for row in cursor.fetchall():
                users.append(UserProfile(
                    user_id=row[0],
                    username=row[1],
                    email=row[2],
                    created_at=row[3],
                    last_active=row[4],
                    preferences=json.loads(row[5]) if row[5] else {},
                    total_views=row[6],
                    total_ratings=row[7],
                    total_comments=row[8]
                ))
            return users
    
    def search_users(self, search_term: str, limit: int = 50) -> List[UserProfile]:
        """Search users by username, user_id, or email."""
        if not search_term or not search_term.strip():
            return self.list_users(limit)
        
        search_term = f"%{search_term.strip()}%"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, username, email, created_at, last_active, 
                       preferences, total_views, total_ratings, total_comments
                FROM users 
                WHERE username LIKE ? OR user_id LIKE ? OR email LIKE ?
                ORDER BY 
                    CASE 
                        WHEN username LIKE ? THEN 1
                        WHEN user_id LIKE ? THEN 2
                        WHEN email LIKE ? THEN 3
                        ELSE 4
                    END,
                    last_active DESC
                LIMIT ?
            ''', (search_term, search_term, search_term, 
                  search_term.replace('%', ''), search_term.replace('%', ''), search_term.replace('%', ''), 
                  limit))
            
            users = []
            for row in cursor.fetchall():
                users.append(UserProfile(
                    user_id=row[0],
                    username=row[1],
                    email=row[2],
                    created_at=row[3],
                    last_active=row[4],
                    preferences=json.loads(row[5]) if row[5] else {},
                    total_views=row[6],
                    total_ratings=row[7],
                    total_comments=row[8]
                ))
            return users
    
    def find_user_by_id_or_username(self, identifier: str) -> Optional[UserProfile]:
        """Find user by exact user_id or username match."""
        if not identifier or not identifier.strip():
            return None
        
        identifier = identifier.strip()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, username, email, created_at, last_active, 
                       preferences, total_views, total_ratings, total_comments
                FROM users 
                WHERE user_id = ? OR username = ?
            ''', (identifier, identifier))
            
            row = cursor.fetchone()
            if row:
                return UserProfile(
                    user_id=row[0],
                    username=row[1],
                    email=row[2],
                    created_at=row[3],
                    last_active=row[4],
                    preferences=json.loads(row[5]) if row[5] else {},
                    total_views=row[6],
                    total_ratings=row[7],
                    total_comments=row[8]
                )
        return None
    
    def update_user_activity(self, user_id: str):
        """Update user's last active timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET last_active = ? WHERE user_id = ?
            ''', (datetime.now().isoformat(), user_id))
            conn.commit()
    
    def update_user_preferences(self, user_id: str, preferences: Dict):
        """Update user preferences."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET preferences = ? WHERE user_id = ?
            ''', (json.dumps(preferences), user_id))
            conn.commit()
    
    def track_behavior(self, user_id: str, laptop_id: int, behavior_type: str, 
                      data: Optional[Dict] = None) -> str:
        """Track user behavior (view, rating, comment, etc.)."""
        behavior_id = str(uuid.uuid4())
        
        if data is None:
            data = {}
        
        behavior = UserBehavior(
            behavior_id=behavior_id,
            user_id=user_id,
            laptop_id=laptop_id,
            behavior_type=behavior_type,
            data=data
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert behavior record
            cursor.execute('''
                INSERT INTO user_behavior (behavior_id, user_id, laptop_id, behavior_type, timestamp, data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                behavior_id, user_id, laptop_id, behavior_type, 
                behavior.timestamp, json.dumps(data)
            ))
            
            # Update user activity (within the same connection)
            cursor.execute('''
                UPDATE users SET last_active = ? WHERE user_id = ?
            ''', (datetime.now().isoformat(), user_id))
            
            # Update specific behavior counters and tables
            if behavior_type == 'view':
                self._update_view_tracking(cursor, user_id, laptop_id)
            elif behavior_type == 'rating':
                self._update_rating_tracking(cursor, user_id, laptop_id, data)
            elif behavior_type == 'comment':
                self._update_comment_tracking(cursor, user_id, laptop_id, data)
            
            conn.commit()
        
        logger.info(f"Tracked {behavior_type} behavior for user {user_id} on laptop {laptop_id}")
        return behavior_id
    
    def _update_view_tracking(self, cursor, user_id: str, laptop_id: int):
        """Update view tracking in dedicated table."""
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_views (user_id, laptop_id, view_count, first_viewed, last_viewed)
            VALUES (?, ?, 
                COALESCE((SELECT view_count FROM user_views WHERE user_id = ? AND laptop_id = ?), 0) + 1,
                COALESCE((SELECT first_viewed FROM user_views WHERE user_id = ? AND laptop_id = ?), ?),
                ?)
        ''', (user_id, laptop_id, user_id, laptop_id, user_id, laptop_id, now, now))
        
        # Update user's total views counter
        cursor.execute('''
            UPDATE users SET total_views = total_views + 1 WHERE user_id = ?
        ''', (user_id,))
    
    def _update_rating_tracking(self, cursor, user_id: str, laptop_id: int, data: Dict):
        """Update rating tracking in dedicated table."""
        rating = data.get('rating', 0)
        comment = data.get('comment', '')
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_ratings (user_id, laptop_id, rating, comment, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, laptop_id, rating, comment, timestamp))
        
        # Update user's total ratings counter
        cursor.execute('''
            UPDATE users SET total_ratings = total_ratings + 1 WHERE user_id = ?
        ''', (user_id,))
    
    def _update_comment_tracking(self, cursor, user_id: str, laptop_id: int, data: Dict):
        """Update comment tracking."""
        # Update user's total comments counter
        cursor.execute('''
            UPDATE users SET total_comments = total_comments + 1 WHERE user_id = ?
        ''', (user_id,))
    
    def get_user_views(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's view history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT laptop_id, view_count, first_viewed, last_viewed
                FROM user_views 
                WHERE user_id = ? 
                ORDER BY last_viewed DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            views = []
            for row in cursor.fetchall():
                views.append({
                    'laptop_id': row[0],
                    'view_count': row[1],
                    'first_viewed': row[2],
                    'last_viewed': row[3]
                })
            return views
    
    def get_user_ratings(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's rating history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT laptop_id, rating, comment, timestamp
                FROM user_ratings 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            ratings = []
            for row in cursor.fetchall():
                ratings.append({
                    'laptop_id': row[0],
                    'rating': row[1],
                    'comment': row[2],
                    'timestamp': row[3]
                })
            return ratings
    
    def get_user_behavior_history(self, user_id: str, behavior_type: Optional[str] = None, 
                                 limit: int = 100) -> List[Dict]:
        """Get user's complete behavior history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if behavior_type:
                cursor.execute('''
                    SELECT behavior_id, laptop_id, behavior_type, timestamp, data
                    FROM user_behavior 
                    WHERE user_id = ? AND behavior_type = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (user_id, behavior_type, limit))
            else:
                cursor.execute('''
                    SELECT behavior_id, laptop_id, behavior_type, timestamp, data
                    FROM user_behavior 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (user_id, limit))
            
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
    
    def get_user_statistics(self, user_id: str) -> Dict:
        """Get comprehensive user statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get basic user info
            user = self.get_user(user_id)
            if not user:
                return {}
            
            # Get behavior counts by type
            cursor.execute('''
                SELECT behavior_type, COUNT(*) as count
                FROM user_behavior 
                WHERE user_id = ?
                GROUP BY behavior_type
            ''', (user_id,))
            
            behavior_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get recent activity (last 30 days)
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
            cursor.execute('''
                SELECT COUNT(*) as recent_activity
                FROM user_behavior 
                WHERE user_id = ? AND timestamp >= ?
            ''', (user_id, thirty_days_ago))
            
            recent_activity = cursor.fetchone()[0]
            
            # Get most viewed laptops
            cursor.execute('''
                SELECT laptop_id, view_count
                FROM user_views 
                WHERE user_id = ? 
                ORDER BY view_count DESC 
                LIMIT 5
            ''', (user_id,))
            
            top_viewed = [{'laptop_id': row[0], 'view_count': row[1]} for row in cursor.fetchall()]
            
            # Get average rating given by user
            cursor.execute('''
                SELECT AVG(rating) as avg_rating, COUNT(*) as rating_count
                FROM user_ratings 
                WHERE user_id = ?
            ''', (user_id,))
            
            rating_stats = cursor.fetchone()
            avg_rating = rating_stats[0] if rating_stats[0] else 0
            rating_count = rating_stats[1] if rating_stats[1] else 0
            
            return {
                'user_profile': asdict(user),
                'behavior_counts': behavior_counts,
                'recent_activity': recent_activity,
                'top_viewed_laptops': top_viewed,
                'rating_statistics': {
                    'average_rating_given': avg_rating,
                    'total_ratings': rating_count
                }
            }
    
    def get_existing_users_from_ratings(self, df_rating: pd.DataFrame) -> List[Dict]:
        """Get list of existing users from the rating dataset."""
        if df_rating is None or 'user_id_encoded' not in df_rating.columns:
            return []
        
        # Get unique users from the rating dataset
        unique_users = df_rating['user_id_encoded'].unique()
        
        users_info = []
        for user_id_encoded in unique_users[:100]:  # Limit to first 100 for performance
            # Get some sample data for this user
            user_ratings = df_rating[df_rating['user_id_encoded'] == user_id_encoded]
            
            # Convert numpy types to Python native types
            user_id_encoded_py = int(user_id_encoded) if hasattr(user_id_encoded, 'item') else user_id_encoded
            total_ratings = int(len(user_ratings))
            avg_rating = float(user_ratings['rating'].mean()) if 'rating' in user_ratings.columns else 0.0
            first_rating = user_ratings['timestamp'].min() if 'timestamp' in user_ratings.columns else None
            last_rating = user_ratings['timestamp'].max() if 'timestamp' in user_ratings.columns else None
            
            # Convert timestamp to string if it's not None
            if first_rating is not None and hasattr(first_rating, 'item'):
                first_rating = str(first_rating)
            if last_rating is not None and hasattr(last_rating, 'item'):
                last_rating = str(last_rating)
            
            users_info.append({
                'user_id_encoded': user_id_encoded_py,
                'username': f"User_{user_id_encoded_py}",
                'total_ratings': total_ratings,
                'avg_rating': avg_rating,
                'first_rating': first_rating,
                'last_rating': last_rating
            })
        
        return users_info
    
    def create_user_from_existing(self, user_id_encoded: str, username: Optional[str] = None) -> UserProfile:
        """Create a new user profile from an existing user in the rating dataset."""
        if username is None:
            username = f"User_{user_id_encoded}"
        
        # Check if user already exists
        existing_user = self.get_user_by_username(username)
        if existing_user:
            return existing_user
        
        # Create new user
        user_profile = self.create_user(username=username)
        
        logger.info(f"Created user profile for existing user {user_id_encoded} as {username}")
        return user_profile
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user and all their data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete all user data
                cursor.execute('DELETE FROM user_behavior WHERE user_id = ?', (user_id,))
                cursor.execute('DELETE FROM user_ratings WHERE user_id = ?', (user_id,))
                cursor.execute('DELETE FROM user_views WHERE user_id = ?', (user_id,))
                cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
                
                conn.commit()
            
            logger.info(f"Deleted user {user_id} and all associated data")
            return True
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {e}")
            return False


def create_user_manager(db_path: str = "data/user_data.db") -> UserManager:
    """Factory function to create a UserManager instance."""
    return UserManager(db_path)


def main():
    """Test the user management system."""
    # Create user manager
    user_manager = create_user_manager()
    
    # Create a test user
    user = user_manager.create_user("test_user", "test@example.com")
    print(f"Created user: {user.username} (ID: {user.user_id})")
    
    # Track some behavior
    user_manager.track_behavior(user.user_id, 1, "view", {"page": "laptop_detail"})
    user_manager.track_behavior(user.user_id, 1, "rating", {"rating": 4.5, "comment": "Great laptop!"})
    user_manager.track_behavior(user.user_id, 2, "view", {"page": "laptop_detail"})
    
    # Get user statistics
    stats = user_manager.get_user_statistics(user.user_id)
    print(f"User statistics: {stats}")
    
    # List all users
    users = user_manager.list_users()
    print(f"Total users: {len(users)}")


if __name__ == "__main__":
    main()
