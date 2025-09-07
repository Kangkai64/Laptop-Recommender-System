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
    comments: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_active is None:
            self.last_active = datetime.now().isoformat()
        if self.preferences is None:
            self.preferences = {}
        if self.comments is None:
            self.comments = []

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

@dataclass
class ViewedProduct:
    """Enhanced viewed product data structure with laptop details."""
    laptop_id: int
    view_count: int
    first_viewed: str
    last_viewed: str
    laptop_title: Optional[str] = None
    laptop_brand: Optional[str] = None
    laptop_price: Optional[float] = None
    laptop_rating: Optional[float] = None
    laptop_image: Optional[str] = None

@dataclass
class UserRating:
    """Enhanced user rating data structure with laptop details."""
    laptop_id: int
    rating: float
    timestamp: str
    comment: Optional[str] = None
    laptop_title: Optional[str] = None
    laptop_brand: Optional[str] = None
    laptop_price: Optional[float] = None
    laptop_rating: Optional[float] = None
    laptop_image: Optional[str] = None

@dataclass
class UserStatsHistory:
    """Comprehensive user statistics with history objects."""
    user_id: str
    username: str
    email: Optional[str]
    created_at: str
    last_active: str
    total_views: int
    total_ratings: int
    total_comments: int
    viewed_products: List[ViewedProduct]
    rating_history: List[UserRating]
    recent_activity: List[UserBehavior]

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
                    total_comments INTEGER DEFAULT 0,
                    comments TEXT
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
            
            # Add comments column to existing users table if it doesn't exist
            try:
                cursor.execute('ALTER TABLE users ADD COLUMN comments TEXT')
                logger.info("Added comments column to users table")
            except sqlite3.OperationalError:
                # Column already exists, ignore
                pass
            
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
                INSERT INTO users (user_id, username, email, created_at, last_active, preferences, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, username, email, user_profile.created_at, 
                user_profile.last_active, json.dumps(preferences), json.dumps(user_profile.comments)
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
                       preferences, total_views, total_ratings, total_comments, comments, comments
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
                    total_comments=row[8],
                    comments=json.loads(row[9]) if row[9] else []
                )
        return None
    
    def get_user_by_username(self, username: str) -> Optional[UserProfile]:
        """Get user profile by username."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, username, email, created_at, last_active, 
                       preferences, total_views, total_ratings, total_comments, comments
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
                    total_comments=row[8],
                    comments=json.loads(row[9]) if row[9] else []
                )
        return None
    
    def list_users(self, limit: int = 100) -> List[UserProfile]:
        """List all users with pagination."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, username, email, created_at, last_active, 
                       preferences, total_views, total_ratings, total_comments, comments
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
                    total_comments=row[8],
                    comments=json.loads(row[9]) if row[9] else []
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
                       preferences, total_views, total_ratings, total_comments, comments
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
                    total_comments=row[8],
                    comments=json.loads(row[9]) if row[9] else []
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
                       preferences, total_views, total_ratings, total_comments, comments
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
                    total_comments=row[8],
                    comments=json.loads(row[9]) if row[9] else []
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
        
        # Check if this user has already rated this laptop
        cursor.execute('''
            SELECT COUNT(*) FROM user_ratings WHERE user_id = ? AND laptop_id = ?
        ''', (user_id, laptop_id))
        existing_rating_count = cursor.fetchone()[0]
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_ratings (user_id, laptop_id, rating, comment, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, laptop_id, rating, comment, timestamp))
        
        # Only increment total ratings counter if this is a new rating
        if existing_rating_count == 0:
            cursor.execute('''
                UPDATE users SET total_ratings = total_ratings + 1 WHERE user_id = ?
            ''', (user_id,))
        
        # Update total comments counter if comment is provided and not empty
        if comment and comment.strip():
            # Check if this is a new comment (not just updating existing rating)
            cursor.execute('''
                SELECT comment FROM user_ratings WHERE user_id = ? AND laptop_id = ?
            ''', (user_id, laptop_id))
            existing_comment = cursor.fetchone()
            
            # Only increment if this is a new comment or updating from empty to non-empty
            if not existing_comment or not existing_comment[0] or not existing_comment[0].strip():
                cursor.execute('''
                    UPDATE users SET total_comments = total_comments + 1 WHERE user_id = ?
                ''', (user_id,))
    
    def _update_comment_tracking(self, cursor, user_id: str, laptop_id: int, data: Dict):
        """Update comment tracking."""
        comment = data.get('comment', '')
        timestamp = datetime.now().isoformat()
        
        # Check if this user has already commented on this laptop
        cursor.execute('''
            SELECT COUNT(*) FROM user_ratings WHERE user_id = ? AND laptop_id = ? AND comment IS NOT NULL AND comment != ''
        ''', (user_id, laptop_id))
        existing_comment_count = cursor.fetchone()[0]
        
        # Insert or update comment in user_ratings table
        cursor.execute('''
            INSERT OR REPLACE INTO user_ratings (user_id, laptop_id, rating, comment, timestamp)
            VALUES (?, ?, 
                COALESCE((SELECT rating FROM user_ratings WHERE user_id = ? AND laptop_id = ?), 0),
                ?, ?)
        ''', (user_id, laptop_id, user_id, laptop_id, comment, timestamp))
        
        # Only increment total comments counter if this is a new comment
        if existing_comment_count == 0:
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
    
    def debug_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Debug method to check user statistics from all sources."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get user info from users table
            cursor.execute('''
                SELECT total_views, total_ratings, total_comments 
                FROM users WHERE user_id = ?
            ''', (user_id,))
            user_row = cursor.fetchone()
            user_stats = {
                'total_views': user_row[0] if user_row else 0,
                'total_ratings': user_row[1] if user_row else 0,
                'total_comments': user_row[2] if user_row else 0
            } if user_row else {'total_views': 0, 'total_ratings': 0, 'total_comments': 0}
            
            # Count actual records in user_views table
            cursor.execute('''
                SELECT COUNT(*), SUM(view_count) FROM user_views WHERE user_id = ?
            ''', (user_id,))
            views_row = cursor.fetchone()
            actual_views = {
                'unique_laptops': views_row[0] if views_row[0] else 0,
                'total_view_count': views_row[1] if views_row[1] else 0
            }
            
            # Count actual records in user_ratings table
            cursor.execute('''
                SELECT COUNT(*) FROM user_ratings WHERE user_id = ?
            ''', (user_id,))
            ratings_count = cursor.fetchone()[0]
            
            # Count actual comments in user_ratings table
            cursor.execute('''
                SELECT COUNT(*) FROM user_ratings WHERE user_id = ? AND comment IS NOT NULL AND comment != ''
            ''', (user_id,))
            comments_count = cursor.fetchone()[0]
            
            return {
                'user_table_stats': user_stats,
                'actual_views': actual_views,
                'actual_ratings_count': ratings_count,
                'actual_comments_count': comments_count,
                'recalculated_stats': self.recalculate_user_stats(user_id)
            }
    
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
    
    def get_enhanced_user_views(self, user_id: str, limit: int = 50) -> List[ViewedProduct]:
        """Get user's view history with enhanced laptop details."""
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
                laptop_id = row[0]
                
                # Try to get laptop details from the main database
                laptop_details = self._get_laptop_details(laptop_id)
                
                viewed_product = ViewedProduct(
                    laptop_id=laptop_id,
                    view_count=row[1],
                    first_viewed=row[2],
                    last_viewed=row[3],
                    laptop_title=laptop_details.get('title_y'),
                    laptop_brand=laptop_details.get('brand'),
                    laptop_price=laptop_details.get('price_myr'),
                    laptop_rating=laptop_details.get('average_rating'),
                    laptop_image=laptop_details.get('image')
                )
                views.append(viewed_product)
            
            return views
    
    def get_enhanced_user_ratings(self, user_id: str, limit: int = 50) -> List[UserRating]:
        """Get user's rating history with enhanced laptop details."""
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
                laptop_id = row[0]
                
                # Try to get laptop details from the main database
                laptop_details = self._get_laptop_details(laptop_id)
                
                user_rating = UserRating(
                    laptop_id=laptop_id,
                    rating=row[1],
                    comment=row[2],
                    timestamp=row[3],
                    laptop_title=laptop_details.get('title_y'),
                    laptop_brand=laptop_details.get('brand'),
                    laptop_price=laptop_details.get('price_myr'),
                    laptop_rating=laptop_details.get('average_rating'),
                    laptop_image=laptop_details.get('image')
                )
                ratings.append(user_rating)
            
            return ratings
    
    def get_comprehensive_user_stats(self, user_id: str, views_limit: int = 20, 
                                   ratings_limit: int = 20, behavior_limit: int = 50) -> UserStatsHistory:
        """Get comprehensive user statistics with enhanced history objects."""
        user = self.get_user(user_id)
        if not user:
            return None
        
        # Recalculate stats to ensure accuracy
        recalculated_stats = self.recalculate_user_stats(user_id)
        
        # Get enhanced data
        viewed_products = self.get_enhanced_user_views(user_id, views_limit)
        rating_history = self.get_enhanced_user_ratings(user_id, ratings_limit)
        recent_activity = self.get_user_behavior_history(user_id, limit=behavior_limit)
        
        # Convert behavior history to UserBehavior objects
        behavior_objects = []
        for behavior_data in recent_activity:
            behavior = UserBehavior(
                behavior_id=behavior_data['behavior_id'],
                user_id=behavior_data['user_id'],
                laptop_id=behavior_data['laptop_id'],
                behavior_type=behavior_data['behavior_type'],
                timestamp=behavior_data['timestamp'],
                data=behavior_data.get('data', {})
            )
            behavior_objects.append(behavior)
        
        return UserStatsHistory(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
            last_active=user.last_active,
            total_views=recalculated_stats['total_views'],
            total_ratings=recalculated_stats['total_ratings'],
            total_comments=recalculated_stats['total_comments'],
            viewed_products=viewed_products,
            rating_history=rating_history,
            recent_activity=behavior_objects
        )
    
    def _get_laptop_details(self, laptop_id: int) -> Dict:
        """Helper method to get laptop details from the main database."""
        try:
            # Try to get laptop details from the main laptops database
            import sqlite3
            main_db_path = "data/laptops.db"  # Adjust path as needed
            
            if os.path.exists(main_db_path):
                with sqlite3.connect(main_db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT title_y, brand, price_myr, average_rating, image
                        FROM laptops 
                        WHERE id = ?
                    ''', (laptop_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        return {
                            'title_y': result[0],
                            'brand': result[1],
                            'price_myr': result[2],
                            'average_rating': result[3],
                            'image': result[4]
                        }
        except Exception as e:
            logger.warning(f"Could not fetch laptop details for {laptop_id}: {e}")
        
        return {}
    
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
    
    def get_existing_users_from_ratings(self, df_rating: pd.DataFrame = None, search_term: str = None, 
                                       min_rating_count: int = None, max_rating_count: int = None,
                                       min_avg_rating: float = None, max_avg_rating: float = None,
                                       limit: int = 100) -> List[Dict]:
        """
        Get list of existing users from the rating dataset using SQL queries.
        
        Args:
            df_rating: Rating dataframe (deprecated, kept for backward compatibility)
            search_term: Search term for user ID
            min_rating_count: Minimum number of ratings
            max_rating_count: Maximum number of ratings
            min_avg_rating: Minimum average rating given
            max_avg_rating: Maximum average rating given
            limit: Maximum number of results
            
        Returns:
            List[Dict]: List of user information dictionaries
        """
        try:
            # Try to use the new SQL client first
            from huggingface_sql_client import create_hf_sql_client
            
            hf_client = create_hf_sql_client()
            users = hf_client.search_users_with_filters(
                search_term=search_term,
                min_rating_count=min_rating_count,
                max_rating_count=max_rating_count,
                min_avg_rating=min_avg_rating,
                max_avg_rating=max_avg_rating,
                limit=limit
            )
            
            logger.info(f"Retrieved {len(users)} users using SQL client")
            return users
            
        except Exception as e:
            logger.warning(f"SQL client failed, falling back to pandas: {e}")
            
            # Fallback to original pandas-based implementation
            if df_rating is None or 'user_id_encoded' not in df_rating.columns:
                return []
            
            # Get unique users from the rating dataset
            unique_users = df_rating['user_id_encoded'].unique()
            
            users_info = []
            processed_count = 0
            
            for user_id_encoded in unique_users:
                # If we have a search term, check if this user matches
                if search_term:
                    user_id_str = str(user_id_encoded)
                    # Check if search term matches the user ID (exact or partial)
                    # The user_id_encoded contains the original string user ID like "AGCRVAT5OCWRNXLVEKKIX5ZPRETA"
                    if search_term.upper() not in user_id_str.upper():
                        continue
                
                # Get some sample data for this user
                user_ratings = df_rating[df_rating['user_id_encoded'] == user_id_encoded]
                
                # Convert numpy types to Python native types
                user_id_encoded_py = str(user_id_encoded) if hasattr(user_id_encoded, 'item') else str(user_id_encoded)
                total_ratings = int(len(user_ratings))
                avg_rating = float(user_ratings['rating'].mean()) if 'rating' in user_ratings.columns else 0.0
                first_rating = user_ratings['timestamp'].min() if 'timestamp' in user_ratings.columns else None
                last_rating = user_ratings['timestamp'].max() if 'timestamp' in user_ratings.columns else None
                
                # Apply rating count filters
                if min_rating_count is not None and total_ratings < min_rating_count:
                    continue
                if max_rating_count is not None and total_ratings > max_rating_count:
                    continue
                
                # Apply average rating filters
                if min_avg_rating is not None and avg_rating < min_avg_rating:
                    continue
                if max_avg_rating is not None and avg_rating > max_avg_rating:
                    continue
                
                # Convert timestamp to string if it's not None
                if first_rating is not None and hasattr(first_rating, 'item'):
                    first_rating = str(first_rating)
                if last_rating is not None and hasattr(last_rating, 'item'):
                    last_rating = str(last_rating)
                
                users_info.append({
                    'user_id_encoded': user_id_encoded_py,
                    'username': f"User_{user_id_encoded_py[:8]}...",  # Show first 8 chars for readability
                    'total_ratings': total_ratings,
                    'avg_rating': avg_rating,
                    'first_rating': first_rating,
                    'last_rating': last_rating
                })
                
                processed_count += 1
                
                # If we have no search term and processed enough users, stop
                # If we have a search term, continue searching through all users
                if not search_term and processed_count >= limit:
                    break
            
            return users_info
    
    def get_rating_count_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of rating counts across users.
        
        Returns:
            Dict[str, int]: Distribution of rating counts
        """
        try:
            # Try to use the new SQL client first
            from huggingface_sql_client import create_hf_sql_client
            
            hf_client = create_hf_sql_client()
            distribution = hf_client.get_rating_count_distribution()
            
            logger.info(f"Retrieved rating count distribution using SQL client")
            return distribution
            
        except Exception as e:
            logger.warning(f"SQL client failed for rating distribution: {e}")
            return {}
    
    def get_user_detailed_stats(self, user_id_encoded: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific user.
        
        Args:
            user_id_encoded: The encoded user ID
            
        Returns:
            Dict[str, Any]: Detailed user statistics
        """
        try:
            # Try to use the new SQL client first
            from huggingface_sql_client import create_hf_sql_client
            
            hf_client = create_hf_sql_client()
            stats = hf_client.get_user_detailed_stats(user_id_encoded)
            
            logger.info(f"Retrieved detailed stats for user {user_id_encoded} using SQL client")
            return stats
            
        except Exception as e:
            logger.warning(f"SQL client failed for user stats: {e}")
            return {}
    
    def create_user_from_existing(self, user_id_encoded: str, username: Optional[str] = None, df_rating: Optional[pd.DataFrame] = None) -> UserProfile:
        """Create a new user profile from an existing user in the rating dataset."""
        if username is None:
            username = f"User_{user_id_encoded}"
        
        # Check if user already exists
        existing_user = self.get_user_by_username(username)
        if existing_user:
            # Recalculate stats for existing user
            self.recalculate_user_stats(existing_user.user_id)
            return existing_user
        
        # Create new user
        user_profile = self.create_user(username=username)
        
        # Sync stats from Amazon data
        amazon_stats = self.sync_user_stats_from_amazon_data(user_id_encoded)
        
        # Extract comments from df_rating if available
        user_comments = []
        if df_rating is not None and 'text' in df_rating.columns and 'user_id_encoded' in df_rating.columns:
            user_ratings = df_rating[df_rating['user_id_encoded'] == user_id_encoded]
            # Extract non-empty text comments
            user_comments = user_ratings['text'].dropna().astype(str).tolist()
            # Filter out very short comments (less than 10 characters)
            user_comments = [comment for comment in user_comments if len(comment.strip()) >= 10]
            logger.info(f"Extracted {len(user_comments)} comments for user {user_id_encoded}")
        
        # Update the user profile with correct stats
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET 
                    total_ratings = ?,
                    total_views = ?,
                    total_comments = ?,
                    comments = ?
                WHERE user_id = ?
            ''', (amazon_stats['total_ratings'], amazon_stats['total_views'], amazon_stats['total_comments'], json.dumps(user_comments), user_profile.user_id))
            conn.commit()
        
        # Update the user profile object
        user_profile.total_ratings = amazon_stats['total_ratings']
        user_profile.total_views = amazon_stats['total_views']
        user_profile.total_comments = amazon_stats['total_comments']
        user_profile.comments = user_comments
        
        logger.info(f"Created user profile for existing user {user_id_encoded} as {username} with {amazon_stats['total_ratings']} ratings")
        return user_profile
    
    def sync_user_stats_from_amazon_data(self, user_id_encoded: str) -> Dict[str, int]:
        """
        Synchronize user statistics with actual data from Amazon dataset.
        
        Args:
            user_id_encoded: The encoded user ID from Amazon dataset
            
        Returns:
            Dict[str, int]: Updated statistics
        """
        try:
            # Try to use the new SQL client first
            from huggingface_sql_client import create_hf_sql_client
            
            hf_client = create_hf_sql_client()
            stats = hf_client.get_user_detailed_stats(user_id_encoded)
            
            # Extract the actual counts from Amazon data
            actual_ratings = stats.get('total_ratings', 0)
            actual_comments = stats.get('total_comments', 0)
            # Views aren't tracked in Amazon data, so we'll use ratings count as views
            # This ensures consistency with the sync function in data_preprocessing.py
            actual_views = actual_ratings
            
            logger.info(f"Synchronized stats for user {user_id_encoded}: ratings={actual_ratings}, comments={actual_comments}, views={actual_views}")
            
            return {
                'total_ratings': actual_ratings,
                'total_comments': actual_comments,
                'total_views': actual_views
            }
            
        except Exception as e:
            logger.warning(f"Failed to sync stats from Amazon data: {e}")
            return {'total_ratings': 0, 'total_comments': 0, 'total_views': 0}
    
    def recalculate_user_stats(self, user_id: str) -> Dict[str, int]:
        """
        Recalculate user statistics from the behavior tables.
        
        Args:
            user_id: The user ID
            
        Returns:
            Dict[str, int]: Recalculated statistics
        """
        # Check if this is an existing user from Amazon dataset
        if user_id.startswith('existing_'):
            # Extract the original user_id_encoded
            user_id_encoded = user_id[9:]  # Remove 'existing_' prefix
            
            # Try to get stats from Amazon dataset first
            try:
                from huggingface_sql_client import create_hf_sql_client
                hf_client = create_hf_sql_client()
                amazon_stats = hf_client.get_user_detailed_stats(user_id_encoded)
                
                if amazon_stats:
                    # Use Amazon dataset stats
                    actual_ratings = amazon_stats.get('total_ratings', 0)
                    actual_comments = amazon_stats.get('total_comments', 0)
                    actual_views = actual_ratings  # Use ratings count as views for consistency
                    
                    logger.info(f"Using Amazon dataset stats for user {user_id}: ratings={actual_ratings}, comments={actual_comments}, views={actual_views}")
                    
                    # Update the local user table with Amazon stats
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE users SET 
                                total_ratings = ?,
                                total_views = ?,
                                total_comments = ?
                            WHERE user_id = ?
                        ''', (actual_ratings, actual_views, actual_comments, user_id))
                        conn.commit()
                    
                    return {
                        'total_ratings': actual_ratings,
                        'total_views': actual_views,
                        'total_comments': actual_comments
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to get Amazon stats for user {user_id}: {e}")
                # Fall through to local database calculation
        
        # Fallback to local database calculation for regular users
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count actual ratings
            cursor.execute('''
                SELECT COUNT(*) FROM user_ratings WHERE user_id = ?
            ''', (user_id,))
            actual_ratings = cursor.fetchone()[0]
            
            # Count actual views
            cursor.execute('''
                SELECT SUM(view_count) FROM user_views WHERE user_id = ?
            ''', (user_id,))
            result = cursor.fetchone()[0]
            actual_views = result if result is not None else 0
            
            # Count actual comments (ratings with non-empty comments)
            cursor.execute('''
                SELECT COUNT(*) FROM user_ratings WHERE user_id = ? AND comment IS NOT NULL AND comment != ''
            ''', (user_id,))
            actual_comments = cursor.fetchone()[0]
            
            # Update the user table with correct counts
            cursor.execute('''
                UPDATE users SET 
                    total_ratings = ?,
                    total_views = ?,
                    total_comments = ?
                WHERE user_id = ?
            ''', (actual_ratings, actual_views, actual_comments, user_id))
            
            conn.commit()
            
            logger.info(f"Recalculated stats for user {user_id}: ratings={actual_ratings}, views={actual_views}, comments={actual_comments}")
            
            return {
                'total_ratings': actual_ratings,
                'total_views': actual_views,
                'total_comments': actual_comments
            }
    
    def sync_all_amazon_users_to_local_db(self) -> Dict[str, int]:
        """
        Sync all Amazon dataset users to local database with their statistics.
        This function can be called manually to ensure data consistency.
        
        Returns:
            Dict[str, int]: Sync results with counts
        """
        try:
            logger.info("Starting manual sync of all Amazon users to local database...")
            
            # Import data preprocessor to use its sync function
            from data_preprocessing import LaptopDataPreprocessor
            
            # Initialize preprocessor
            preprocessor = LaptopDataPreprocessor()
            
            # Load and process data to get rating dataframe
            df_laptop, df_rating = preprocessor.preprocess_separated_pipeline(force_reprocess=False)
            
            # Call the sync function
            preprocessor.sync_amazon_stats_to_local_db(df_rating)
            
            # Get summary statistics
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count total users in local database
                cursor.execute('SELECT COUNT(*) FROM users')
                total_users = cursor.fetchone()[0]
                
                # Count users with Amazon data
                cursor.execute('SELECT COUNT(*) FROM users WHERE username LIKE "Amazon_User_%"')
                amazon_users = cursor.fetchone()[0]
                
                # Get total ratings, views, and comments
                cursor.execute('SELECT SUM(total_ratings), SUM(total_views), SUM(total_comments) FROM users')
                stats = cursor.fetchone()
                total_ratings = stats[0] if stats[0] else 0
                total_views = stats[1] if stats[1] else 0
                total_comments = stats[2] if stats[2] else 0
            
            result = {
                'total_users': total_users,
                'amazon_users': amazon_users,
                'total_ratings': total_ratings,
                'total_views': total_views,
                'total_comments': total_comments
            }
            
            logger.info(f"Manual sync completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in manual sync: {e}")
            return {'error': str(e)}
    
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
