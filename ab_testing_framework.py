"""
A/B Testing Framework for Laptop Recommender System

This module provides a comprehensive A/B testing framework for comparing
different recommendation algorithms and configurations. It includes:
- Experiment design and management
- Statistical significance testing
- Performance comparison
- User segmentation
- Results analysis and reporting

Author: Laptop Recommender System Team
License: MIT
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import os
from scipy import stats
import random
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of A/B testing experiments."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MetricType(Enum):
    """Types of metrics to track in A/B tests."""
    CONVERSION = "conversion"
    ENGAGEMENT = "engagement"
    SATISFACTION = "satisfaction"
    PERFORMANCE = "performance"
    BUSINESS = "business"


@dataclass
class ExperimentConfig:
    """Configuration for an A/B testing experiment."""
    experiment_id: str
    name: str
    description: str
    start_date: str
    end_date: str
    status: ExperimentStatus
    variants: List[Dict[str, Any]]
    metrics: List[str]
    target_audience: Dict[str, Any]
    sample_size: int
    confidence_level: float
    minimum_effect_size: float
    created_at: str
    created_by: str


@dataclass
class ExperimentResult:
    """Results of an A/B testing experiment."""
    experiment_id: str
    variant_a_results: Dict[str, float]
    variant_b_results: Dict[str, float]
    statistical_significance: Dict[str, bool]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    effect_sizes: Dict[str, float]
    winner: Optional[str]
    recommendation: str
    analysis_date: str


class ABTestingFramework:
    """Comprehensive A/B testing framework for recommendation systems."""
    
    def __init__(self, db_path: str = "ab_testing.db"):
        """
        Initialize the A/B testing framework.
        
        Args:
            db_path: Path to SQLite database for storing experiment data
        """
        self.db_path = db_path
        self._initialize_database()
        
        logger.info("ABTestingFramework initialized successfully")
    
    def _initialize_database(self):
        """Initialize SQLite database for storing A/B testing data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    variants TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    target_audience TEXT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    confidence_level REAL NOT NULL,
                    minimum_effect_size REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL
                )
            ''')
            
            # Create experiment_results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    variant_a_results TEXT NOT NULL,
                    variant_b_results TEXT NOT NULL,
                    statistical_significance TEXT NOT NULL,
                    p_values TEXT NOT NULL,
                    confidence_intervals TEXT NOT NULL,
                    effect_sizes TEXT NOT NULL,
                    winner TEXT,
                    recommendation TEXT,
                    analysis_date TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            # Create user_assignments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_assignments (
                    user_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, experiment_id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            # Create experiment_events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiment_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_value REAL,
                    event_data TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("A/B testing database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing A/B testing database: {e}")
    
    def create_experiment(self, name: str, description: str, variants: List[Dict[str, Any]], 
                         metrics: List[str], duration_days: int = 14, 
                         sample_size: int = 1000, confidence_level: float = 0.95,
                         minimum_effect_size: float = 0.05, 
                         target_audience: Dict[str, Any] = None,
                         created_by: str = "system") -> str:
        """
        Create a new A/B testing experiment.
        
        Args:
            name: Name of the experiment
            description: Description of the experiment
            variants: List of variants to test (e.g., [{"name": "A", "config": {...}}, {"name": "B", "config": {...}}])
            metrics: List of metrics to track
            duration_days: Duration of the experiment in days
            sample_size: Target sample size for the experiment
            confidence_level: Confidence level for statistical significance (0.95 = 95%)
            minimum_effect_size: Minimum effect size to detect
            target_audience: Target audience criteria
            created_by: Creator of the experiment
            
        Returns:
            Experiment ID
        """
        try:
            experiment_id = str(uuid.uuid4())
            start_date = datetime.now().isoformat()
            end_date = (datetime.now() + timedelta(days=duration_days)).isoformat()
            
            if target_audience is None:
                target_audience = {"all_users": True}
            
            config = ExperimentConfig(
                experiment_id=experiment_id,
                name=name,
                description=description,
                start_date=start_date,
                end_date=end_date,
                status=ExperimentStatus.DRAFT,
                variants=variants,
                metrics=metrics,
                target_audience=target_audience,
                sample_size=sample_size,
                confidence_level=confidence_level,
                minimum_effect_size=minimum_effect_size,
                created_at=datetime.now().isoformat(),
                created_by=created_by
            )
            
            # Store experiment in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO experiments 
                (experiment_id, name, description, start_date, end_date, status, 
                 variants, metrics, target_audience, sample_size, confidence_level, 
                 minimum_effect_size, created_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                config.experiment_id,
                config.name,
                config.description,
                config.start_date,
                config.end_date,
                config.status.value,
                json.dumps(config.variants),
                json.dumps(config.metrics),
                json.dumps(config.target_audience),
                config.sample_size,
                config.confidence_level,
                config.minimum_effect_size,
                config.created_at,
                config.created_by
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Experiment created: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            return ""
    
    def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an A/B testing experiment.
        
        Args:
            experiment_id: ID of the experiment to start
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update experiment status
            cursor.execute('''
                UPDATE experiments 
                SET status = ? 
                WHERE experiment_id = ?
            ''', (ExperimentStatus.RUNNING.value, experiment_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Experiment started: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting experiment: {e}")
            return False
    
    def assign_user_to_variant(self, user_id: str, experiment_id: str) -> Optional[str]:
        """
        Assign a user to a variant for an experiment.
        
        Args:
            user_id: ID of the user
            experiment_id: ID of the experiment
            
        Returns:
            Variant name assigned to the user, or None if assignment failed
        """
        try:
            # Check if user is already assigned
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT variant FROM user_assignments 
                WHERE user_id = ? AND experiment_id = ?
            ''', (user_id, experiment_id))
            
            result = cursor.fetchone()
            if result:
                conn.close()
                return result[0]
            
            # Get experiment configuration
            cursor.execute('''
                SELECT variants, status FROM experiments 
                WHERE experiment_id = ?
            ''', (experiment_id,))
            
            result = cursor.fetchone()
            if not result:
                conn.close()
                return None
            
            variants_data = json.loads(result[0])
            status = result[1]
            
            if status != ExperimentStatus.RUNNING.value:
                conn.close()
                return None
            
            # Assign user to variant (simple random assignment)
            variant_names = [v['name'] for v in variants_data]
            assigned_variant = random.choice(variant_names)
            
            # Store assignment
            cursor.execute('''
                INSERT INTO user_assignments (user_id, experiment_id, variant, assigned_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, experiment_id, assigned_variant, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"User {user_id} assigned to variant {assigned_variant} for experiment {experiment_id}")
            return assigned_variant
            
        except Exception as e:
            logger.error(f"Error assigning user to variant: {e}")
            return None
    
    def track_event(self, experiment_id: str, user_id: str, event_type: str, 
                   event_value: float = None, event_data: Dict[str, Any] = None) -> bool:
        """
        Track an event for a user in an experiment.
        
        Args:
            experiment_id: ID of the experiment
            user_id: ID of the user
            event_type: Type of event (e.g., 'click', 'conversion', 'satisfaction')
            event_value: Numeric value of the event
            event_data: Additional data about the event
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO experiment_events 
                (experiment_id, user_id, event_type, event_value, event_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                user_id,
                event_type,
                event_value,
                json.dumps(event_data) if event_data else None,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error tracking event: {e}")
            return False
    
    def analyze_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """
        Analyze the results of an A/B testing experiment.
        
        Args:
            experiment_id: ID of the experiment to analyze
            
        Returns:
            ExperimentResult object with analysis results
        """
        try:
            # Get experiment configuration
            conn = sqlite3.connect(self.db_path)
            
            experiment_df = pd.read_sql_query('''
                SELECT * FROM experiments WHERE experiment_id = ?
            ''', conn, params=(experiment_id,))
            
            if experiment_df.empty:
                conn.close()
                return None
            
            experiment_config = experiment_df.iloc[0]
            variants = json.loads(experiment_config['variants'])
            metrics = json.loads(experiment_config['metrics'])
            confidence_level = experiment_config['confidence_level']
            
            # Get user assignments
            assignments_df = pd.read_sql_query('''
                SELECT * FROM user_assignments WHERE experiment_id = ?
            ''', conn, params=(experiment_id,))
            
            # Get experiment events
            events_df = pd.read_sql_query('''
                SELECT * FROM experiment_events WHERE experiment_id = ?
            ''', conn, params=(experiment_id,))
            
            conn.close()
            
            if assignments_df.empty or events_df.empty:
                logger.warning(f"No data found for experiment {experiment_id}")
                return None
            
            # Merge assignments with events
            data_df = events_df.merge(assignments_df, on=['user_id', 'experiment_id'], how='inner')
            
            # Analyze each metric
            variant_a_results = {}
            variant_b_results = {}
            statistical_significance = {}
            p_values = {}
            confidence_intervals = {}
            effect_sizes = {}
            
            variant_a_name = variants[0]['name']
            variant_b_name = variants[1]['name']
            
            for metric in metrics:
                # Get data for each variant
                variant_a_data = data_df[data_df['variant'] == variant_a_name][metric].dropna()
                variant_b_data = data_df[data_df['variant'] == variant_b_name][metric].dropna()
                
                if len(variant_a_data) == 0 or len(variant_b_data) == 0:
                    continue
                
                # Calculate basic statistics
                variant_a_results[metric] = {
                    'mean': float(variant_a_data.mean()),
                    'std': float(variant_a_data.std()),
                    'count': len(variant_a_data)
                }
                
                variant_b_results[metric] = {
                    'mean': float(variant_b_data.mean()),
                    'std': float(variant_b_data.std()),
                    'count': len(variant_b_data)
                }
                
                # Perform statistical test
                if metric in ['conversion_rate', 'click_rate']:
                    # Chi-square test for proportions
                    success_a = int(variant_a_data.sum())
                    total_a = len(variant_a_data)
                    success_b = int(variant_b_data.sum())
                    total_b = len(variant_b_data)
                    
                    # Create contingency table
                    contingency_table = np.array([[success_a, total_a - success_a],
                                                [success_b, total_b - success_b]])
                    
                    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                    
                    # Calculate effect size (Cramér's V)
                    n = total_a + total_b
                    effect_size = np.sqrt(chi2 / n)
                    
                else:
                    # T-test for continuous variables
                    t_stat, p_value = stats.ttest_ind(variant_a_data, variant_b_data)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(variant_a_data) - 1) * variant_a_data.var() + 
                                        (len(variant_b_data) - 1) * variant_b_data.var()) / 
                                       (len(variant_a_data) + len(variant_b_data) - 2))
                    effect_size = abs(variant_a_data.mean() - variant_b_data.mean()) / pooled_std
                
                # Calculate confidence interval for difference
                mean_diff = variant_a_data.mean() - variant_b_data.mean()
                se_diff = np.sqrt(variant_a_data.var() / len(variant_a_data) + 
                                variant_b_data.var() / len(variant_b_data))
                
                alpha = 1 - confidence_level
                t_critical = stats.t.ppf(1 - alpha/2, len(variant_a_data) + len(variant_b_data) - 2)
                margin_error = t_critical * se_diff
                
                ci_lower = mean_diff - margin_error
                ci_upper = mean_diff + margin_error
                
                # Store results
                statistical_significance[metric] = p_value < alpha
                p_values[metric] = float(p_value)
                confidence_intervals[metric] = (float(ci_lower), float(ci_upper))
                effect_sizes[metric] = float(effect_size)
            
            # Determine winner
            winner = None
            if statistical_significance:
                significant_metrics = [m for m, sig in statistical_significance.items() if sig]
                if significant_metrics:
                    # For now, use the first significant metric to determine winner
                    primary_metric = significant_metrics[0]
                    if variant_a_results[primary_metric]['mean'] > variant_b_results[primary_metric]['mean']:
                        winner = variant_a_name
                    else:
                        winner = variant_b_name
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                variant_a_results, variant_b_results, statistical_significance, 
                p_values, effect_sizes, winner
            )
            
            # Create result object
            result = ExperimentResult(
                experiment_id=experiment_id,
                variant_a_results=variant_a_results,
                variant_b_results=variant_b_results,
                statistical_significance=statistical_significance,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                effect_sizes=effect_sizes,
                winner=winner,
                recommendation=recommendation,
                analysis_date=datetime.now().isoformat()
            )
            
            # Store results in database
            self._store_experiment_result(result)
            
            logger.info(f"Experiment analysis completed: {experiment_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing experiment: {e}")
            return None
    
    def _generate_recommendation(self, variant_a_results: Dict, variant_b_results: Dict,
                               statistical_significance: Dict, p_values: Dict,
                               effect_sizes: Dict, winner: Optional[str]) -> str:
        """Generate a recommendation based on experiment results."""
        try:
            if not statistical_significance:
                return "No significant differences found. Consider running the experiment longer or increasing sample size."
            
            significant_metrics = [m for m, sig in statistical_significance.items() if sig]
            
            if not significant_metrics:
                return "No statistically significant differences found. Consider running the experiment longer."
            
            if winner:
                return f"Variant {winner} is the winner with significant improvements in {len(significant_metrics)} metric(s). Consider implementing this variant."
            else:
                return "Significant differences found but no clear winner. Consider analyzing individual metrics for specific improvements."
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "Error generating recommendation"
    
    def _store_experiment_result(self, result: ExperimentResult):
        """Store experiment result in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO experiment_results 
                (experiment_id, variant_a_results, variant_b_results, statistical_significance,
                 p_values, confidence_intervals, effect_sizes, winner, recommendation, analysis_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.experiment_id,
                json.dumps(result.variant_a_results),
                json.dumps(result.variant_b_results),
                json.dumps(result.statistical_significance),
                json.dumps(result.p_values),
                json.dumps(result.confidence_intervals),
                json.dumps(result.effect_sizes),
                result.winner,
                result.recommendation,
                result.analysis_date
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing experiment result: {e}")
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of an experiment."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get experiment details
            experiment_df = pd.read_sql_query('''
                SELECT * FROM experiments WHERE experiment_id = ?
            ''', conn, params=(experiment_id,))
            
            if experiment_df.empty:
                conn.close()
                return None
            
            experiment = experiment_df.iloc[0]
            
            # Get user assignment count
            assignments_df = pd.read_sql_query('''
                SELECT variant, COUNT(*) as count 
                FROM user_assignments 
                WHERE experiment_id = ? 
                GROUP BY variant
            ''', conn, params=(experiment_id,))
            
            # Get event count
            events_df = pd.read_sql_query('''
                SELECT COUNT(*) as event_count 
                FROM experiment_events 
                WHERE experiment_id = ?
            ''', conn, params=(experiment_id,))
            
            conn.close()
            
            return {
                'experiment_id': experiment['experiment_id'],
                'name': experiment['name'],
                'status': experiment['status'],
                'start_date': experiment['start_date'],
                'end_date': experiment['end_date'],
                'sample_size': experiment['sample_size'],
                'assigned_users': int(assignments_df['count'].sum()) if not assignments_df.empty else 0,
                'variant_assignments': assignments_df.to_dict('records') if not assignments_df.empty else [],
                'total_events': int(events_df.iloc[0]['event_count']) if not events_df.empty else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting experiment status: {e}")
            return None
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """List all experiments, optionally filtered by status."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if status:
                query = 'SELECT * FROM experiments WHERE status = ? ORDER BY created_at DESC'
                params = (status.value,)
            else:
                query = 'SELECT * FROM experiments ORDER BY created_at DESC'
                params = ()
            
            experiments_df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return experiments_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return []
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an A/B testing experiment."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE experiments 
                SET status = ? 
                WHERE experiment_id = ?
            ''', (ExperimentStatus.COMPLETED.value, experiment_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Experiment stopped: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping experiment: {e}")
            return False


def create_ab_testing_framework(db_path: str = "ab_testing.db") -> ABTestingFramework:
    """
    Factory function to create an ABTestingFramework instance.
    
    Args:
        db_path: Path to SQLite database for storing experiment data
        
    Returns:
        ABTestingFramework instance
    """
    return ABTestingFramework(db_path)


if __name__ == "__main__":
    # Example usage
    print("A/B Testing Framework for Laptop Recommender System")
    print("This module provides comprehensive A/B testing capabilities.")
    print("Use create_ab_testing_framework() to get started.")
