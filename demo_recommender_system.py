"""
Demo Script for Laptop Recommender System

This script demonstrates the capabilities of the Laptop Recommender System
by showing various recommendation scenarios and use cases.

Author: Laptop Recommender System Team
License: MIT
"""

import sys
import logging
from typing import Dict, List
import json

# Import our recommender system
from Laptop_Recommender_System import create_laptop_recommender_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_recommendations(title: str, recommendations: List[Dict], max_display: int = 5):
    """Print recommendations in a formatted way."""
    print(f"\n{title}")
    print("-" * 80)
    
    if not recommendations:
        print("No recommendations found.")
        return
    
    for i, rec in enumerate(recommendations[:max_display], 1):
        print(f"{i}. {rec.get('title', 'Unknown Title')}")
        print(f"   Brand: {rec.get('brand', 'Unknown')}")
        print(f"   Price: RM {rec.get('price_myr', 0):.2f}")
        print(f"   Rating: {rec.get('rating', 0):.1f}")
        print(f"   Score: {rec.get('recommendation_score', 0):.3f}")
        print(f"   Method: {rec.get('method', 'Unknown')}")
        if 'explanation' in rec:
            print(f"   Explanation: {rec['explanation']}")
        print()


def demo_content_based_filtering(recommender):
    """Demonstrate content-based filtering."""
    print("\n" + "="*80)
    print("DEMO: Content-Based Filtering")
    print("="*80)
    
    # Example 1: Gaming laptop preferences
    print("\n1. Gaming Laptop Recommendations")
    gaming_preferences = {
        'search_terms': ['gaming', 'gpu', 'graphics', 'performance'],
        'min_rating': 4.0,
        'max_price': 8000
    }
    
    try:
        gaming_recs = recommender.get_content_based_recommendations(gaming_preferences, 5)
        print_recommendations("Gaming Laptop Recommendations", gaming_recs)
    except Exception as e:
        print(f"Error getting gaming recommendations: {e}")
    
    # Example 2: Student laptop preferences
    print("\n2. Student Laptop Recommendations")
    student_preferences = {
        'search_terms': ['student', 'budget', 'affordable', 'basic'],
        'min_rating': 3.5,
        'max_price': 3000
    }
    
    try:
        student_recs = recommender.get_content_based_recommendations(student_preferences, 5)
        print_recommendations("Student Laptop Recommendations", student_recs)
    except Exception as e:
        print(f"Error getting student recommendations: {e}")
    
    # Example 3: Business laptop preferences
    print("\n3. Business Laptop Recommendations")
    business_preferences = {
        'search_terms': ['business', 'professional', 'work', 'office'],
        'min_rating': 4.0,
        'max_price': 6000
    }
    
    try:
        business_recs = recommender.get_content_based_recommendations(business_preferences, 5)
        print_recommendations("Business Laptop Recommendations", business_recs)
    except Exception as e:
        print(f"Error getting business recommendations: {e}")


def demo_collaborative_filtering(recommender):
    """Demonstrate collaborative filtering."""
    print("\n" + "="*80)
    print("DEMO: Collaborative Filtering")
    print("="*80)
    
    # Example 1: User-based recommendations
    print("\n1. User-Based Recommendations")
    try:
        user_recs = recommender.get_collaborative_filtering_recommendations(
            user_id=1, method='user_based', n_recommendations=5
        )
        print_recommendations("User-Based Recommendations", user_recs)
    except Exception as e:
        print(f"Error getting user-based recommendations: {e}")
    
    # Example 2: Item-based recommendations
    print("\n2. Item-Based Recommendations")
    try:
        item_recs = recommender.get_collaborative_filtering_recommendations(
            user_id=1, method='item_based', n_recommendations=5
        )
        print_recommendations("Item-Based Recommendations", item_recs)
    except Exception as e:
        print(f"Error getting item-based recommendations: {e}")
    
    # Example 3: Matrix factorization recommendations
    print("\n3. Matrix Factorization Recommendations")
    try:
        mf_recs = recommender.get_collaborative_filtering_recommendations(
            user_id=1, method='matrix_factorization', n_recommendations=5
        )
        print_recommendations("Matrix Factorization Recommendations", mf_recs)
    except Exception as e:
        print(f"Error getting matrix factorization recommendations: {e}")
    
    # Example 4: Hybrid collaborative filtering
    print("\n4. Hybrid Collaborative Filtering")
    try:
        hybrid_recs = recommender.get_collaborative_filtering_recommendations(
            user_id=1, method='hybrid', n_recommendations=5
        )
        print_recommendations("Hybrid Collaborative Filtering", hybrid_recs)
    except Exception as e:
        print(f"Error getting hybrid collaborative filtering recommendations: {e}")


def demo_hybrid_recommendations(recommender):
    """Demonstrate hybrid recommendations."""
    print("\n" + "="*80)
    print("DEMO: Hybrid Recommendations")
    print("="*80)
    
    # Example: Hybrid recommendations for a user
    user_id = 1
    user_preferences = {
        'search_terms': ['performance', 'reliable', 'good_value'],
        'min_rating': 3.5,
        'max_price': 5000
    }
    
    try:
        hybrid_recs = recommender.get_hybrid_recommendations(
            user_id=user_id,
            preferences=user_preferences,
            n_recommendations=5
        )
        print_recommendations("Hybrid Recommendations", hybrid_recs)
        
        # Show individual scores
        print("\nDetailed Score Breakdown:")
        for i, rec in enumerate(hybrid_recs[:3], 1):
            print(f"\n{i}. {rec['title']}")
            if 'individual_scores' in rec:
                for method, score in rec['individual_scores'].items():
                    print(f"   {method}: {score:.3f}")
            print(f"   Combined Score: {rec['recommendation_score']:.3f}")
            
    except Exception as e:
        print(f"Error getting hybrid recommendations: {e}")


def demo_use_case_recommendations(recommender):
    """Demonstrate use case based recommendations."""
    print("\n" + "="*80)
    print("DEMO: Use Case Based Recommendations")
    print("="*80)
    
    use_cases = [
        ('gaming', 8000),
        ('student', 3000),
        ('work', 6000),
        ('creative', 7000),
        ('travel', 4000)
    ]
    
    for use_case, budget in use_cases:
        print(f"\n{use_case.title()} Laptops (Budget: RM {budget})")
        print("-" * 60)
        
        try:
            recs = recommender.get_recommendations_by_use_case(use_case, budget, 3)
            if recs:
                for i, rec in enumerate(recs, 1):
                    print(f"{i}. {rec['title']}")
                    print(f"   Brand: {rec['brand']}, Price: RM {rec['price_myr']:.2f}")
                    print(f"   Rating: {rec['rating']:.1f}, Score: {rec['recommendation_score']:.3f}")
            else:
                print("No recommendations found within budget.")
        except Exception as e:
            print(f"Error getting {use_case} recommendations: {e}")


def demo_similar_laptops(recommender):
    """Demonstrate finding similar laptops."""
    print("\n" + "="*80)
    print("DEMO: Finding Similar Laptops")
    print("="*80)
    
    # Get a sample laptop ID from the dataset
    if recommender.df_laptop is not None and len(recommender.df_laptop) > 0:
        sample_laptop_id = recommender.df_laptop.iloc[0]['asin']
        sample_title = recommender.df_laptop.iloc[0].get('title_y_clean', 'Sample Laptop')
        
        print(f"\nFinding laptops similar to: {sample_title}")
        print(f"Laptop ID: {sample_laptop_id}")
        
        try:
            # Content-based similarity
            similar_content = recommender.find_similar_laptops(
                sample_laptop_id, method='content_based', n_recommendations=3
            )
            print_recommendations("Content-Based Similar Laptops", similar_content, 3)
            
        except Exception as e:
            print(f"Error finding similar laptops: {e}")
    else:
        print("No laptop data available for similarity search.")


def demo_system_capabilities(recommender):
    """Demonstrate system capabilities and summary."""
    print("\n" + "="*80)
    print("DEMO: System Capabilities & Summary")
    print("="*80)
    
    try:
        # Get system summary
        summary = recommender.get_system_summary()
        
        print("\nSystem Status:")
        print(f"  Status: {summary['system_status']}")
        print(f"  Content-Based Engine: {'✓ Active' if summary['engines_status']['content_based'] else '✗ Inactive'}")
        print(f"  Collaborative Engine: {'✓ Active' if summary['engines_status']['collaborative'] else '✗ Inactive'}")
        
        print("\nData Information:")
        print(f"  Laptop Records: {summary['data_info']['laptop_records']:,}")
        print(f"  Rating Records: {summary['data_info']['rating_records']:,}")
        print(f"  Unique Users: {summary['data_info']['unique_users']:,}")
        print(f"  Unique Laptops: {summary['data_info']['unique_laptops']:,}")
        
        print("\nConfiguration:")
        print(f"  Max Recommendations: {summary['configuration']['system']['max_recommendations']}")
        print(f"  Content-Based Weight: {summary['configuration']['hybrid']['content_based_weight']}")
        print(f"  Collaborative Weight: {summary['configuration']['hybrid']['collaborative_weight']}")
        
        print(f"\nTimestamp: {summary['timestamp']}")
        
    except Exception as e:
        print(f"Error getting system summary: {e}")


def main():
    """Main demo function."""
    print("Laptop Recommender System - Comprehensive Demo")
    print("=" * 80)
    print("This demo showcases the capabilities of our hybrid recommendation system")
    print("combining Content-Based and Collaborative Filtering approaches.")
    print("\nLoading system and data...")
    
    try:
        # Create recommender system
        recommender = create_laptop_recommender_system()
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df_laptop, df_rating = recommender.load_and_preprocess_data()
        
        # Initialize recommendation engines
        print("Initializing recommendation engines...")
        recommender.initialize_recommendation_engines()
        
        print("\nSystem ready! Starting demonstrations...")
        
        # Run all demos
        demo_system_capabilities(recommender)
        demo_content_based_filtering(recommender)
        demo_collaborative_filtering(recommender)
        demo_hybrid_recommendations(recommender)
        demo_use_case_recommendations(recommender)
        demo_similar_laptops(recommender)
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nThe Laptop Recommender System demonstrates:")
        print("✓ Content-Based Filtering using TF-IDF and cosine similarity")
        print("✓ Collaborative Filtering with user-based, item-based, and matrix factorization")
        print("✓ Hybrid recommendations combining multiple approaches")
        print("✓ Use case specific recommendations")
        print("✓ Similar laptop discovery")
        print("✓ Comprehensive system monitoring and configuration")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        logger.error(f"Demo error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
