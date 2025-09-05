#!/usr/bin/env python3
"""
Data Preprocessing Utility Script

This script allows you to preprocess the laptop data once and save it to cache files.
This eliminates the need to reprocess data every time the web application starts.

Usage:
    python preprocess_data.py                    # Preprocess and cache data
    python preprocess_data.py --force            # Force reprocessing even if cache exists
    python preprocess_data.py --clear            # Clear existing cache
    python preprocess_data.py --status           # Check cache status
"""

import argparse
import sys
import os
from datetime import datetime
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import LaptopDataPreprocessor
from Laptop_Recommender_System import LaptopRecommenderSystem

def preprocess_and_cache(force: bool = False):
    """Preprocess data and save to cache."""
    print("=" * 60)
    print("LAPTOP DATA PREPROCESSING UTILITY")
    print("=" * 60)
    
    try:
        # Initialize preprocessor
        print("Initializing data preprocessor...")
        preprocessor = LaptopDataPreprocessor()
        
        # Check if cache exists and is fresh
        if not force:
            cached_data = preprocessor.load_cached_data()
            if cached_data is not None:
                print("✅ Fresh cached data found! No preprocessing needed.")
                print(f"   Cached data: {len(cached_data[0])} laptops, {len(cached_data[1])} ratings")
                return True
        
        print("🔄 Starting data preprocessing...")
        print("   This may take several minutes for the first run...")
        
        # Run preprocessing pipeline
        df_laptop, df_rating = preprocessor.preprocess_separated_pipeline(force_reprocess=force)
        
        print("✅ Data preprocessing completed successfully!")
        print(f"   Processed data: {len(df_laptop)} laptops, {len(df_rating)} ratings")
        
        # Test the recommender system with cached data
        print("\n🧪 Testing recommender system with cached data...")
        recommender = LaptopRecommenderSystem()
        recommender.load_and_preprocess_data()
        recommender.initialize_recommendation_engines()
        
        print("✅ Recommender system test successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        return False

def clear_cache():
    """Clear all cached data."""
    print("=" * 60)
    print("CLEARING CACHE")
    print("=" * 60)
    
    try:
        preprocessor = LaptopDataPreprocessor()
        preprocessor.clear_cache()
        print("✅ Cache cleared successfully!")
        return True
    except Exception as e:
        print(f"❌ Error clearing cache: {str(e)}")
        return False

def check_cache_status():
    """Check the status of cached data."""
    print("=" * 60)
    print("CACHE STATUS")
    print("=" * 60)
    
    try:
        preprocessor = LaptopDataPreprocessor()
        cached_data = preprocessor.load_cached_data()
        
        if cached_data is None:
            print("❌ No valid cache found")
            print("   Run 'python preprocess_data.py' to create cache")
            return False
        
        df_laptop, df_rating = cached_data
        
        # Load metadata
        metadata_path = "data/cache/cache_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            created_at = datetime.fromisoformat(metadata['created_at'])
            age_hours = (datetime.now() - created_at).total_seconds() / 3600
            
            print("✅ Valid cache found!")
            print(f"   Created: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Age: {age_hours:.1f} hours")
            print(f"   Laptop records: {len(df_laptop)}")
            print(f"   Rating records: {len(df_rating)}")
            print(f"   Laptop columns: {len(df_laptop.columns)}")
            print(f"   Rating columns: {len(df_rating.columns)}")
            
            # Check if cache is fresh (less than 24 hours old)
            if age_hours < 24:
                print("   Status: ✅ Fresh (less than 24 hours old)")
            else:
                print("   Status: ⚠️  Stale (older than 24 hours)")
                print("   Consider running 'python preprocess_data.py --force' to refresh")
            
            return True
        else:
            print("❌ Cache metadata not found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking cache status: {str(e)}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Laptop Data Preprocessing Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_data.py              # Preprocess and cache data
  python preprocess_data.py --force      # Force reprocessing
  python preprocess_data.py --clear      # Clear cache
  python preprocess_data.py --status     # Check cache status
        """
    )
    
    parser.add_argument(
        '--force', 
        action='store_true', 
        help='Force reprocessing even if cache exists'
    )
    
    parser.add_argument(
        '--clear', 
        action='store_true', 
        help='Clear existing cache'
    )
    
    parser.add_argument(
        '--status', 
        action='store_true', 
        help='Check cache status'
    )
    
    args = parser.parse_args()
    
    # Handle different commands
    if args.clear:
        success = clear_cache()
    elif args.status:
        success = check_cache_status()
    else:
        success = preprocess_and_cache(force=args.force)
    
    if success:
        print("\n🎉 Operation completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
