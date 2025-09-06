#!/usr/bin/env python3
"""
Demonstration of Data Caching Performance Improvement

This script demonstrates the performance difference between:
1. First run (with preprocessing)
2. Subsequent runs (with cached data)

Run this script to see the performance improvement in action.
"""

import time
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Laptop_Recommender_System import LaptopRecommenderSystem

def measure_loading_time():
    """Measure the time taken to load and initialize the system."""
    start_time = time.time()
    
    try:
        # Initialize recommender system
        recommender = LaptopRecommenderSystem()
        
        # Load and preprocess data
        df_laptop, df_rating = recommender.load_and_preprocess_data()
        
        # Initialize recommendation engines
        recommender.initialize_recommendation_engines()
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        return loading_time, len(df_laptop), len(df_rating)
        
    except Exception as e:
        end_time = time.time()
        loading_time = end_time - start_time
        print(f"Error during loading: {e}")
        return loading_time, 0, 0

def main():
    """Main demonstration function."""
    print("=" * 70)
    print("LAPTOP RECOMMENDER SYSTEM - CACHING PERFORMANCE DEMO")
    print("=" * 70)
    
    print("\nThis demo shows the performance improvement from data caching.")
    print("The first run will preprocess data and create cache files.")
    print("Subsequent runs will load from cache (much faster).\n")
    
    # First run
    print("🔄 First run (with preprocessing)...")
    start_time = time.time()
    loading_time, laptop_count, rating_count = measure_loading_time()
    
    if laptop_count > 0 and rating_count > 0:
        print(f"✅ First run completed successfully!")
        print(f"   Loading time: {loading_time:.2f} seconds")
        print(f"   Data loaded: {laptop_count} laptops, {rating_count} ratings")
        print(f"   Cache files created in data/cache/")
    else:
        print(f"❌ First run failed after {loading_time:.2f} seconds")
        return
    
    print("\n" + "-" * 50)
    
    # Second run (should use cache)
    print("⚡ Second run (with cached data)...")
    start_time = time.time()
    loading_time2, laptop_count2, rating_count2 = measure_loading_time()
    
    if laptop_count2 > 0 and rating_count2 > 0:
        print(f"✅ Second run completed successfully!")
        print(f"   Loading time: {loading_time2:.2f} seconds")
        print(f"   Data loaded: {laptop_count2} laptops, {rating_count2} ratings")
        print(f"   Used cached data from data/cache/")
        
        # Calculate improvement
        if loading_time > 0:
            improvement = ((loading_time - loading_time2) / loading_time) * 100
            speedup = loading_time / loading_time2 if loading_time2 > 0 else float('inf')
            
            print(f"\n📊 Performance Improvement:")
            print(f"   Time saved: {loading_time - loading_time2:.2f} seconds")
            print(f"   Speed improvement: {improvement:.1f}%")
            print(f"   Speedup factor: {speedup:.1f}x faster")
            
            if improvement > 50:
                print(f"   🎉 Excellent! Caching provides significant performance improvement!")
            elif improvement > 20:
                print(f"   👍 Good! Caching provides noticeable performance improvement!")
            else:
                print(f"   ⚠️  Caching provides some improvement, but may need optimization.")
    else:
        print(f"❌ Second run failed after {loading_time2:.2f} seconds")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETED")
    print("=" * 70)
    
    print("\n💡 Tips:")
    print("   - Run 'python preprocess_data.py' to preprocess data once")
    print("   - Run 'python app.py' to start the web application")
    print("   - Check 'data/cache/' directory for cached files")
    print("   - Use 'python preprocess_data.py --status' to check cache status")

if __name__ == "__main__":
    main()
