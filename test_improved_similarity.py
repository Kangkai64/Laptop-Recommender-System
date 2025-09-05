#!/usr/bin/env python3
"""
Test script for improved similarity system with benchmark integration.

This script tests the new specification-focused similarity calculation
and demonstrates how similar laptops now include benchmark data.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Laptop_Recommender_System import LaptopRecommenderSystem

def test_improved_similarity():
    """Test the improved similarity system with benchmark data."""
    print("=" * 70)
    print("TESTING IMPROVED SIMILARITY SYSTEM WITH BENCHMARK INTEGRATION")
    print("=" * 70)
    
    try:
        # Initialize recommender system
        print("🔄 Initializing recommender system...")
        recommender = LaptopRecommenderSystem()
        
        # Load data (will use cache if available)
        print("📊 Loading data...")
        df_laptop, df_rating = recommender.load_and_preprocess_data()
        
        # Initialize recommendation engines
        print("⚙️ Initializing recommendation engines...")
        recommender.initialize_recommendation_engines()
        
        print(f"✅ System initialized successfully!")
        print(f"   Laptop records: {len(df_laptop)}")
        print(f"   Rating records: {len(df_rating)}")
        
        # Check if benchmark data is available
        benchmark_columns = ['cpu_benchmark_score', 'gpu_benchmark_score', 'total_benchmark_score']
        available_benchmarks = [col for col in benchmark_columns if col in df_laptop.columns]
        
        if available_benchmarks:
            print(f"✅ Benchmark data available: {available_benchmarks}")
            
            # Show benchmark statistics
            for col in available_benchmarks:
                non_zero_count = (df_laptop[col] > 0).sum()
                print(f"   {col}: {non_zero_count}/{len(df_laptop)} laptops have data")
        else:
            print("⚠️ No benchmark data found - similarity will be based on basic specifications")
        
        # Test similar laptop recommendations
        print("\n🔍 Testing similar laptop recommendations...")
        
        # Find a laptop with benchmark data if available
        test_laptop_id = None
        if available_benchmarks:
            # Find a laptop with benchmark data
            for col in available_benchmarks:
                laptops_with_benchmarks = df_laptop[df_laptop[col] > 0]
                if not laptops_with_benchmarks.empty:
                    test_laptop_id = laptops_with_benchmarks.iloc[0]['laptop_id']
                    break
        
        if test_laptop_id is None:
            # Use any laptop
            test_laptop_id = df_laptop.iloc[0]['laptop_id']
        
        print(f"   Testing with laptop ID: {test_laptop_id}")
        
        # Get laptop details
        laptop_data = df_laptop[df_laptop['laptop_id'] == test_laptop_id].iloc[0]
        print(f"   Laptop: {laptop_data.get('title_y', 'Unknown')[:50]}...")
        print(f"   Brand: {laptop_data.get('brand', 'Unknown')}")
        
        if 'cpu_benchmark_score' in laptop_data and laptop_data['cpu_benchmark_score'] > 0:
            print(f"   CPU Benchmark: {laptop_data['cpu_benchmark_score']}")
        if 'gpu_benchmark_score' in laptop_data and laptop_data['gpu_benchmark_score'] > 0:
            print(f"   GPU Benchmark: {laptop_data['gpu_benchmark_score']}")
        
        # Get similar laptops using specification-focused similarity
        print("\n📋 Finding similar laptops with specification-focused similarity...")
        similar_laptops = recommender.find_similar_laptops(
            laptop_id=test_laptop_id,
            n_recommendations=5,
            use_spec_similarity=True
        )
        
        if similar_laptops:
            print(f"✅ Found {len(similar_laptops)} similar laptops:")
            print()
            
            for i, similar in enumerate(similar_laptops, 1):
                print(f"   {i}. {similar.get('title_y', 'Unknown')[:60]}...")
                print(f"      Brand: {similar.get('brand', 'Unknown')}")
                print(f"      Price: RM {similar.get('price_myr', 0):.2f}")
                
                # Show specifications
                specs = []
                if similar.get('ram_gb', 0) > 0:
                    specs.append(f"{similar['ram_gb']:.0f}GB RAM")
                if similar.get('storage_gb', 0) > 0:
                    specs.append(f"{similar['storage_gb']:.0f}GB Storage")
                if similar.get('screen_size_inches', 0) > 0:
                    specs.append(f"{similar['screen_size_inches']:.1f}\" Screen")
                
                if specs:
                    print(f"      Specs: {', '.join(specs)}")
                
                # Show benchmark scores
                benchmarks = []
                if similar.get('cpu_benchmark_score', 0) > 0:
                    benchmarks.append(f"CPU: {similar['cpu_benchmark_score']:.0f}")
                if similar.get('gpu_benchmark_score', 0) > 0:
                    benchmarks.append(f"GPU: {similar['gpu_benchmark_score']:.0f}")
                if similar.get('performance_tier', 'Unknown') != 'Unknown':
                    benchmarks.append(f"Tier: {similar['performance_tier']}")
                
                if benchmarks:
                    print(f"      Benchmarks: {', '.join(benchmarks)}")
                
                # Show similarity score
                if similar.get('similarity_score'):
                    print(f"      Similarity: {similar['similarity_score']:.3f}")
                
                print()
        else:
            print("❌ No similar laptops found")
        
        # Test with standard similarity for comparison
        print("📋 Testing with standard similarity for comparison...")
        standard_similar = recommender.find_similar_laptops(
            laptop_id=test_laptop_id,
            n_recommendations=3,
            use_spec_similarity=False
        )
        
        if standard_similar:
            print(f"✅ Found {len(standard_similar)} laptops with standard similarity:")
            for i, similar in enumerate(standard_similar, 1):
                print(f"   {i}. {similar.get('title_y', 'Unknown')[:50]}...")
                if similar.get('similarity_score'):
                    print(f"      Similarity: {similar['similarity_score']:.3f}")
        
        print("\n🎉 Test completed successfully!")
        print("\n💡 Key improvements:")
        print("   - Similar laptops now include detailed specifications")
        print("   - Benchmark scores are displayed for performance comparison")
        print("   - Specification-focused similarity gives better matches")
        print("   - Similar laptops are more relevant to the reference laptop")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_similarity()
