"""
Simple Test Script for Laptop Recommender System

This script tests the basic functionality of the system components
without requiring full data loading.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        import collaborative_filtering
        print("✓ Collaborative filtering module imported")
    except Exception as e:
        print(f"✗ Collaborative filtering import failed: {e}")
        return False
    
    try:
        import content_based_filtering
        print("✓ Content-based filtering module imported")
    except Exception as e:
        print(f"✗ Content-based filtering import failed: {e}")
        return False
    
    try:
        import Laptop_Recommender_System
        print("✓ Main system module imported")
    except Exception as e:
        print(f"✗ Main system import failed: {e}")
        return False
    
    try:
        import data_preprocessing
        print("✓ Data preprocessing module imported")
    except Exception as e:
        print(f"✗ Data preprocessing import failed: {e}")
        return False
    
    return True

def test_class_creation():
    """Test that classes can be instantiated."""
    print("\nTesting class creation...")
    
    try:
        from collaborative_filtering import CollaborativeFiltering
        from content_based_filtering import ContentBasedFiltering
        from Laptop_Recommender_System import LaptopRecommenderSystem
        
        # Create empty DataFrames for testing
        import pandas as pd
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        
        # Test collaborative filtering
        cf = CollaborativeFiltering(test_df, test_df)
        print("✓ CollaborativeFiltering class created")
        
        # Test content-based filtering
        cbf = ContentBasedFiltering(test_df, test_df)
        print("✓ ContentBasedFiltering class created")
        
        # Test main system
        lrs = LaptopRecommenderSystem()
        print("✓ LaptopRecommenderSystem class created")
        
        return True
        
    except Exception as e:
        print(f"✗ Class creation failed: {e}")
        traceback.print_exc()
        return False

def test_factory_functions():
    """Test factory functions."""
    print("\nTesting factory functions...")
    
    try:
        from collaborative_filtering import create_collaborative_filtering
        from content_based_filtering import create_content_based_filtering
        from Laptop_Recommender_System import create_laptop_recommender_system
        
        # Create empty DataFrames for testing
        import pandas as pd
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        
        # Test factory functions
        cf = create_collaborative_filtering(test_df, test_df)
        print("✓ create_collaborative_filtering works")
        
        cbf = create_content_based_filtering(test_df, test_df)
        print("✓ create_content_based_filtering works")
        
        lrs = create_laptop_recommender_system()
        print("✓ create_laptop_recommender_system works")
        
        return True
        
    except Exception as e:
        print(f"✗ Factory functions failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Laptop Recommender System - Component Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_class_creation,
        test_factory_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            traceback.print_exc()
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System components are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
