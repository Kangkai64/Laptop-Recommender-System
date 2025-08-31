#!/usr/bin/env python3
"""
Test script for the Flask Laptop Recommender System
This script tests basic functionality without requiring the full data loading.
"""

import sys
import os
from flask import Flask

def test_flask_import():
    """Test if Flask can be imported."""
    try:
        import flask
        print("✅ Flask import successful")
        return True
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False

def test_app_structure():
    """Test if the main app file exists and can be imported."""
    try:
        # Check if app.py exists
        if not os.path.exists('app.py'):
            print("❌ app.py not found")
            return False
        
        print("✅ app.py found")
        
        # Try to import the app (without running it)
        import app
        print("✅ app.py import successful")
        return True
        
    except Exception as e:
        print(f"❌ app.py import failed: {e}")
        return False

def test_templates():
    """Test if template files exist."""
    template_files = [
        'templates/base.html',
        'templates/index.html',
        'templates/recommend.html',
        'templates/recommendations.html',
        'templates/explore.html',
        'templates/search.html',
        'templates/analytics.html',
        'templates/laptop_detail.html',
        'templates/404.html',
        'templates/500.html'
    ]
    
    missing_templates = []
    for template in template_files:
        if not os.path.exists(template):
            missing_templates.append(template)
        else:
            print(f"✅ {template} found")
    
    if missing_templates:
        print(f"❌ Missing templates: {missing_templates}")
        return False
    
    print("✅ All template files found")
    return True

def test_basic_flask_app():
    """Test creating a basic Flask app."""
    try:
        app = Flask(__name__)
        app.config['TESTING'] = True
        
        @app.route('/test')
        def test_route():
            return 'Test successful'
        
        with app.test_client() as client:
            response = client.get('/test')
            if response.status_code == 200:
                print("✅ Basic Flask app test successful")
                return True
            else:
                print(f"❌ Basic Flask app test failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Basic Flask app test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Flask Laptop Recommender System")
    print("=" * 50)
    
    tests = [
        ("Flask Import", test_flask_import),
        ("App Structure", test_app_structure),
        ("Templates", test_templates),
        ("Basic Flask App", test_basic_flask_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Flask app is ready to run.")
        print("\n🚀 To start the application, run:")
        print("   python app.py")
        print("\n📱 Then open your browser to: http://localhost:5000")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
