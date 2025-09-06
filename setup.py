#!/usr/bin/env python3
"""
Setup script for Laptop Recommender System

This script helps set up the complete laptop recommender system including:
- Main application dependencies (requirements.txt)
- Collaborative filtering dependencies (requirements_collaborative_filtering.txt)
- Content-based filtering dependencies (requirements_content_based_filtering.txt)

The script installs all dependencies and verifies the installation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_data_file():
    """Check if the data file exists."""
    print("\n📁 Checking data file...")
    data_path = Path("data/Cleaned_Laptop_data.csv")
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data file is in the correct location.")
        return False
    print(f"✅ Data file found: {data_path}")
    return True

def install_dependencies():
    """Install required dependencies from all requirements files."""
    print("\n📦 Installing dependencies...")
    
    requirements_files = [
        "requirements.txt",
        "requirements_collaborative_filtering.txt", 
        "requirements_content_based_filtering.txt"
    ]
    
    for req_file in requirements_files:
        if not os.path.exists(req_file):
            print(f"⚠️ Warning: {req_file} not found, skipping...")
            continue
            
        print(f"📦 Installing from {req_file}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print(f"✅ Dependencies from {req_file} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies from {req_file}: {e}")
            return False
    
    print("✅ All dependencies installed successfully")
    return True

def test_imports():
    """Test if all required modules can be imported."""
    print("\n🔍 Testing imports...")
    
    # Core modules from main requirements
    core_modules = [
        "pandas",
        "numpy", 
        "sklearn",
        "mcp",
        "flask",
        "fastapi",
        "uvicorn"
    ]
    
    # Additional modules from collaborative filtering requirements
    collaborative_modules = [
        "scipy",
        "datasets"
    ]
    
    # Additional modules from content-based filtering requirements
    content_modules = [
        "nltk",
        "textblob",
        "psutil",
        "matplotlib",
        "seaborn",
        "tqdm",
        "joblib"
    ]
    
    all_modules = core_modules + collaborative_modules + content_modules
    failed_imports = []
    
    print("📦 Testing core modules...")
    for module in core_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    print("\n📦 Testing collaborative filtering modules...")
    for module in collaborative_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"⚠️ {module} (optional)")
    
    print("\n📦 Testing content-based filtering modules...")
    for module in content_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"⚠️ {module} (optional)")
    
    if failed_imports:
        print(f"\n❌ Failed to import core modules: {', '.join(failed_imports)}")
        return False
    
    print("\n✅ All core modules imported successfully")
    print("ℹ️ Some optional modules may not be available, but core functionality should work")
    return True

def create_mcp_config():
    """Create MCP configuration file if it doesn't exist."""
    print("\n⚙️ Setting up MCP configuration...")
    config_path = Path("mcp_config.json")
    
    if config_path.exists():
        print("✅ MCP configuration already exists")
        return True
    
    config_content = {
        "mcpServers": {
            "laptop-recommender": {
                "command": "python",
                "args": ["laptop_recommender_mcp.py"],
                "env": {
                    "PYTHONPATH": "."
                }
            }
        }
    }
    
    try:
        import json
        with open(config_path, 'w') as f:
            json.dump(config_content, f, indent=2)
        print("✅ MCP configuration created")
        return True
    except Exception as e:
        print(f"❌ Failed to create MCP configuration: {e}")
        return False

def show_next_steps():
    """Show next steps for using the laptop recommender system."""
    print("\n🎯 Setup Complete!")
    print("=" * 50)
    print("\n📖 Next Steps:")
    print("1. Run the Flask web application:")
    print("   python app.py")
    print("\n2. Run the MCP server (optional):")
    print("   python laptop_recommender_mcp.py")
    print("\n3. Test the recommendation system:")
    print("   python demo_recommender_system.py")
    print("   python demo_evaluation_system.py")
    print("\n4. Available components:")
    print("   - Web Application: Interactive web interface")
    print("   - Collaborative Filtering: User-based recommendations")
    print("   - Content-Based Filtering: Feature-based recommendations")
    print("   - MCP Server: AI assistant integration")
    print("   - Evaluation System: Performance testing")
    print("\n5. Web interface features:")
    print("   - Browse and search laptops")
    print("   - Get personalized recommendations")
    print("   - Compare laptops side-by-side")
    print("   - View detailed specifications")
    print("\n📚 For more information, see README.md and other documentation files")

def main():
    """Main setup function."""
    print("🚀 Laptop Recommender System Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_data_file():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup failed. Some modules could not be imported.")
        sys.exit(1)
    
    # Create MCP configuration
    if not create_mcp_config():
        print("\n⚠️ Warning: MCP configuration could not be created.")
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()
