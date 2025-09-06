#!/usr/bin/env python3
"""
Dependency Installation Script for Laptop Recommender System

This script installs all required dependencies for the laptop recommender system.
Run this script if you encounter import errors when running the training notebook.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Main installation function."""
    print("🔧 Laptop Recommender System - Dependency Installer")
    print("=" * 60)
    
    # List of required packages
    required_packages = [
        "datasets>=2.14.0",
        "pandas>=2.0.0", 
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyarrow>=10.0.0",
        "transformers>=4.30.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "flask>=2.3.0"
    ]
    
    print("📋 Installing required packages...")
    print("This may take several minutes...")
    print()
    
    success_count = 0
    total_packages = len(required_packages)
    
    for package in required_packages:
        if install_package(package):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"📊 Installation Summary:")
    print(f"   ✅ Successfully installed: {success_count}/{total_packages} packages")
    print(f"   ❌ Failed installations: {total_packages - success_count}/{total_packages} packages")
    
    if success_count == total_packages:
        print("\n🎉 All dependencies installed successfully!")
        print("You can now run the training notebook without issues.")
    else:
        print(f"\n⚠️  {total_packages - success_count} packages failed to install.")
        print("You may need to install them manually or check your Python environment.")
    
    print("\n💡 Next steps:")
    print("   1. Run the training notebook: laptop_recommender_training.ipynb")
    print("   2. Or run the preprocessing script: python preprocess_data.py")
    print("   3. Or start the web application: python app.py")

if __name__ == "__main__":
    main()
