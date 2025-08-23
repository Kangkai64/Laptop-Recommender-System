#!/usr/bin/env python3
"""
Setup script for Laptop Recommender MCP Server

This script helps set up the MCP server and verify the installation.
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
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("\n🔍 Testing imports...")
    required_modules = [
        "pandas",
        "numpy", 
        "sklearn",
        "mcp"
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All modules imported successfully")
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
    """Show next steps for using the MCP server."""
    print("\n🎯 Setup Complete!")
    print("=" * 50)
    print("\n📖 Next Steps:")
    print("1. Run the MCP server:")
    print("   python laptop_recommender_mcp.py")
    print("\n2. Configure your AI assistant:")
    print("   - Add the mcp_config.json to your MCP settings")
    print("   - Or use the server directly with MCP-compatible tools")
    print("\n3. Test the server:")
    print("   python test_mcp_server.py")
    print("\n4. Available tools:")
    print("   - get_recommendations: Get personalized laptop suggestions")
    print("   - get_similar_laptops: Find similar laptops")
    print("   - get_laptop_details: Get detailed laptop information")
    print("   - get_statistics: View dataset statistics")
    print("   - search_laptops: Search by brand, processor, etc.")
    print("\n📚 For more information, see README.md")

def main():
    """Main setup function."""
    print("🚀 Laptop Recommender MCP Server Setup")
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
