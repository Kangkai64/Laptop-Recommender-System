#!/usr/bin/env python3
"""
Setup script for Laptop Recommender System
Combines dependencies from requirements_collaborative_filtering.txt and requirements_content_based_filtering.txt
"""

from setuptools import setup, find_packages
import os

def read_requirements(filename):
    """Read requirements from a file and return a list of dependencies."""
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Remove inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    requirements.append(line)
    return requirements

# Read requirements from both files
collaborative_requirements = read_requirements('requirements_collaborative_filtering.txt')
content_based_requirements = read_requirements('requirements_content_based_filtering.txt')

# Combine and deduplicate requirements
all_requirements = list(set(collaborative_requirements + content_based_requirements))

# Add Flask and web-related dependencies
web_requirements = [
    'flask>=2.3.0',
    'jinja2>=3.1.0',
    'werkzeug>=2.3.0',
    'itsdangerous>=2.1.0',
    'click>=8.1.0',
    'blinker>=1.6.0',
    'markupsafe>=2.1.0'
]

# Combine all requirements
all_requirements.extend(web_requirements)
all_requirements = list(set(all_requirements))  # Remove duplicates

# Sort requirements for better readability
all_requirements.sort()

setup(
    name="laptop-recommender-system",
    version="1.0.0",
    description="Intelligent Laptop Recommendation System using Content-Based and Collaborative Filtering",
    long_description="A comprehensive laptop recommendation system that combines content-based and collaborative filtering approaches to provide personalized laptop recommendations based on user preferences and specifications.",
    author="Laptop Recommender System Team",
    author_email="",
    url="https://github.com/Kangkai64/Laptop-Recommender-System",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=all_requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
            'memory-profiler>=0.60.0',
            'line-profiler>=3.5.0'
        ],
        'gpu': [
            'torch>=2.0.0'
        ],
        'viz': [
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0'
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="recommendation-system, machine-learning, collaborative-filtering, content-based-filtering, laptop-recommendations",
    project_urls={
        "Bug Reports": "https://github.com/Kangkai64/Laptop-Recommender-System/issues",
        "Source": "https://github.com/Kangkai64/Laptop-Recommender-System",
    },
    entry_points={
        'console_scripts': [
            'laptop-recommender=app:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)