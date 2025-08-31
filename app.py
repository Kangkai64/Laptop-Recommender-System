"""
Flask Web Application for Laptop Recommender System
Based on SAUE (System Active, User Engage) approach

This application provides an interactive web interface for users to:
1. Input their preferences and requirements
2. Receive personalized laptop recommendations
3. Explore similar laptops
4. View detailed laptop information
5. Get system insights and analytics

Author: Laptop Recommender System Team
License: MIT
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import json
from typing import Dict, List, Optional, Any

# Import our recommendation system
from Laptop_Recommender_System import LaptopRecommenderSystem
from data_preprocessing import LaptopDataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'laptop_recommender_secret_key_2024'

# Global variables
recommender_system = None
df_laptop = None
df_rating = None

def initialize_system():
    """Initialize the recommendation system and load data."""
    global recommender_system, df_laptop, df_rating
    
    try:
        logger.info("Initializing Laptop Recommender System...")
        
        # Initialize the main recommender system
        recommender_system = LaptopRecommenderSystem()
        
        # Load and preprocess data
        preprocessor = LaptopDataPreprocessor()
        df_laptop, df_rating = preprocessor.preprocess_separated_pipeline()
        
        # Add original brand column back for display purposes
        if 'brand_encoded' in df_laptop.columns and 'brand' not in df_laptop.columns:
            # Get the original brand data from the preprocessor
            original_data = preprocessor.df
            if 'brand' in original_data.columns:
                # Map back to laptop dataframe using asin
                brand_mapping = original_data[['asin', 'brand']].drop_duplicates(subset=['asin'])
                df_laptop = df_laptop.merge(brand_mapping, on='asin', how='left')
        
        # Set the data in the recommender system
        recommender_system.df_laptop = df_laptop
        recommender_system.df_rating = df_rating
        
        # Initialize the recommendation algorithms
        recommender_system.initialize_recommendation_engines()
        
        logger.info("System initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return False

@app.route('/')
def index():
    """Main landing page with system overview and quick start options."""
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """Main recommendation interface where users can input preferences."""
    if request.method == 'POST':
        # Get user preferences from form
        preferences = {
            'budget_min': float(request.form.get('budget_min', 0)),
            'budget_max': float(request.form.get('budget_max', 50000)),
            'brand': request.form.get('brand', ''),
            'processor_type': request.form.get('processor_type', ''),
            'ram_min': int(request.form.get('ram_min', 4)),
            'storage_min': int(request.form.get('storage_min', 256)),
            'use_case': request.form.get('use_case', 'general'),
            'priority': request.form.get('priority', 'performance')
        }
        
        # Store preferences in session
        session['user_preferences'] = preferences
        
        # Get recommendations
        try:
            recommendations = get_recommendations(preferences)
            return render_template('recommendations.html', 
                                 recommendations=recommendations,
                                 preferences=preferences)
        except Exception as e:
            flash(f'Error getting recommendations: {str(e)}', 'error')
            return render_template('recommend.html', error=str(e))
    
    return render_template('recommend.html')

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """API endpoint for getting recommendations via AJAX."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert preferences to the format expected by the recommendation system
        preferences = {
            'budget_min': float(data.get('budget_min', 0)),
            'budget_max': float(data.get('budget_max', 50000)),
            'brand': data.get('brand', ''),
            'processor_type': data.get('processor_type', ''),
            'ram_min': int(data.get('ram_min', 4)),
            'storage_min': int(data.get('storage_min', 256)),
            'use_case': data.get('use_case', 'general'),
            'priority': data.get('priority', 'performance')
        }
        
        # Get recommendations
        recommendations = get_recommendations(preferences)
        
        # Convert numpy types to Python types for JSON serialization
        serializable_recommendations = []
        for rec in recommendations:
            serializable_rec = {}
            for key, value in rec.items():
                if hasattr(value, 'item'):  # numpy scalar
                    serializable_rec[key] = value.item()
                elif isinstance(value, (list, dict)):
                    serializable_rec[key] = value
                else:
                    serializable_rec[key] = value
            serializable_recommendations.append(serializable_rec)
        
        return jsonify({
            'success': True,
            'recommendations': serializable_recommendations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_recommendations(preferences: Dict) -> List[Dict]:
    """Get personalized recommendations based on user preferences."""
    if not recommender_system:
        raise Exception("Recommendation system not initialized")
    
    # Convert preferences to system format
    query = {
        'budget_range': (preferences['budget_min'], preferences['budget_max']),
        'brand_preference': preferences['brand'] if preferences['brand'] else None,
        'processor_preference': preferences['processor_type'] if preferences['processor_type'] else None,
        'min_ram': preferences['ram_min'],
        'min_storage': preferences['storage_min'],
        'use_case': preferences['use_case'],
        'priority': preferences['priority']
    }
    
    # Try different recommendation methods
    try:
        # First try content-based recommendations
        recommendations = recommender_system.get_content_based_recommendations(
            preferences=query,
            n_recommendations=10
        )
        return recommendations
    except Exception as e:
        try:
            # Fallback to use case recommendations
            use_case = preferences.get('use_case', 'general')
            budget = preferences.get('budget_max', 10000)
            recommendations = recommender_system.get_recommendations_by_use_case(
                use_case=use_case,
                budget=budget
            )
            return recommendations
        except Exception as e2:
            # Final fallback: return sample laptops filtered by budget
            logger.warning(f"Recommendation methods failed, using fallback: {e2}")
            return get_fallback_recommendations(preferences)

def get_fallback_recommendations(preferences: Dict) -> List[Dict]:
    """Fallback method to get recommendations when main methods fail."""
    global df_laptop
    
    if df_laptop is None:
        return []
    
    # Filter by budget
    budget_max = preferences.get('budget_max', 10000)
    filtered_df = df_laptop[df_laptop['price_myr'] <= budget_max]
    
    # Get sample laptops
    sample_laptops = filtered_df.sample(min(10, len(filtered_df)))
    
    # Format for templates
    results = []
    for _, laptop in sample_laptops.iterrows():
        laptop_dict = laptop.to_dict()
        
        # Map column names to what templates expect
        laptop_dict['title_y'] = laptop_dict.get('title_y_clean', 'Unknown Title')
        laptop_dict['features'] = laptop_dict.get('features_clean', '')
        
        # Ensure brand is available
        if 'brand' not in laptop_dict and 'brand_encoded' in laptop_dict:
            laptop_dict['brand'] = f"Brand_{laptop_dict['brand_encoded']}"
        
        # Add recommendation score
        laptop_dict['recommendation_score'] = 0.8  # Default score
        laptop_dict['method'] = 'fallback'
        
        results.append(laptop_dict)
    
    return results

@app.route('/explore')
def explore():
    """Explore page to browse laptops and get insights."""
    if df_laptop is None:
        flash('Data not loaded. Please try again.', 'error')
        return redirect(url_for('index'))
    
    # Get some statistics
    stats = {
        'total_laptops': len(df_laptop),
        'brands': df_laptop['brand'].nunique() if 'brand' in df_laptop.columns else 0,
        'price_range': {
            'min': df_laptop['price_myr'].min(),
            'max': df_laptop['price_myr'].max()
        },
        'avg_rating': df_laptop['average_rating'].mean()
    }
    
    # Get sample laptops for browsing and format for templates
    sample_laptops_raw = df_laptop.sample(min(20, len(df_laptop)))
    sample_laptops = []
    
    for _, laptop in sample_laptops_raw.iterrows():
        laptop_dict = laptop.to_dict()
        
        # Map column names to what templates expect
        laptop_dict['title_y'] = laptop_dict.get('title_y_clean', 'Unknown Title')
        laptop_dict['features'] = laptop_dict.get('features_clean', '')
        
        # Ensure brand is available
        if 'brand' not in laptop_dict and 'brand_encoded' in laptop_dict:
            laptop_dict['brand'] = f"Brand_{laptop_dict['brand_encoded']}"
        
        sample_laptops.append(laptop_dict)
    
    return render_template('explore.html', stats=stats, laptops=sample_laptops)

@app.route('/laptop/<laptop_id>')
def laptop_detail(laptop_id):
    """Detailed view of a specific laptop."""
    if df_laptop is None:
        flash('Data not loaded. Please try again.', 'error')
        return redirect(url_for('index'))
    
    # Find the laptop
    laptop = df_laptop[df_laptop['asin'] == laptop_id]
    if laptop.empty:
        flash('Laptop not found.', 'error')
        return redirect(url_for('explore'))
    
    laptop = laptop.iloc[0].to_dict()
    
    # Map column names to what templates expect
    laptop['title_y'] = laptop.get('title_y_clean', laptop.get('title_y', 'Unknown Title'))
    laptop['features'] = laptop.get('features_clean', laptop.get('features', ''))
    laptop['average_rating'] = laptop.get('average_rating', 0.0)  # Ensure average_rating exists
    
    # Ensure brand is available
    if 'brand' not in laptop and 'brand_encoded' in laptop:
        laptop['brand'] = f"Brand_{laptop['brand_encoded']}"
    elif 'brand' not in laptop:
        laptop['brand'] = 'Unknown Brand'
    
    # Ensure price is available
    if 'price_myr' not in laptop:
        laptop['price_myr'] = 0.0
    
    # Get similar laptops
    try:
        similar_laptops = recommender_system.find_similar_laptops(
            laptop_id=laptop_id,
            n_recommendations=5
        )
        
        # Format similar laptops for templates
        formatted_similar = []
        for similar in similar_laptops:
            similar['title_y'] = similar.get('title_y_clean', similar.get('title_y', 'Unknown Title'))
            similar['features'] = similar.get('features_clean', similar.get('features', ''))
            similar['average_rating'] = similar.get('average_rating', 0.0)
            if 'brand' not in similar and 'brand_encoded' in similar:
                similar['brand'] = f"Brand_{similar['brand_encoded']}"
            elif 'brand' not in similar:
                similar['brand'] = 'Unknown Brand'
            if 'price_myr' not in similar:
                similar['price_myr'] = 0.0
            formatted_similar.append(similar)
        similar_laptops = formatted_similar
    except Exception as e:
        logger.warning(f"Could not get similar laptops: {e}")
        similar_laptops = []
    
    return render_template('laptop_detail.html', 
                         laptop=laptop, 
                         similar_laptops=similar_laptops)

@app.route('/search')
def search():
    """Search functionality for laptops."""
    query = request.args.get('q', '')
    brand = request.args.get('brand', '')
    price_min = request.args.get('price_min', '')
    price_max = request.args.get('price_max', '')
    
    if df_laptop is None:
        flash('Data not loaded. Please try again.', 'error')
        return redirect(url_for('index'))
    
    # Filter laptops based on search criteria
    filtered_df = df_laptop.copy()
    
    if query:
        title_mask = filtered_df['title_y_clean'].str.contains(query, case=False, na=False) if 'title_y_clean' in filtered_df.columns else False
        brand_mask = filtered_df['brand'].str.contains(query, case=False, na=False) if 'brand' in filtered_df.columns else False
        mask = title_mask | brand_mask
        filtered_df = filtered_df[mask]
    
    if brand and 'brand' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['brand'] == brand]
    
    if price_min:
        filtered_df = filtered_df[filtered_df['price_myr'] >= float(price_min)]
    
    if price_max:
        filtered_df = filtered_df[filtered_df['price_myr'] <= float(price_max)]
    
    # Format results for templates
    results = []
    for _, laptop in filtered_df.head(50).iterrows():
        laptop_dict = laptop.to_dict()
        
        # Map column names to what templates expect
        laptop_dict['title_y'] = laptop_dict.get('title_y_clean', laptop_dict.get('title_y', 'Unknown Title'))
        laptop_dict['features'] = laptop_dict.get('features_clean', laptop_dict.get('features', ''))
        laptop_dict['average_rating'] = laptop_dict.get('average_rating', 0.0)
        
        # Ensure brand is available
        if 'brand' not in laptop_dict and 'brand_encoded' in laptop_dict:
            laptop_dict['brand'] = f"Brand_{laptop_dict['brand_encoded']}"
        elif 'brand' not in laptop_dict:
            laptop_dict['brand'] = 'Unknown Brand'
        
        # Ensure price is available
        if 'price_myr' not in laptop_dict:
            laptop_dict['price_myr'] = 0.0
        
        results.append(laptop_dict)
    
    return render_template('search.html', 
                         results=results, 
                         query=query,
                         total_results=len(filtered_df))

@app.route('/analytics')
def analytics():
    """Analytics and insights page."""
    if df_laptop is None or df_rating is None:
        flash('Data not loaded. Please try again.', 'error')
        return redirect(url_for('index'))
    
    # Calculate various statistics
    analytics_data = {
        'total_laptops': int(len(df_laptop)),
        'total_reviews': int(len(df_rating)),
        'unique_users': int(df_rating['user_id_encoded'].nunique()),
        'avg_rating': float(df_laptop['average_rating'].mean()),
        'price_stats': {
            'mean': float(df_laptop['price_myr'].mean()),
            'median': float(df_laptop['price_myr'].median()),
            'std': float(df_laptop['price_myr'].std())
        },
        'top_brands': df_laptop['brand'].value_counts().head(10).to_dict() if 'brand' in df_laptop.columns else {},
        'rating_distribution': {}
    }
    
    # Create proper rating distribution (1-5 stars)
    if 'average_rating' in df_laptop.columns:
        rating_counts = {}
        for rating in range(1, 6):
            # Count laptops with ratings in each star category
            if rating == 1:
                count = len(df_laptop[(df_laptop['average_rating'] >= 1.0) & (df_laptop['average_rating'] < 2.0)])
            elif rating == 5:
                count = len(df_laptop[df_laptop['average_rating'] >= 5.0])
            else:
                count = len(df_laptop[(df_laptop['average_rating'] >= rating) & (df_laptop['average_rating'] < rating + 1)])
            rating_counts[str(rating)] = int(count)
        analytics_data['rating_distribution'] = rating_counts
    
    return render_template('analytics.html', analytics=analytics_data)

@app.route('/api/stats')
def api_stats():
    """API endpoint for getting system statistics."""
    if df_laptop is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        stats = {
            'total_laptops': int(len(df_laptop)),
            'total_reviews': int(len(df_rating)) if df_rating is not None else 0,
            'brands': int(df_laptop['brand'].nunique()) if 'brand' in df_laptop.columns else 0,
            'avg_price': float(df_laptop['price_myr'].mean()),
            'avg_rating': float(df_laptop['average_rating'].mean())
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-more-laptops')
def api_load_more_laptops():
    """API endpoint for loading more laptops with pagination."""
    if df_laptop is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Get laptops for the current page
        laptops_page = df_laptop.iloc[offset:offset + per_page]
        
        # Format laptops for response
        laptops = []
        for _, laptop in laptops_page.iterrows():
            laptop_dict = laptop.to_dict()
            
            # Map column names to what templates expect
            laptop_dict['title_y'] = laptop_dict.get('title_y_clean', laptop_dict.get('title_y', 'Unknown Title'))
            laptop_dict['features'] = laptop_dict.get('features_clean', laptop_dict.get('features', ''))
            laptop_dict['average_rating'] = laptop_dict.get('average_rating', 0.0)
            
            # Ensure brand is available
            if 'brand' not in laptop_dict and 'brand_encoded' in laptop_dict:
                laptop_dict['brand'] = f"Brand_{laptop_dict['brand_encoded']}"
            elif 'brand' not in laptop_dict:
                laptop_dict['brand'] = 'Unknown Brand'
            
            # Ensure price is available
            if 'price_myr' not in laptop_dict:
                laptop_dict['price_myr'] = 0.0
            
            # Convert numpy types to standard Python types and handle NaN values
            for key, value in laptop_dict.items():
                # Handle NaN values
                if pd.isna(value):
                    laptop_dict[key] = None
                # Handle numpy types
                elif hasattr(value, 'item'):
                    laptop_dict[key] = value.item()
                # Handle numpy arrays
                elif hasattr(value, 'tolist'):
                    laptop_dict[key] = value.tolist()
                # Handle other numpy types
                elif hasattr(value, 'dtype'):
                    if value.dtype.kind in 'iuf':  # integer, unsigned integer, float
                        laptop_dict[key] = float(value) if value.dtype.kind == 'f' else int(value)
                    else:
                        laptop_dict[key] = str(value)
            
            laptops.append(laptop_dict)
        
        # Check if there are more laptops
        has_more = offset + per_page < len(df_laptop)
        
        return jsonify({
            'laptops': laptops,
            'has_more': has_more,
            'current_page': page,
            'total_laptops': len(df_laptop)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Initialize the system before starting the app
    if initialize_system():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize the recommendation system. Please check the data files.")
