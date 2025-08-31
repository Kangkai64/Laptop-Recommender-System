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
        
        # Add media columns back if they exist in original data
        media_columns = ['images_y', 'videos']
        for col in media_columns:
            if col in original_data.columns and col not in df_laptop.columns:
                # Map back to laptop dataframe using asin
                media_mapping = original_data[['asin', col]].drop_duplicates(subset=['asin'])
                df_laptop = df_laptop.merge(media_mapping, on='asin', how='left')
                logger.info(f"Added back {col} column from original data")
        
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

# Helper functions for media data
def convert_numpy_to_python(obj):
    """Recursively convert numpy objects to Python native types for JSON serialization."""
    if obj is None:
        return None
    elif hasattr(obj, 'shape') and len(obj.shape) > 0:  # numpy array
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'dtype'):  # other numpy types
        if obj.dtype.kind in 'iuf':  # integer, unsigned integer, float
            return float(obj) if obj.dtype.kind == 'f' else int(obj)
        else:
            return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    elif str(type(obj)).startswith("<class 'numpy"):
        return str(obj)
    else:
        # Check for NaN values safely
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass
        return obj

def extract_image_urls(images_data):
    """Extract image URLs from the complex images_y data structure."""
    if not images_data or pd.isna(images_data):
        return []
    
    try:
        if isinstance(images_data, dict):
            # Extract hi_res images first, then large, then thumb
            if 'hi_res' in images_data and images_data['hi_res'] is not None:
                if hasattr(images_data['hi_res'], 'tolist'):
                    urls = images_data['hi_res'].tolist()
                elif isinstance(images_data['hi_res'], (list, tuple)):
                    urls = list(images_data['hi_res'])
                else:
                    urls = []
                
                # Filter out invalid URLs
                valid_urls = []
                for url in urls:
                    if url and str(url).startswith('http') and str(url) != 'null':
                        valid_urls.append(str(url))
                return valid_urls
                
            elif 'large' in images_data and images_data['large'] is not None:
                if hasattr(images_data['large'], 'tolist'):
                    urls = images_data['large'].tolist()
                elif isinstance(images_data['large'], (list, tuple)):
                    urls = list(images_data['large'])
                else:
                    urls = []
                
                # Filter out invalid URLs
                valid_urls = []
                for url in urls:
                    if url and str(url).startswith('http') and str(url) != 'null':
                        valid_urls.append(str(url))
                return valid_urls
                
            elif 'thumb' in images_data and images_data['thumb'] is not None:
                if hasattr(images_data['thumb'], 'tolist'):
                    urls = images_data['thumb'].tolist()
                elif isinstance(images_data['thumb'], (list, tuple)):
                    urls = list(images_data['thumb'])
                else:
                    urls = []
                
                # Filter out invalid URLs
                valid_urls = []
                for url in urls:
                    if url and str(url).startswith('http') and str(url) != 'null':
                        valid_urls.append(str(url))
                return valid_urls
                
        elif isinstance(images_data, (list, tuple)):
            urls = list(images_data)
            # Filter out invalid URLs
            valid_urls = []
            for url in urls:
                if url and str(url).startswith('http') and str(url) != 'null':
                    valid_urls.append(str(url))
            return valid_urls
            
        elif isinstance(images_data, str):
            if images_data.startswith('http') and images_data != 'null':
                return [images_data]
    except Exception as e:
        logger.warning(f"Error extracting image URLs: {e}")
    
    return []

def extract_video_urls(videos_data, laptop_title=None, laptop_brand=None):
    """Extract video URLs from the complex videos data structure with relevance filtering."""
    if not videos_data or pd.isna(videos_data):
        return []
    
    try:
        video_urls = []
        
        if isinstance(videos_data, dict):
            # Extract video URLs and titles
            if 'url' in videos_data and videos_data['url'] is not None:
                urls = videos_data['url']
                titles = videos_data.get('title', [])
                
                if hasattr(urls, 'tolist'):
                    urls = urls.tolist()
                elif isinstance(urls, (list, tuple)):
                    urls = list(urls)
                
                if hasattr(titles, 'tolist'):
                    titles = titles.tolist()
                elif isinstance(titles, (list, tuple)):
                    titles = list(titles)
                
                # Filter videos based on relevance
                for i, url in enumerate(urls):
                    title = titles[i] if i < len(titles) else ""
                    if is_video_relevant(title, url, laptop_title, laptop_brand):
                        video_urls.append(url)
                        
        elif isinstance(videos_data, (list, tuple)):
            video_urls = list(videos_data)
        elif isinstance(videos_data, str):
            video_urls = [videos_data]
            
    except Exception as e:
        logger.warning(f"Error extracting video URLs: {e}")
    
    return video_urls

def is_video_relevant(video_title, video_url, laptop_title=None, laptop_brand=None):
    """
    Check if a video is relevant to the specific laptop.
    
    Args:
        video_title: Title of the video
        video_url: URL of the video
        laptop_title: Title of the laptop
        laptop_brand: Brand of the laptop
        
    Returns:
        bool: True if video is relevant to the laptop
    """
    if not video_title or not laptop_title:
        return False
    
    # Convert to lowercase for comparison
    video_title_lower = str(video_title).lower()
    laptop_title_lower = str(laptop_title).lower()
    laptop_brand_lower = str(laptop_brand).lower() if laptop_brand else ""
    
    # Check if video title contains laptop brand or model keywords
    brand_keywords = []
    if laptop_brand_lower:
        brand_keywords.extend([laptop_brand_lower, laptop_brand_lower.replace(" ", "")])
    
    # Extract model keywords from laptop title
    model_keywords = []
    if laptop_title_lower:
        # Common laptop model patterns
        words = laptop_title_lower.split()
        for word in words:
            if len(word) >= 3 and any(c.isdigit() for c in word):
                model_keywords.append(word)
    
    # Check for relevance
    relevant_keywords = brand_keywords + model_keywords
    
    for keyword in relevant_keywords:
        if keyword and keyword in video_title_lower:
            return True
    
    # If no specific match found, the video is likely generic
    return False

def extract_video_titles(videos_data):
    """Extract video titles from the complex videos data structure."""
    if not videos_data or pd.isna(videos_data):
        return []
    
    try:
        if isinstance(videos_data, dict):
            # Extract video titles
            if 'title' in videos_data and videos_data['title'] is not None:
                if hasattr(videos_data['title'], 'tolist'):
                    return videos_data['title'].tolist()
                elif isinstance(videos_data['title'], (list, tuple)):
                    return list(videos_data['title'])
        elif isinstance(videos_data, (list, tuple)):
            return list(videos_data)
        elif isinstance(videos_data, str):
            return [videos_data]
    except Exception as e:
        logger.warning(f"Error extracting video titles: {e}")
    
    return []

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
            
            # Process image data for recommendations
            for rec in recommendations:
                # Convert numpy objects to Python native types
                rec = convert_numpy_to_python(rec)
                
                # Extract and process image URLs
                rec['images'] = extract_image_urls(rec.get('images_y'))
                rec['videos'] = extract_video_urls(rec.get('videos'), rec.get('title_y'), rec.get('brand'))
                
                # Ensure title is available
                if 'title_y' not in rec and 'title' in rec:
                    rec['title_y'] = rec['title']
                
                # Ensure brand is available
                if 'brand' not in rec and 'brand_encoded' in rec:
                    rec['brand'] = f"Brand_{rec['brand_encoded']}"
                elif 'brand' not in rec:
                    rec['brand'] = 'Unknown Brand'
                
                # Ensure price is available
                if 'price_myr' not in rec:
                    rec['price_myr'] = 0.0
                
                # Ensure rating is available
                if 'average_rating' not in rec and 'rating' in rec:
                    rec['average_rating'] = rec['rating']
                elif 'average_rating' not in rec:
                    rec['average_rating'] = 0.0
            
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
    budget_min = preferences.get('budget_min', 0)
    
    # Apply budget filtering
    filtered_df = df_laptop[
        (df_laptop['price_myr'] >= budget_min) & 
        (df_laptop['price_myr'] <= budget_max)
    ]
    
    # If no results with budget filter, try without budget constraint
    if len(filtered_df) == 0:
        logger.warning(f"No laptops found in budget range RM {budget_min} - RM {budget_max}, showing all laptops")
        filtered_df = df_laptop
    
    # Get sample laptops
    sample_laptops = filtered_df.sample(min(10, len(filtered_df)))
    
    # Format for templates
    results = []
    for _, laptop in sample_laptops.iterrows():
        laptop_dict = laptop.to_dict()
        # Convert numpy objects to Python native types
        laptop_dict = convert_numpy_to_python(laptop_dict)
        
        # Map column names to what templates expect
        laptop_dict['title_y'] = laptop_dict.get('title_y_clean', laptop_dict.get('title_y', 'Unknown Title'))
        laptop_dict['features'] = laptop_dict.get('features_clean', laptop_dict.get('features', ''))
        
        # Extract media content
        laptop_dict['images'] = extract_image_urls(laptop_dict.get('images_y'))
        laptop_dict['videos'] = extract_video_urls(laptop_dict.get('videos'), laptop_dict.get('title_y'), laptop_dict.get('brand'))
        
        # Ensure images is a list
        if not isinstance(laptop_dict['images'], list):
            laptop_dict['images'] = []
        
        # Ensure brand is available
        if 'brand' not in laptop_dict and 'brand_encoded' in laptop_dict:
            laptop_dict['brand'] = f"Brand_{laptop_dict['brand_encoded']}"
        elif 'brand' not in laptop_dict:
            laptop_dict['brand'] = 'Unknown Brand'
        
        # Ensure price is available
        if 'price_myr' not in laptop_dict:
            laptop_dict['price_myr'] = 0.0
        
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
        laptop_dict['images'] = extract_image_urls(laptop_dict.get('images_y'))  # Include images
        laptop_dict['videos'] = extract_video_urls(laptop_dict.get('videos'), laptop_dict.get('title_y'), laptop_dict.get('brand'))  # Include videos
        
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
    laptop['images'] = extract_image_urls(laptop.get('images_y'))  # Include images
    laptop['videos'] = extract_video_urls(laptop.get('videos'), laptop.get('title_y'), laptop.get('brand'))  # Include videos
    
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
            similar['images'] = extract_image_urls(similar.get('images_y'))  # Include images
            similar['videos'] = extract_video_urls(similar.get('videos'), similar.get('title_y'), similar.get('brand'))  # Include videos
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
        laptop_dict['images'] = extract_image_urls(laptop_dict.get('images_y'))  # Include images
        laptop_dict['videos'] = extract_video_urls(laptop_dict.get('videos'), laptop_dict.get('title_y'), laptop_dict.get('brand'))  # Include videos
        
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
        
        # Debug logging
        logger.info(f"Load more laptops request: page={page}, per_page={per_page}")
        logger.info(f"Total laptops in dataset: {len(df_laptop)}")
        
        # Calculate offset
        offset = (page - 1) * per_page
        logger.info(f"Calculated offset: {offset}")
        
        # Get laptops for the current page
        laptops_page = df_laptop.iloc[offset:offset + per_page]
        logger.info(f"Retrieved {len(laptops_page)} laptops for page {page}")
        
        # Format laptops for response
        laptops = []
        for idx, laptop in laptops_page.iterrows():
            try:
                laptop_dict = laptop.to_dict()
            except Exception as e:
                logger.warning(f"Error converting laptop {idx} to dict: {e}")
                continue
            
            # Map column names to what templates expect
            laptop_dict['title_y'] = laptop_dict.get('title_y_clean', laptop_dict.get('title_y', 'Unknown Title'))
            laptop_dict['features'] = laptop_dict.get('features_clean', laptop_dict.get('features', ''))
            laptop_dict['average_rating'] = laptop_dict.get('average_rating', 0.0)
            laptop_dict['images'] = extract_image_urls(laptop_dict.get('images_y'))  # Include images
            laptop_dict['videos'] = extract_video_urls(laptop_dict.get('videos'), laptop_dict.get('title_y'), laptop_dict.get('brand'))  # Include videos
            
            # Ensure brand is available
            if 'brand' not in laptop_dict and 'brand_encoded' in laptop_dict:
                laptop_dict['brand'] = f"Brand_{laptop_dict['brand_encoded']}"
            elif 'brand' not in laptop_dict:
                laptop_dict['brand'] = 'Unknown Brand'
            
            # Ensure price is available
            if 'price_myr' not in laptop_dict:
                laptop_dict['price_myr'] = 0.0
            
            # Convert all numpy objects to Python native types
            laptop_dict = convert_numpy_to_python(laptop_dict)
            
            laptops.append(laptop_dict)
        
        # Check if there are more laptops
        has_more = offset + per_page < len(df_laptop)
        
        # Final JSON serialization check
        response_data = {
            'laptops': laptops,
            'has_more': has_more,
            'current_page': page,
            'total_laptops': len(df_laptop)
        }
        
        # Test JSON serialization before returning
        try:
            json.dumps(response_data)
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            # Try to fix any remaining non-serializable objects
            for i, laptop in enumerate(laptops):
                for key, value in laptop.items():
                    if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        logger.warning(f"Non-serializable object found in laptop {i}, key '{key}': {type(value)} - {value}")
                        laptop[key] = str(value)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in load-more-laptops API: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
