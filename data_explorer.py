"""
Data Explorer for Laptop Recommender System
Provides comprehensive data analysis and visualization capabilities for Amazon laptop reviews enriched dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from data_preprocessing import LaptopDataPreprocessor
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')  # Use default style for better compatibility
sns.set_palette("husl")

class LaptopDataExplorer:
    """
    Comprehensive data explorer for laptop recommendation system.
    Provides analysis, visualization, and insights from the processed Amazon laptop reviews data.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the data explorer.
        
        Args:
            data_path (str): Optional path to the processed CSV data file. If None, will use preprocessing pipeline.
        """
        self.data_path = data_path
        self.df_laptop = None
        self.df_rating = None
        self.df_combined = None
        self.preprocessor = None
        self.load_data()
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the processed dataset using the preprocessing pipeline.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_laptop, df_rating)
        """
        self.preprocessor = LaptopDataPreprocessor()
        self.df_laptop, self.df_rating = self.preprocessor.preprocess_separated_pipeline()
        
        # Scale up ratings from 0-1 to 1-5 range
        if self.df_rating is not None and 'rating' in self.df_rating.columns:
            # Check if ratings are in 0-1 range (normalized)
            if self.df_rating['rating'].max() <= 1.0:
                self.df_rating['rating_scaled'] = self.df_rating['rating'] * 5
                print("Ratings scaled from 0-1 to 1-5 range")
            else:
                self.df_rating['rating_scaled'] = self.df_rating['rating']
                print("Ratings already in 1-5 range")
        
        print(f"Data loaded from preprocessing pipeline.")
        print(f"Laptop data shape: {self.df_laptop.shape}")
        print(f"Rating data shape: {self.df_rating.shape}")
        return self.df_laptop, self.df_rating
    
    def get_basic_info(self) -> Dict:
        """
        Get basic information about the dataset.
        
        Returns:
            Dict: Basic dataset information
        """
        if self.df_laptop is not None and self.df_rating is not None:
            # Working with separated dataframes
            laptop_info = {
                'shape': self.df_laptop.shape,
                'columns': list(self.df_laptop.columns),
                'data_types': self.df_laptop.dtypes.to_dict(),
                'missing_values': self.df_laptop.isnull().sum().to_dict(),
                'memory_usage': self.df_laptop.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
            
            rating_info = {
                'shape': self.df_rating.shape,
                'columns': list(self.df_rating.columns),
                'data_types': self.df_rating.dtypes.to_dict(),
                'missing_values': self.df_rating.isnull().sum().to_dict(),
                'memory_usage': self.df_rating.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
            
            return {
                'laptop_data': laptop_info,
                'rating_data': rating_info
            }
        elif self.df_combined is not None:
            # Working with combined dataframe
            return {
                'shape': self.df_combined.shape,
                'columns': list(self.df_combined.columns),
                'data_types': self.df_combined.dtypes.to_dict(),
                'missing_values': self.df_combined.isnull().sum().to_dict(),
                'memory_usage': self.df_combined.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
        else:
            return {'error': 'No data loaded'}
    
    def analyze_price_distribution(self) -> Dict:
        """
        Analyze price distribution and statistics.
        
        Returns:
            Dict: Price analysis results
        """
        if self.df_laptop is None:
            return {'error': 'Laptop data not available'}
        
        # Check for different possible price column names
        price_cols = ['price_myr', 'price_usd', 'price_numeric', 'price']
        price_col = None
        for col in price_cols:
            if col in self.df_laptop.columns:
                price_col = col
                break
        
        if price_col is None:
            return {'error': 'Price column not found'}
        
        analysis = {
            'price_statistics': {
                'min': self.df_laptop[price_col].min(),
                'max': self.df_laptop[price_col].max(),
                'mean': self.df_laptop[price_col].mean(),
                'median': self.df_laptop[price_col].median(),
                'std': self.df_laptop[price_col].std(),
                'q25': self.df_laptop[price_col].quantile(0.25),
                'q75': self.df_laptop[price_col].quantile(0.75)
            }
        }
        
        # Add price categories if available
        if 'price_category_myr' in self.df_laptop.columns:
            analysis['price_categories'] = self.df_laptop['price_category_myr'].value_counts().to_dict()
            analysis['price_range_analysis'] = {
                'budget_count': len(self.df_laptop[self.df_laptop['price_category_myr'] == 'Budget']),
                'mid_range_count': len(self.df_laptop[self.df_laptop['price_category_myr'] == 'Mid-range']),
                'high_end_count': len(self.df_laptop[self.df_laptop['price_category_myr'] == 'High-end']),
                'premium_count': len(self.df_laptop[self.df_laptop['price_category_myr'] == 'Premium'])
            }
        
        return analysis
    
    def analyze_brands(self) -> Dict:
        """
        Analyze brand distribution and statistics.
        
        Returns:
            Dict: Brand analysis results
        """
        if self.df_laptop is None:
            return {'error': 'Laptop data not available'}
        
        # Get actual brand names for display
        df_with_brands = self.get_actual_brand_names(self.df_laptop)
        
        # Use brand_display column for analysis
        brand_col = 'brand_display' if 'brand_display' in df_with_brands.columns else 'brand'
        
        if brand_col not in df_with_brands.columns:
            return {'error': 'Brand column not found'}
        
        brand_analysis = df_with_brands[brand_col].value_counts()
        
        analysis = {
            'total_brands': len(brand_analysis),
            'top_brands': brand_analysis.head(10).to_dict(),
            'brand_stats': {
                'most_popular': brand_analysis.index[0],
                'most_popular_count': brand_analysis.iloc[0],
                'avg_models_per_brand': brand_analysis.mean()
            }
        }
        
        # Add rating analysis by brand if available
        rating_col = 'average_rating' if 'average_rating' in self.df_laptop.columns else None
        if rating_col:
            brand_ratings = df_with_brands.groupby(brand_col)[rating_col].agg(['mean', 'count']).sort_values('count', ascending=False)
            analysis['brand_ratings'] = brand_ratings.head(10).to_dict('index')
        
        # Add note about brand restoration
        if 'brand_encoded' in self.df_laptop.columns and 'brand' not in self.df_laptop.columns:
            analysis['note'] = 'Brand names restored from encoded values using label encoder.'
        
        return analysis
    
    def analyze_ratings(self) -> Dict:
        """
        Analyze rating distribution and patterns.
        
        Returns:
            Dict: Rating analysis results
        """
        if self.df_rating is None:
            return {'error': 'Rating data not available'}
        
        # Use scaled ratings if available, otherwise use original ratings
        rating_col = 'rating_scaled' if 'rating_scaled' in self.df_rating.columns else 'rating'
        
        if rating_col not in self.df_rating.columns:
            return {'error': 'Rating column not found'}
        
        analysis = {
            'rating_distribution': self.df_rating[rating_col].value_counts().sort_index().to_dict(),
            'rating_statistics': {
                'mean': self.df_rating[rating_col].mean(),
                'median': self.df_rating[rating_col].median(),
                'std': self.df_rating[rating_col].std(),
                'min': self.df_rating[rating_col].min(),
                'max': self.df_rating[rating_col].max()
            }
        }
        
        # Add note about rating scaling
        if 'rating_scaled' in self.df_rating.columns:
            analysis['note'] = f'Ratings scaled to 1-5 range. Original range was {self.df_rating["rating"].min():.2f}-{self.df_rating["rating"].max():.2f}'
            analysis['original_rating_statistics'] = {
                'mean': self.df_rating['rating'].mean(),
                'median': self.df_rating['rating'].median(),
                'min': self.df_rating['rating'].min(),
                'max': self.df_rating['rating'].max()
            }
        
        # Analyze helpful votes
        if 'helpful_vote' in self.df_rating.columns:
            analysis['helpful_votes'] = {
                'total_helpful_votes': self.df_rating['helpful_vote'].sum(),
                'avg_helpful_votes': self.df_rating['helpful_vote'].mean(),
                'max_helpful_votes': self.df_rating['helpful_vote'].max()
            }
        
        return analysis
    
    def analyze_reviews(self) -> Dict:
        """
        Analyze review content and patterns.
        
        Returns:
            Dict: Review analysis results
        """
        if self.df_rating is None:
            return {'error': 'Rating data not available'}
        
        # Check for text column (could be 'text' or 'text_clean')
        text_col = 'text' if 'text' in self.df_rating.columns else 'text_clean' if 'text_clean' in self.df_rating.columns else None
        
        if text_col is None:
            return {'error': 'Review text column not found'}
        
        analysis = {
            'review_statistics': {
                'total_reviews': len(self.df_rating),
                'avg_review_length': self.df_rating[text_col].astype(str).str.len().mean(),
                'max_review_length': self.df_rating[text_col].astype(str).str.len().max(),
                'min_review_length': self.df_rating[text_col].astype(str).str.len().min()
            }
        }
        
        # Analyze review length distribution
        review_lengths = self.df_rating[text_col].astype(str).str.len()
        analysis['review_length_categories'] = pd.cut(
            review_lengths, 
            bins=[0, 100, 500, 1000, float('inf')], 
            labels=['Short', 'Medium', 'Long', 'Very Long']
        ).value_counts().to_dict()
        
        return analysis
    
    def analyze_specifications(self) -> Dict:
        """
        Analyze laptop specifications extracted from details.
        
        Returns:
            Dict: Specifications analysis results
        """
        if self.df_laptop is None:
            return {'error': 'Laptop data not available'}
        
        analysis = {}
        
        # Analyze RAM distribution
        if 'ram_gb' in self.df_laptop.columns:
            ram_data = self.df_laptop['ram_gb'].dropna()
            if len(ram_data) > 0:
                analysis['ram_analysis'] = {
                    'common_ram_sizes': ram_data.value_counts().head(5).to_dict(),
                    'ram_statistics': {
                        'mean': ram_data.mean(),
                        'median': ram_data.median(),
                        'min': ram_data.min(),
                        'max': ram_data.max()
                    }
                }
        
        # Analyze storage distribution
        if 'storage_gb' in self.df_laptop.columns:
            storage_data = self.df_laptop['storage_gb'].dropna()
            if len(storage_data) > 0:
                analysis['storage_analysis'] = {
                    'common_storage_sizes': storage_data.value_counts().head(5).to_dict(),
                    'storage_statistics': {
                        'mean': storage_data.mean(),
                        'median': storage_data.median(),
                        'min': storage_data.min(),
                        'max': storage_data.max()
                    }
                }
        
        # Analyze screen sizes
        if 'screen_size' in self.df_laptop.columns:
            screen_data = self.df_laptop['screen_size'].dropna()
            if len(screen_data) > 0:
                analysis['screen_analysis'] = {
                    'common_screen_sizes': screen_data.value_counts().head(5).to_dict(),
                    'screen_statistics': {
                        'mean': screen_data.mean(),
                        'median': screen_data.median(),
                        'min': screen_data.min(),
                        'max': screen_data.max()
                    }
                }
        
        # Analyze operating systems (check both original and encoded)
        os_col = 'os' if 'os' in self.df_laptop.columns else 'os_encoded' if 'os_encoded' in self.df_laptop.columns else None
        if os_col:
            analysis['os_analysis'] = {
                'os_distribution': self.df_laptop[os_col].value_counts().to_dict(),
                'most_common_os': self.df_laptop[os_col].mode().iloc[0] if len(self.df_laptop[os_col].mode()) > 0 else None
            }
            if os_col == 'os_encoded':
                analysis['os_analysis']['note'] = 'OS values are encoded. Use original OS column for actual OS names.'
        
        # Add note about available specification columns
        available_spec_cols = [col for col in self.df_laptop.columns if any(x in col.lower() for x in ['ram', 'storage', 'screen', 'processor'])]
        if available_spec_cols:
            analysis['available_specifications'] = available_spec_cols
        else:
            analysis['note'] = 'No detailed specifications columns found. Data may be encoded or in different format.'
        
        return analysis
    
    def create_visualizations(self, save_path: str = "data/visualizations/"):
        """
        Create comprehensive visualizations of the dataset.
        
        Args:
            save_path (str): Path to save visualizations
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Price Distribution (Laptop Data)
        if self.df_laptop is not None and 'price_myr' in self.df_laptop.columns:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.hist(self.df_laptop['price_myr'].dropna(), bins=50, alpha=0.7, edgecolor='black')
            plt.title('Price Distribution (MYR)')
            plt.xlabel('Price (MYR)')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 3, 2)
            if 'price_category_myr' in self.df_laptop.columns:
                self.df_laptop['price_category_myr'].value_counts().plot(kind='bar')
                plt.title('Price Categories Distribution')
                plt.xlabel('Price Category')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
            
            plt.subplot(1, 3, 3)
            if 'price_usd' in self.df_laptop.columns:
                plt.hist(self.df_laptop['price_usd'].dropna(), bins=50, alpha=0.7, edgecolor='black')
                plt.title('Price Distribution (USD)')
                plt.xlabel('Price (USD)')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}price_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Rating Analysis (Rating Data)
        if self.df_rating is not None and 'rating' in self.df_rating.columns:
            plt.figure(figsize=(15, 5))
            
            # Use scaled ratings if available
            rating_col = 'rating_scaled' if 'rating_scaled' in self.df_rating.columns else 'rating'
            
            plt.subplot(1, 3, 1)
            self.df_rating[rating_col].value_counts().sort_index().plot(kind='bar')
            plt.title('Rating Distribution (1-5 Scale)')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            
            plt.subplot(1, 3, 2)
            plt.hist(self.df_rating[rating_col], bins=20, alpha=0.7, edgecolor='black')
            plt.title('Rating Histogram (1-5 Scale)')
            plt.xlabel('Rating')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 3, 3)
            if 'helpful_vote' in self.df_rating.columns:
                plt.hist(self.df_rating['helpful_vote'].dropna(), bins=30, alpha=0.7, edgecolor='black')
                plt.title('Helpful Votes Distribution')
                plt.xlabel('Helpful Votes')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}rating_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Brand Analysis (Laptop Data)
        if self.df_laptop is not None and ('brand' in self.df_laptop.columns or 'brand_encoded' in self.df_laptop.columns):
            plt.figure(figsize=(12, 6))
            
            # Get actual brand names for display
            df_with_brands = self.get_actual_brand_names(self.df_laptop)
            brand_col = 'brand_display' if 'brand_display' in df_with_brands.columns else 'brand'
            
            if brand_col in df_with_brands.columns:
                top_brands = df_with_brands[brand_col].value_counts().head(10)
                top_brands.plot(kind='bar')
                plt.title('Top 10 Brands by Number of Products')
                plt.xlabel('Brand')
                plt.ylabel('Number of Products')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{save_path}brand_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # 4. Review Length Analysis (Rating Data)
        if self.df_rating is not None and 'text' in self.df_rating.columns:
            plt.figure(figsize=(12, 6))
            review_lengths = self.df_rating['text'].astype(str).str.len()
            
            plt.subplot(1, 2, 1)
            plt.hist(review_lengths, bins=50, alpha=0.7, edgecolor='black')
            plt.title('Review Length Distribution')
            plt.xlabel('Review Length (characters)')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            if 'rating' in self.df_rating.columns:
                # Use scaled ratings if available
                rating_col = 'rating_scaled' if 'rating_scaled' in self.df_rating.columns else 'rating'
                plt.scatter(review_lengths, self.df_rating[rating_col], alpha=0.5)
                plt.title('Review Length vs Rating (1-5 Scale)')
                plt.xlabel('Review Length (characters)')
                plt.ylabel('Rating')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}review_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Specifications Analysis (Laptop Data)
        if self.df_laptop is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # RAM analysis
            if 'ram_gb' in self.df_laptop.columns:
                ram_data = self.df_laptop['ram_gb'].dropna()
                if len(ram_data) > 0:
                    axes[0, 0].hist(ram_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[0, 0].set_title('RAM Distribution')
                    axes[0, 0].set_xlabel('RAM (GB)')
                    axes[0, 0].set_ylabel('Frequency')
            
            # Storage analysis
            if 'storage_gb' in self.df_laptop.columns:
                storage_data = self.df_laptop['storage_gb'].dropna()
                if len(storage_data) > 0:
                    axes[0, 1].hist(storage_data, bins=20, alpha=0.7, edgecolor='black')
                    axes[0, 1].set_title('Storage Distribution')
                    axes[0, 1].set_xlabel('Storage (GB)')
                    axes[0, 1].set_ylabel('Frequency')
            
            # Screen size analysis
            if 'screen_size' in self.df_laptop.columns:
                screen_data = self.df_laptop['screen_size'].dropna()
                if len(screen_data) > 0:
                    axes[1, 0].hist(screen_data, bins=15, alpha=0.7, edgecolor='black')
                    axes[1, 0].set_title('Screen Size Distribution')
                    axes[1, 0].set_xlabel('Screen Size (inches)')
                    axes[1, 0].set_ylabel('Frequency')
            
            # OS distribution
            if 'os' in self.df_laptop.columns:
                self.df_laptop['os'].value_counts().head(8).plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Operating System Distribution')
                axes[1, 1].set_xlabel('Operating System')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}specifications_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {save_path}")
    
    def generate_report(self, output_path: str = "data/analysis_report.txt"):
        """
        Generate a comprehensive analysis report.
        
        Args:
            output_path (str): Path to save the report
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("AMAZON LAPTOP REVIEWS ENRICHED DATASET ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Basic Information
        basic_info = self.get_basic_info()
        if 'error' not in basic_info:
            if 'laptop_data' in basic_info:
                # Separated data structure
                report_lines.append("DATASET OVERVIEW")
                report_lines.append("-" * 30)
                report_lines.append(f"Laptop Data: {basic_info['laptop_data']['shape'][0]:,} products, {basic_info['laptop_data']['shape'][1]} features")
                report_lines.append(f"Rating Data: {basic_info['rating_data']['shape'][0]:,} reviews, {basic_info['rating_data']['shape'][1]} features")
                report_lines.append(f"Total Memory Usage: {basic_info['laptop_data']['memory_usage'] + basic_info['rating_data']['memory_usage']:.2f} MB")
            else:
                # Combined data structure
                report_lines.append("DATASET OVERVIEW")
                report_lines.append("-" * 30)
                report_lines.append(f"Total Records: {basic_info['shape'][0]:,}")
                report_lines.append(f"Total Features: {basic_info['shape'][1]}")
                report_lines.append(f"Memory Usage: {basic_info['memory_usage']:.2f} MB")
            report_lines.append("")
        
        # Price Analysis
        price_analysis = self.analyze_price_distribution()
        if 'error' not in price_analysis:
            report_lines.append("PRICE ANALYSIS")
            report_lines.append("-" * 30)
            stats = price_analysis['price_statistics']
            report_lines.append(f"Price Range: RM {stats['min']:.2f} - RM {stats['max']:.2f}")
            report_lines.append(f"Average Price: RM {stats['mean']:.2f}")
            report_lines.append(f"Median Price: RM {stats['median']:.2f}")
            report_lines.append("")
            
            if 'price_categories' in price_analysis:
                report_lines.append("Price Categories:")
                for category, count in price_analysis['price_categories'].items():
                    report_lines.append(f"  {category}: {count:,} laptops")
                report_lines.append("")
        
        # Brand Analysis
        brand_analysis = self.analyze_brands()
        if 'error' not in brand_analysis:
            report_lines.append("BRAND ANALYSIS")
            report_lines.append("-" * 30)
            report_lines.append(f"Total Brands: {brand_analysis['total_brands']}")
            report_lines.append(f"Most Popular Brand: {brand_analysis['brand_stats']['most_popular']}")
            report_lines.append(f"Average Models per Brand: {brand_analysis['brand_stats']['avg_models_per_brand']:.1f}")
            
            # Add note about brand restoration if applicable
            if 'note' in brand_analysis:
                report_lines.append(f"Note: {brand_analysis['note']}")
            
            report_lines.append("")
            report_lines.append("Top 10 Brands:")
            for i, (brand, count) in enumerate(list(brand_analysis['top_brands'].items())[:10], 1):
                report_lines.append(f"  {i}. {brand}: {count:,} products")
            report_lines.append("")
        
        # Rating Analysis
        rating_analysis = self.analyze_ratings()
        if 'error' not in rating_analysis:
            report_lines.append("RATING ANALYSIS")
            report_lines.append("-" * 30)
            stats = rating_analysis['rating_statistics']
            report_lines.append(f"Average Rating: {stats['mean']:.2f}/5.0")
            report_lines.append(f"Median Rating: {stats['median']:.2f}/5.0")
            report_lines.append(f"Rating Range: {stats['min']:.1f} - {stats['max']:.1f}")
            
            # Add note about rating scaling if applicable
            if 'note' in rating_analysis:
                report_lines.append(f"Note: {rating_analysis['note']}")
                if 'original_rating_statistics' in rating_analysis:
                    orig_stats = rating_analysis['original_rating_statistics']
                    report_lines.append(f"Original Rating Range: {orig_stats['min']:.2f} - {orig_stats['max']:.2f}")
                report_lines.append("")
            
            if 'helpful_votes' in rating_analysis:
                helpful = rating_analysis['helpful_votes']
                report_lines.append(f"Total Helpful Votes: {helpful['total_helpful_votes']:,}")
                report_lines.append(f"Average Helpful Votes per Review: {helpful['avg_helpful_votes']:.1f}")
                report_lines.append("")
        
        # Review Analysis
        review_analysis = self.analyze_reviews()
        if 'error' not in review_analysis:
            report_lines.append("REVIEW ANALYSIS")
            report_lines.append("-" * 30)
            stats = review_analysis['review_statistics']
            report_lines.append(f"Total Reviews: {stats['total_reviews']:,}")
            report_lines.append(f"Average Review Length: {stats['avg_review_length']:.0f} characters")
            report_lines.append(f"Review Length Range: {stats['min_review_length']} - {stats['max_review_length']} characters")
            report_lines.append("")
            
            if 'review_length_categories' in review_analysis:
                report_lines.append("Review Length Categories:")
                for category, count in review_analysis['review_length_categories'].items():
                    report_lines.append(f"  {category}: {count:,} reviews")
                report_lines.append("")
        
        # Specifications Analysis
        spec_analysis = self.analyze_specifications()
        if spec_analysis and 'error' not in spec_analysis:
            report_lines.append("SPECIFICATIONS ANALYSIS")
            report_lines.append("-" * 30)
            
            if 'ram_analysis' in spec_analysis:
                ram_stats = spec_analysis['ram_analysis']['ram_statistics']
                report_lines.append(f"RAM Range: {ram_stats['min']:.0f} - {ram_stats['max']:.0f} GB")
                report_lines.append(f"Average RAM: {ram_stats['mean']:.1f} GB")
                report_lines.append("")
            
            if 'storage_analysis' in spec_analysis:
                storage_stats = spec_analysis['storage_analysis']['storage_statistics']
                report_lines.append(f"Storage Range: {storage_stats['min']:.0f} - {storage_stats['max']:.0f} GB")
                report_lines.append(f"Average Storage: {storage_stats['mean']:.1f} GB")
                report_lines.append("")
            
            if 'os_analysis' in spec_analysis:
                report_lines.append(f"Most Common OS: {spec_analysis['os_analysis']['most_common_os']}")
                report_lines.append("")
        

        
        # Save the report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Analysis report saved to {output_path}")
    
    def get_recommendation_insights(self) -> Dict:
        """
        Get insights useful for the recommendation system.
        
        Returns:
            Dict: Recommendation insights
        """
        insights = {}
        
        # Price insights
        if self.df_laptop is not None and 'price_myr' in self.df_laptop.columns:
            price_quartiles = self.df_laptop['price_myr'].quantile([0.25, 0.5, 0.75])
            insights['price_insights'] = {
                'budget_threshold': price_quartiles[0.25],
                'mid_range_threshold': price_quartiles[0.5],
                'high_end_threshold': price_quartiles[0.75]
            }
        
        # Rating insights
        if self.df_rating is not None and 'rating' in self.df_rating.columns:
            insights['rating_insights'] = {
                'highly_rated_threshold': self.df_rating['rating'].quantile(0.8),
                'average_rating': self.df_rating['rating'].mean(),
                'rating_distribution': self.df_rating['rating'].value_counts().sort_index().to_dict()
            }
        
        # Brand insights
        if self.df_laptop is not None and 'brand' in self.df_laptop.columns:
            top_brands = self.df_laptop['brand'].value_counts().head(10)
            insights['brand_insights'] = {
                'top_brands': top_brands.to_dict(),
                'brand_count': len(self.df_laptop['brand'].unique())
            }
        
        # Feature importance insights
        if (self.df_rating is not None and 'helpful_vote' in self.df_rating.columns 
            and 'rating' in self.df_rating.columns):
            insights['feature_insights'] = {
                'helpful_reviews_threshold': self.df_rating['helpful_vote'].quantile(0.8),
                'correlation_rating_helpful': self.df_rating['rating'].corr(self.df_rating['helpful_vote'])
            }
        
        return insights
    
    def get_dataset_summary(self) -> Dict:
        """
        Get a comprehensive summary of the datasets.
        
        Returns:
            Dict: Dataset summary
        """
        summary = {}
        
        if self.df_laptop is not None:
            # Get non-encoded brand information for better readability
            brand_column = 'brand'
            if 'brand_encoded' in self.df_laptop.columns and 'brand' not in self.df_laptop.columns:
                # If we only have encoded brands, try to get original brands from the combined data
                if hasattr(self, 'df_combined') and self.df_combined is not None:
                    if 'brand' in self.df_combined.columns:
                        # Map encoded brands back to original brands
                        brand_mapping = self.df_combined[['brand', 'brand_encoded']].drop_duplicates()
                        brand_mapping = brand_mapping.set_index('brand_encoded')['brand'].to_dict()
                        readable_brands = self.df_laptop['brand_encoded'].map(brand_mapping)
                        brand_count = readable_brands.nunique()
                        top_brands = readable_brands.value_counts().head(5).to_dict()
                    else:
                        brand_count = self.df_laptop['brand_encoded'].nunique()
                        top_brands = self.df_laptop['brand_encoded'].value_counts().head(5).to_dict()
                else:
                    brand_count = self.df_laptop['brand_encoded'].nunique()
                    top_brands = self.df_laptop['brand_encoded'].value_counts().head(5).to_dict()
            else:
                brand_count = self.df_laptop[brand_column].nunique() if brand_column in self.df_laptop.columns else 0
                top_brands = self.df_laptop[brand_column].value_counts().head(5).to_dict() if brand_column in self.df_laptop.columns else {}
            
            laptop_summary = {
                'total_products': len(self.df_laptop),
                'total_features': len(self.df_laptop.columns),
                'brands_count': brand_count,
                'top_brands': top_brands,
                'price_range_myr': {
                    'min': self.df_laptop['price_myr'].min() if 'price_myr' in self.df_laptop.columns else None,
                    'max': self.df_laptop['price_myr'].max() if 'price_myr' in self.df_laptop.columns else None,
                    'mean': self.df_laptop['price_myr'].mean() if 'price_myr' in self.df_laptop.columns else None
                },
                'average_rating': self.df_laptop['average_rating'].mean() if 'average_rating' in self.df_laptop.columns else None,
                'specifications': {}
            }
            
            # Add specification information if available
            if 'ram_gb' in self.df_laptop.columns:
                ram_stats = self.df_laptop['ram_gb'].describe()
                laptop_summary['specifications']['ram'] = {
                    'found': int(self.df_laptop['ram_gb'].notna().sum()),
                    'total': len(self.df_laptop),
                    'mean_gb': float(ram_stats['mean']) if not pd.isna(ram_stats['mean']) else None,
                    'min_gb': int(ram_stats['min']) if not pd.isna(ram_stats['min']) else None,
                    'max_gb': int(ram_stats['max']) if not pd.isna(ram_stats['max']) else None
                }
            
            if 'storage_gb' in self.df_laptop.columns:
                storage_stats = self.df_laptop['storage_gb'].describe()
                laptop_summary['specifications']['storage'] = {
                    'found': int(self.df_laptop['storage_gb'].notna().sum()),
                    'total': len(self.df_laptop),
                    'mean_gb': float(storage_stats['mean']) if not pd.isna(storage_stats['mean']) else None,
                    'min_gb': int(storage_stats['min']) if not pd.isna(storage_stats['min']) else None,
                    'max_gb': int(storage_stats['max']) if not pd.isna(storage_stats['max']) else None
                }
            
            if 'screen_size_inches' in self.df_laptop.columns:
                screen_stats = self.df_laptop['screen_size_inches'].describe()
                laptop_summary['specifications']['screen_size'] = {
                    'found': int(self.df_laptop['screen_size_inches'].notna().sum()),
                    'total': len(self.df_laptop),
                    'mean_inches': float(screen_stats['mean']) if not pd.isna(screen_stats['mean']) else None,
                    'min_inches': float(screen_stats['min']) if not pd.isna(screen_stats['min']) else None,
                    'max_inches': float(screen_stats['max']) if not pd.isna(screen_stats['max']) else None
                }
            
            # Add column categories for better organization
            laptop_cols = list(self.df_laptop.columns)
            laptop_summary['column_categories'] = {
                'product_info': [col for col in laptop_cols if any(x in col for x in ['title', 'brand', 'os', 'color', 'store'])],
                'pricing': [col for col in laptop_cols if 'price' in col],
                'ratings': [col for col in laptop_cols if 'rating' in col],
                'specifications': [col for col in laptop_cols if any(x in col for x in ['ram', 'storage', 'screen', 'processor', 'gpu'])],
                'benchmarks': [col for col in laptop_cols if 'benchmark' in col],
                'categories': [col for col in laptop_cols if 'category' in col],
                'normalized': [col for col in laptop_cols if any(x in col for x in ['encoded', 'clean', 'normalized'])]
            }
            
            summary['laptop_data'] = laptop_summary
        
        if self.df_rating is not None:
            rating_summary = {
                'total_reviews': len(self.df_rating),
                'total_features': len(self.df_rating.columns),
                'unique_users': self.df_rating['user_id'].nunique() if 'user_id' in self.df_rating.columns else 0,
                'unique_products': self.df_rating['asin'].nunique() if 'asin' in self.df_rating.columns else 0,
                'rating_stats': {
                    'mean': self.df_rating['rating'].mean() if 'rating' in self.df_rating.columns else None,
                    'median': self.df_rating['rating'].median() if 'rating' in self.df_rating.columns else None
                }
            }
            
            summary['rating_data'] = rating_summary
        
        return summary
    
    def get_actual_brand_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace encoded brand values with actual brand names for display.
        
        Args:
            df (pd.DataFrame): DataFrame with encoded brands
            
        Returns:
            pd.DataFrame: DataFrame with actual brand names
        """
        if df is None:
            return df
        
        df_display = df.copy()
        
        # Check if we have brand_encoded column and can map it back
        if 'brand_encoded' in df_display.columns and 'brand' not in df_display.columns:
            if hasattr(self, 'preprocessor') and self.preprocessor and 'laptop_brand' in self.preprocessor.label_encoders:
                le = self.preprocessor.label_encoders['laptop_brand']
                # Create reverse mapping from encoded values to original brand names
                brand_mapping = {i: name for i, name in enumerate(le.classes_)}
                df_display['brand_display'] = df_display['brand_encoded'].map(brand_mapping)
                print("Brand names restored from label encoder")
            else:
                # If no label encoder available, keep encoded values but mark them
                df_display['brand_display'] = df_display['brand_encoded'].astype(str) + ' (encoded)'
                print("Warning: Brand names could not be restored - using encoded values")
        elif 'brand' in df_display.columns:
            df_display['brand_display'] = df_display['brand']
        else:
            df_display['brand_display'] = 'Unknown'
        
        return df_display

    def get_original_brand_names(self) -> Dict:
        """
        Get original brand names if available in the preprocessing pipeline.
        
        Returns:
            Dict: Mapping of encoded values to original brand names
        """
        try:
            if hasattr(self, 'preprocessor') and self.preprocessor:
                # Try to get label encoders from the preprocessor
                if 'laptop_brand' in self.preprocessor.label_encoders:
                    le = self.preprocessor.label_encoders['laptop_brand']
                    # Create reverse mapping
                    brand_mapping = {i: name for i, name in enumerate(le.classes_)}
                    return brand_mapping
            return {}
        except:
            return {}
    
    def get_data_quality_insights(self) -> Dict:
        """
        Get insights about data quality and preprocessing.
        
        Returns:
            Dict: Data quality insights
        """
        insights = {}
        
        if self.df_laptop is not None:
            insights['laptop_data_quality'] = {
                'total_products': len(self.df_laptop),
                'missing_values': self.df_laptop.isnull().sum().sum(),
                'missing_percentage': (self.df_laptop.isnull().sum().sum() / (len(self.df_laptop) * len(self.df_laptop.columns))) * 100,
                'encoded_columns': [col for col in self.df_laptop.columns if col.endswith('_encoded')],
                'clean_columns': [col for col in self.df_laptop.columns if col.endswith('_clean')],
                'numerical_columns': [col for col in self.df_laptop.columns if self.df_laptop[col].dtype in ['int64', 'float64']]
            }
        
        if self.df_rating is not None:
            insights['rating_data_quality'] = {
                'total_reviews': len(self.df_rating),
                'missing_values': self.df_rating.isnull().sum().sum(),
                'missing_percentage': (self.df_rating.isnull().sum().sum() / (len(self.df_rating) * len(self.df_rating.columns))) * 100,
                'encoded_columns': [col for col in self.df_rating.columns if col.endswith('_encoded')],
                'clean_columns': [col for col in self.df_rating.columns if col.endswith('_clean')],
                'temporal_columns': [col for col in self.df_rating.columns if col in ['year', 'month', 'day_of_week']]
            }
        
        return insights

    def identify_and_clean_invalid_specifications(self) -> Dict:
        """
        Identify and automatically drop records with RAM and storage values outside valid ranges.
        
        Returns:
            Dict: Information about invalid records that were dropped
        """
        if self.df_laptop is None:
            return {'error': 'Laptop data not available'}
        
        invalid_records = {
            'ram_invalid': [],
            'storage_invalid': [],
            'both_invalid': [],
            'summary': {}
        }
        
        # Define valid ranges
        valid_ram_range = (1, 128)  # 1GB to 128GB
        valid_storage_range = (16, 8192)  # 16GB to 8TB
        
        # Check RAM validity
        if 'ram_gb' in self.df_laptop.columns:
            ram_mask = (
                (self.df_laptop['ram_gb'] < valid_ram_range[0]) | 
                (self.df_laptop['ram_gb'] > valid_ram_range[1])
            ) & self.df_laptop['ram_gb'].notna()
            
            if ram_mask.any():
                invalid_ram_indices = self.df_laptop[ram_mask].index.tolist()
                invalid_ram_values = self.df_laptop.loc[ram_mask, 'ram_gb'].tolist()
                invalid_records['ram_invalid'] = list(zip(invalid_ram_indices, invalid_ram_values))
        
        # Check storage validity
        if 'storage_gb' in self.df_laptop.columns:
            storage_mask = (
                (self.df_laptop['storage_gb'] < valid_storage_range[0]) | 
                (self.df_laptop['storage_gb'] > valid_storage_range[1])
            ) & self.df_laptop['storage_gb'].notna()
            
            if storage_mask.any():
                invalid_storage_indices = self.df_laptop[storage_mask].index.tolist()
                invalid_storage_values = self.df_laptop.loc[storage_mask, 'storage_gb'].tolist()
                invalid_records['storage_invalid'] = list(zip(invalid_storage_indices, invalid_storage_values))
        
        # Find records with both invalid
        if invalid_records['ram_invalid'] and invalid_records['storage_invalid']:
            ram_indices = set(idx for idx, _ in invalid_records['ram_invalid'])
            storage_indices = set(idx for idx, _ in invalid_records['storage_invalid'])
            both_invalid_indices = ram_indices.intersection(storage_indices)
            
            for idx in both_invalid_indices:
                ram_val = next(val for i, val in invalid_records['ram_invalid'] if i == idx)
                storage_val = next(val for i, val in invalid_records['storage_invalid'] if i == idx)
                invalid_records['both_invalid'].append((idx, ram_val, storage_val))
        
        # Get all invalid indices for dropping
        invalid_indices = set()
        if invalid_records['ram_invalid']:
            invalid_indices.update([idx for idx, _ in invalid_records['ram_invalid']])
        if invalid_records['storage_invalid']:
            invalid_indices.update([idx for idx, _ in invalid_records['storage_invalid']])
        
        # Drop invalid records automatically
        if invalid_indices:
            original_count = len(self.df_laptop)
            self.df_laptop = self.df_laptop.drop(index=list(invalid_indices))
            dropped_count = original_count - len(self.df_laptop)
            
            # Create summary with cleaning results
            invalid_records['summary'] = {
                'total_records_original': original_count,
                'total_records_final': len(self.df_laptop),
                'records_dropped': dropped_count,
                'ram_invalid_count': len(invalid_records['ram_invalid']),
                'storage_invalid_count': len(invalid_records['storage_invalid']),
                'both_invalid_count': len(invalid_records['both_invalid']),
                'total_invalid_records': len(invalid_indices),
                'valid_ram_range': valid_ram_range,
                'valid_storage_range': valid_storage_range,
                'action_taken': 'dropped_invalid_records'
            }
        else:
            # No invalid records found
            invalid_records['summary'] = {
                'total_records_original': len(self.df_laptop),
                'total_records_final': len(self.df_laptop),
                'records_dropped': 0,
                'ram_invalid_count': 0,
                'storage_invalid_count': 0,
                'both_invalid_count': 0,
                'total_invalid_records': 0,
                'valid_ram_range': valid_ram_range,
                'valid_storage_range': valid_storage_range,
                'action_taken': 'no_action_needed'
            }
        
        return invalid_records
    
    def get_invalid_records_details(self) -> pd.DataFrame:
        """
        Get detailed information about records with invalid specifications.
        Note: This method is now mainly for reference since invalid records are automatically dropped.
        
        Returns:
            pd.DataFrame: DataFrame with invalid records and their details (if any remain)
        """
        invalid_info = self.identify_and_clean_invalid_specifications()
        
        if 'error' in invalid_info:
            return pd.DataFrame()
        
        invalid_indices = set()
        if invalid_info['ram_invalid']:
            invalid_indices.update([idx for idx, _ in invalid_info['ram_invalid']])
        if invalid_info['storage_invalid']:
            invalid_indices.update([idx for idx, _ in invalid_info['storage_invalid']])
        
        if not invalid_indices:
            return pd.DataFrame()
        
        # Get the invalid records with all their details
        invalid_df = self.df_laptop.loc[list(invalid_indices)].copy()
        
        # Add validation flags
        invalid_df['ram_invalid'] = False
        invalid_df['storage_invalid'] = False
        
        for idx, _ in invalid_info['ram_invalid']:
            if idx in invalid_df.index:
                invalid_df.loc[idx, 'ram_invalid'] = True
        
        for idx, _ in invalid_info['storage_invalid']:
            if idx in invalid_df.index:
                invalid_df.loc[idx, 'storage_invalid'] = True
        
        # Add validation notes
        invalid_df['validation_notes'] = ''
        for idx in invalid_df.index:
            notes = []
            if invalid_df.loc[idx, 'ram_invalid']:
                ram_val = invalid_df.loc[idx, 'ram_gb']
                notes.append(f"RAM {ram_val}GB outside valid range (1-128GB)")
            if invalid_df.loc[idx, 'storage_invalid']:
                storage_val = invalid_df.loc[idx, 'storage_gb']
                notes.append(f"Storage {storage_val}GB outside valid range (16-8192GB)")
            invalid_df.loc[idx, 'validation_notes'] = '; '.join(notes)
        
        return invalid_df
    
    def clean_invalid_specifications(self, action: str = 'drop') -> Dict:
        """
        Clean records with invalid specifications by either dropping them or marking them.
        
        Args:
            action (str): 'drop' to remove invalid records, 'mark' to add validation flags
            
        Returns:
            Dict: Summary of cleaning operation
        """
        if self.df_laptop is None:
            return {'error': 'Laptop data not available'}
        
        invalid_info = self.identify_and_clean_invalid_specifications()
        
        if 'error' in invalid_info:
            return invalid_info
        
        if invalid_info['summary']['total_invalid_records'] == 0:
            return {
                'message': 'No invalid records found',
                'action': action,
                'records_affected': 0
            }
        
        if action == 'drop':
            # Get all invalid indices
            invalid_indices = set()
            if invalid_info['ram_invalid']:
                invalid_indices.update([idx for idx, _ in invalid_info['ram_invalid']])
            if invalid_info['storage_invalid']:
                invalid_indices.update([idx for idx, _ in invalid_info['storage_invalid']])
            
            # Drop invalid records
            original_count = len(self.df_laptop)
            self.df_laptop = self.df_laptop.drop(index=list(invalid_indices))
            dropped_count = original_count - len(self.df_laptop)
            
            return {
                'action': 'drop',
                'original_count': original_count,
                'final_count': len(self.df_laptop),
                'records_dropped': dropped_count,
                'invalid_ram_count': invalid_info['summary']['ram_invalid_count'],
                'invalid_storage_count': invalid_info['summary']['storage_invalid_count'],
                'both_invalid_count': invalid_info['summary']['both_invalid_count']
            }
        
        elif action == 'mark':
            # Add validation flags
            if 'ram_gb' in self.df_laptop.columns:
                self.df_laptop['ram_valid'] = True
                ram_mask = (
                    (self.df_laptop['ram_gb'] < 1) | 
                    (self.df_laptop['ram_gb'] > 128)
                ) & self.df_laptop['ram_gb'].notna()
                self.df_laptop.loc[ram_mask, 'ram_valid'] = False
            
            if 'storage_gb' in self.df_laptop.columns:
                self.df_laptop['storage_valid'] = True
                storage_mask = (
                    (self.df_laptop['storage_gb'] < 16) | 
                    (self.df_laptop['storage_gb'] > 8192)
                ) & self.df_laptop['storage_gb'].notna()
                self.df_laptop.loc[storage_mask, 'storage_valid'] = False
            
            return {
                'action': 'mark',
                'records_affected': invalid_info['summary']['total_invalid_records'],
                'validation_flags_added': ['ram_valid', 'storage_valid'],
                'invalid_ram_count': invalid_info['summary']['ram_invalid_count'],
                'invalid_storage_count': invalid_info['summary']['storage_invalid_count'],
                'both_invalid_count': invalid_info['summary']['both_invalid_count']
            }
        
        else:
            return {'error': f'Invalid action: {action}. Use "drop" or "mark".'}

    def export_invalid_records(self, output_path: str = None) -> Dict:
        """
        Export records with invalid specifications to a CSV file for further analysis.
        
        Args:
            output_path (str): Path to save the CSV file. If None, uses default path.
            
        Returns:
            Dict: Summary of export operation
        """
        if self.df_laptop is None:
            return {'error': 'Laptop data not available'}
        
        invalid_info = self.identify_and_clean_invalid_specifications()
        
        if 'error' in invalid_info:
            return invalid_info
        
        if invalid_info['summary']['total_invalid_records'] == 0:
            return {
                'message': 'No invalid records to export',
                'records_exported': 0
            }
        
        # Get detailed invalid records
        invalid_df = self.get_invalid_records_details()
        
        if invalid_df.empty:
            return {'error': 'No invalid records found to export'}
        
        # Set default output path if none provided
        if output_path is None:
            import os
            output_dir = 'data'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, 'invalid_specifications.csv')
        
        try:
            # Export to CSV
            invalid_df.to_csv(output_path, index=False)
            
            return {
                'success': True,
                'output_path': output_path,
                'records_exported': len(invalid_df),
                'file_size_mb': os.path.getsize(output_path) / (1024 * 1024),
                'columns_exported': list(invalid_df.columns),
                'summary': {
                    'ram_invalid_count': invalid_info['summary']['ram_invalid_count'],
                    'storage_invalid_count': invalid_info['summary']['storage_invalid_count'],
                    'both_invalid_count': invalid_info['summary']['both_invalid_count']
                }
            }
        except Exception as e:
            return {
                'error': f'Failed to export invalid records: {str(e)}',
                'output_path': output_path
            }


def main():
    """
    Main function to run comprehensive data exploration.
    """
    print("Starting Laptop Dataset Exploration...")
    
    # Initialize explorer
    explorer = LaptopDataExplorer()
    
    # Get dataset summary
    print("\n1. Getting dataset summary...")
    summary = explorer.get_dataset_summary()
    
    if 'laptop_data' in summary:
        laptop_info = summary['laptop_data']
        print(f"Laptop Data: {laptop_info['total_products']} products, {laptop_info['total_features']} features")
        print(f"Brands: {laptop_info['brands_count']}")
        
        # Display top brands if available
        if 'top_brands' in laptop_info and laptop_info['top_brands']:
            print("Top Brands:")
            for brand, count in laptop_info['top_brands'].items():
                print(f"  {brand}: {count} products")
        
        if laptop_info['price_range_myr']['min']:
            print(f"Price Range: RM {laptop_info['price_range_myr']['min']:.2f} - RM {laptop_info['price_range_myr']['max']:.2f}")
        
        # Display specification information
        if 'specifications' in laptop_info:
            print("\nSpecifications:")
            specs = laptop_info['specifications']
            if 'ram' in specs:
                ram_info = specs['ram']
                print(f"  RAM: {ram_info['found']}/{ram_info['total']} products found")
                if ram_info['mean_gb']:
                    print(f"    Range: {ram_info['min_gb']:.0f}GB - {ram_info['max_gb']:.0f}GB, Mean: {ram_info['mean_gb']:.1f}GB")
            
            if 'storage' in specs:
                storage_info = specs['storage']
                print(f"  Storage: {storage_info['found']}/{storage_info['total']} products found")
                if storage_info['mean_gb']:
                    print(f"    Range: {storage_info['min_gb']:.0f}GB - {storage_info['max_gb']:.0f}GB, Mean: {storage_info['mean_gb']:.1f}GB")
            
            if 'screen_size' in specs:
                screen_info = specs['screen_size']
                print(f"  Screen Size: {screen_info['found']}/{screen_info['total']} products found")
                if screen_info['mean_inches']:
                    print(f"    Range: {screen_info['min_inches']:.1f}\" - {screen_info['max_inches']:.1f}\", Mean: {screen_info['mean_inches']:.1f}\"")
        
        # Display column categories
        if 'column_categories' in laptop_info:
            print("\nColumn Categories:")
            categories = laptop_info['column_categories']
            for category, cols in categories.items():
                if cols:
                    print(f"  {category.replace('_', ' ').title()}: {len(cols)} columns")
                    if len(cols) <= 5:  # Show all if 5 or fewer
                        print(f"    {', '.join(cols)}")
                    else:  # Show first few if more than 5
                        print(f"    {', '.join(cols[:3])}... and {len(cols)-3} more")

    if 'rating_data' in summary:
        rating_info = summary['rating_data']
        print(f"Rating Data: {rating_info['total_reviews']} reviews, {rating_info['total_features']} features")
        print(f"Unique Users: {rating_info['unique_users']}")
        print(f"Unique Products: {rating_info['unique_products']}")
    
    # Generate comprehensive analysis
    print("\n2. Analyzing price distribution...")
    price_analysis = explorer.analyze_price_distribution()
    if 'error' not in price_analysis:
        print(f"Price range: RM {price_analysis['price_statistics']['min']:.2f} - RM {price_analysis['price_statistics']['max']:.2f}")
    else:
        print(f"Error analyzing price: {price_analysis['error']}")
    
    print("\n3. Analyzing brands...")
    brand_analysis = explorer.analyze_brands()
    if 'error' not in brand_analysis:
        print(f"Most popular brand: {brand_analysis['brand_stats']['most_popular']} with {brand_analysis['brand_stats']['most_popular_count']} products")
    else:
        print(f"Error analyzing brands: {brand_analysis['error']}")
    
    print("\n4. Analyzing ratings...")
    rating_analysis = explorer.analyze_ratings()
    if 'error' not in rating_analysis:
        print(f"Average rating: {rating_analysis['rating_statistics']['mean']:.2f}/5.0")
    else:
        print(f"Error analyzing ratings: {rating_analysis['error']}")
    
    print("\n5. Analyzing reviews...")
    review_analysis = explorer.analyze_reviews()
    if 'error' not in review_analysis:
        print(f"Total reviews: {review_analysis['review_statistics']['total_reviews']:,}")
    else:
        print(f"Error analyzing reviews: {review_analysis['error']}")
    
    print("\n6. Analyzing specifications...")
    spec_analysis = explorer.analyze_specifications()
    if spec_analysis and 'ram_analysis' in spec_analysis and 'common_ram_sizes' in spec_analysis['ram_analysis']:
        print(f"Most common RAM: {max(spec_analysis['ram_analysis']['common_ram_sizes'], key=spec_analysis['ram_analysis']['common_ram_sizes'].get)} GB")
    elif spec_analysis:
        print("Specifications analyzed but RAM data not available.")
    else:
        print("Error analyzing specifications.")
    
    print("\n7. Identifying and cleaning invalid specifications (RAM/Storage outside valid ranges)...")

    invalid_specs = explorer.identify_and_clean_invalid_specifications()
    if 'error' not in invalid_specs:
        summary = invalid_specs['summary']
        print(f"Specifications Validation Summary:")
        print(f"  Original records: {summary['total_records_original']:,}")
        print(f"  Final records: {summary['total_records_final']:,}")
        print(f"  Records dropped: {summary['records_dropped']:,}")
        print(f"  Records with invalid RAM: {summary['ram_invalid_count']:,}")
        print(f"  Records with invalid storage: {summary['storage_invalid_count']:,}")
        print(f"  Records with both invalid: {summary['both_invalid_count']:,}")
        print(f"  Valid RAM range: {summary['valid_ram_range'][0]}-{summary['valid_ram_range'][1]}GB")
        print(f"  Valid storage range: {summary['valid_storage_range'][0]}-{summary['valid_storage_range'][1]}GB")
        
        if summary['action_taken'] == 'dropped_invalid_records':
            print(f"\n  ✅ Successfully cleaned dataset by dropping {summary['records_dropped']:,} invalid records")
        elif summary['action_taken'] == 'no_action_needed':
            print(f"\n  ✅ No invalid records found - dataset is clean")
    else:
        print(f"Error identifying and cleaning invalid specifications: {invalid_specs['error']}")
    
    print("\n8. Generating visualizations...")
    explorer.create_visualizations()
    
    print("\n9. Generating analysis report...")
    explorer.generate_report()
    
    print("\n10. Getting recommendation insights...")
    insights = explorer.get_recommendation_insights()
    print("Recommendation strategies identified:")
    if 'price_insights' in insights:
        print(f"  - Price-based filtering for budget-conscious users (threshold: RM {insights['price_insights']['budget_threshold']:.2f})")
    if 'rating_insights' in insights:
        print(f"  - Performance-based recommendations for power users (highly rated threshold: {insights['rating_insights']['highly_rated_threshold']:.2f})")
    if 'brand_insights' in insights:
        print(f"  - Brand-based suggestions for brand-loyal customers")
    if 'feature_insights' in insights:
        print(f"  - Value-for-money recommendations using helpful reviews (threshold: {insights['feature_insights']['helpful_reviews_threshold']:.2f})")
    
    print("\n11. Getting data quality insights...")
    quality_insights = explorer.get_data_quality_insights()
    if 'laptop_data_quality' in quality_insights:
        laptop_quality = quality_insights['laptop_data_quality']
        print(f"  Laptop data quality: {laptop_quality['missing_percentage']:.1f}% missing values")
        print(f"  Encoded columns: {len(laptop_quality['encoded_columns'])}")
        print(f"  Clean columns: {len(laptop_quality['clean_columns'])}")
    
    if 'rating_data_quality' in quality_insights:
        rating_quality = quality_insights['rating_data_quality']
        print(f"  Rating data quality: {rating_quality['missing_percentage']:.1f}% missing values")
        print(f"  Temporal features: {len(rating_quality['temporal_columns'])}")
    
    print("\nData exploration completed successfully!")
    print("Check the 'data/visualizations/' folder for charts and 'data/analysis_report.txt' for detailed report.")


if __name__ == "__main__":
    main()
