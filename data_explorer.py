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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LaptopDataExplorer:
    """
    Comprehensive data explorer for laptop recommendation system.
    Provides analysis, visualization, and insights from the processed Amazon laptop reviews data.
    """
    
    def __init__(self, data_path: str = "data/processed_laptop_data.csv"):
        """
        Initialize the data explorer.
        
        Args:
            data_path (str): Path to the processed CSV data file
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the processed dataset.
        
        Returns:
            pd.DataFrame: Processed dataset
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print(f"Processed data file not found: {self.data_path}")
            print("Running preprocessing pipeline first...")
            preprocessor = LaptopDataPreprocessor()
            # Use the separated preprocessing pipeline for better data structure
            df_laptop, df_rating = preprocessor.preprocess_separated_pipeline()
            # For data exploration, we'll use the laptop data as primary
            self.df = df_laptop
            return self.df
    
    def get_basic_info(self) -> Dict:
        """
        Get basic information about the dataset.
        
        Returns:
            Dict: Basic dataset information
        """
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }
        return info
    
    def analyze_price_distribution(self) -> Dict:
        """
        Analyze price distribution and statistics.
        
        Returns:
            Dict: Price analysis results
        """
        # Check for different possible price column names
        price_cols = ['price_myr', 'price_usd', 'price_numeric', 'price']
        price_col = None
        for col in price_cols:
            if col in self.df.columns:
                price_col = col
                break
        
        if price_col is None:
            return {'error': 'Price column not found'}
        
        analysis = {
            'price_statistics': {
                'min': self.df[price_col].min(),
                'max': self.df[price_col].max(),
                'mean': self.df[price_col].mean(),
                'median': self.df[price_col].median(),
                'std': self.df[price_col].std(),
                'q25': self.df[price_col].quantile(0.25),
                'q75': self.df[price_col].quantile(0.75)
            },
            'price_categories': self.df['price_category_myr'].value_counts().to_dict() if 'price_category_myr' in self.df.columns else {},
            'price_range_analysis': {
                'budget_count': len(self.df[self.df['price_category_myr'] == 'Budget']) if 'price_category_myr' in self.df.columns else 0,
                'mid_range_count': len(self.df[self.df['price_category_myr'] == 'Mid-range']) if 'price_category_myr' in self.df.columns else 0,
                'high_end_count': len(self.df[self.df['price_category_myr'] == 'High-end']) if 'price_category_myr' in self.df.columns else 0,
                'premium_count': len(self.df[self.df['price_category_myr'] == 'Premium']) if 'price_category_myr' in self.df.columns else 0
            }
        }
        return analysis
    
    def analyze_brands(self) -> Dict:
        """
        Analyze brand distribution and statistics.
        
        Returns:
            Dict: Brand analysis results
        """
        # Check for brand column (could be encoded or original)
        brand_col = 'brand' if 'brand' in self.df.columns else 'brand_encoded' if 'brand_encoded' in self.df.columns else None
        
        if brand_col is None:
            return {'error': 'Brand column not found'}
        
        brand_analysis = self.df[brand_col].value_counts()
        
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
        rating_col = 'rating' if 'rating' in self.df.columns else 'average_rating' if 'average_rating' in self.df.columns else None
        if rating_col:
            brand_ratings = self.df.groupby(brand_col)[rating_col].agg(['mean', 'count']).sort_values('count', ascending=False)
            analysis['brand_ratings'] = brand_ratings.head(10).to_dict('index')
        
        return analysis
    
    def analyze_ratings(self) -> Dict:
        """
        Analyze rating distribution and patterns.
        
        Returns:
            Dict: Rating analysis results
        """
        if 'rating' not in self.df.columns:
            return {'error': 'Rating column not found'}
        
        analysis = {
            'rating_distribution': self.df['rating'].value_counts().sort_index().to_dict(),
            'rating_statistics': {
                'mean': self.df['rating'].mean(),
                'median': self.df['rating'].median(),
                'std': self.df['rating'].std(),
                'min': self.df['rating'].min(),
                'max': self.df['rating'].max()
            },
            'rating_categories': self.df['rating_category'].value_counts().to_dict() if 'rating_category' in self.df.columns else {}
        }
        
        # Analyze helpful votes
        if 'helpful_vote' in self.df.columns:
            analysis['helpful_votes'] = {
                'total_helpful_votes': self.df['helpful_vote'].sum(),
                'avg_helpful_votes': self.df['helpful_vote'].mean(),
                'max_helpful_votes': self.df['helpful_vote'].max()
            }
        
        return analysis
    
    def analyze_reviews(self) -> Dict:
        """
        Analyze review content and patterns.
        
        Returns:
            Dict: Review analysis results
        """
        if 'text' not in self.df.columns:
            return {'error': 'Review text column not found'}
        
        analysis = {
            'review_statistics': {
                'total_reviews': len(self.df),
                'avg_review_length': self.df['text'].astype(str).str.len().mean() if 'text' in self.df.columns else 0,
                'max_review_length': self.df['text'].astype(str).str.len().max() if 'text' in self.df.columns else 0,
                'min_review_length': self.df['text'].astype(str).str.len().min() if 'text' in self.df.columns else 0
            }
        }
        
        # Analyze review length distribution
        if 'review_length' in self.df.columns:
            analysis['review_length_categories'] = pd.cut(
                self.df['review_length'], 
                bins=[0, 100, 500, 1000, float('inf')], 
                labels=['Short', 'Medium', 'Long', 'Very Long']
            ).value_counts().to_dict()
        
        # Analyze verified purchases
        if 'verified_purchase' in self.df.columns:
            analysis['verified_purchases'] = {
                'verified_count': self.df['verified_purchase'].sum(),
                'verified_percentage': (self.df['verified_purchase'].sum() / len(self.df)) * 100
            }
        
        return analysis
    
    def analyze_specifications(self) -> Dict:
        """
        Analyze laptop specifications extracted from details.
        
        Returns:
            Dict: Specifications analysis results
        """
        analysis = {}
        
        # Analyze RAM distribution
        if 'ram_gb' in self.df.columns:
            ram_data = self.df['ram_gb'].dropna()
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
        if 'storage_gb' in self.df.columns:
            storage_data = self.df['storage_gb'].dropna()
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
        if 'screen_size' in self.df.columns:
            screen_data = self.df['screen_size'].dropna()
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
        
        # Analyze operating systems
        if 'os' in self.df.columns:
            analysis['os_analysis'] = {
                'os_distribution': self.df['os'].value_counts().to_dict(),
                'most_common_os': self.df['os'].mode().iloc[0] if len(self.df['os'].mode()) > 0 else None
            }
        
        return analysis
    
    def create_visualizations(self, save_path: str = "data/visualizations/"):
        """
        Create comprehensive visualizations of the dataset.
        
        Args:
            save_path (str): Path to save visualizations
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Price Distribution
        if 'price_numeric' in self.df.columns:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.hist(self.df['price_numeric'].dropna(), bins=50, alpha=0.7, edgecolor='black')
            plt.title('Price Distribution')
            plt.xlabel('Price ($)')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            if 'price_category' in self.df.columns:
                self.df['price_category'].value_counts().plot(kind='bar')
                plt.title('Price Categories Distribution')
                plt.xlabel('Price Category')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}price_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Rating Analysis
        if 'rating' in self.df.columns:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            self.df['rating'].value_counts().sort_index().plot(kind='bar')
            plt.title('Rating Distribution')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            
            plt.subplot(1, 3, 2)
            if 'rating_category' in self.df.columns:
                self.df['rating_category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title('Rating Categories')
            
            plt.subplot(1, 3, 3)
            if 'helpful_vote' in self.df.columns:
                plt.hist(self.df['helpful_vote'].dropna(), bins=30, alpha=0.7, edgecolor='black')
                plt.title('Helpful Votes Distribution')
                plt.xlabel('Helpful Votes')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}rating_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Brand Analysis
        if 'brand' in self.df.columns:
            plt.figure(figsize=(12, 6))
            top_brands = self.df['brand'].value_counts().head(10)
            top_brands.plot(kind='bar')
            plt.title('Top 10 Brands by Number of Reviews')
            plt.xlabel('Brand')
            plt.ylabel('Number of Reviews')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{save_path}brand_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Review Length Analysis
        if 'review_length' in self.df.columns:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.hist(self.df['review_length'], bins=50, alpha=0.7, edgecolor='black')
            plt.title('Review Length Distribution')
            plt.xlabel('Review Length (characters)')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            if 'rating' in self.df.columns:
                plt.scatter(self.df['review_length'], self.df['rating'], alpha=0.5)
                plt.title('Review Length vs Rating')
                plt.xlabel('Review Length (characters)')
                plt.ylabel('Rating')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}review_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Specifications Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RAM analysis
        if 'ram_gb' in self.df.columns:
            ram_data = self.df['ram_gb'].dropna()
            if len(ram_data) > 0:
                axes[0, 0].hist(ram_data, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 0].set_title('RAM Distribution')
                axes[0, 0].set_xlabel('RAM (GB)')
                axes[0, 0].set_ylabel('Frequency')
        
        # Storage analysis
        if 'storage_gb' in self.df.columns:
            storage_data = self.df['storage_gb'].dropna()
            if len(storage_data) > 0:
                axes[0, 1].hist(storage_data, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('Storage Distribution')
                axes[0, 1].set_xlabel('Storage (GB)')
                axes[0, 1].set_ylabel('Frequency')
        
        # Screen size analysis
        if 'screen_size' in self.df.columns:
            screen_data = self.df['screen_size'].dropna()
            if len(screen_data) > 0:
                axes[1, 0].hist(screen_data, bins=15, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Screen Size Distribution')
                axes[1, 0].set_xlabel('Screen Size (inches)')
                axes[1, 0].set_ylabel('Frequency')
        
        # OS distribution
        if 'os' in self.df.columns:
            self.df['os'].value_counts().head(8).plot(kind='bar', ax=axes[1, 1])
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
            report_lines.append(f"Price Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
            report_lines.append(f"Average Price: ${stats['mean']:.2f}")
            report_lines.append(f"Median Price: ${stats['median']:.2f}")
            report_lines.append("")
            
            if price_analysis['price_categories']:
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
            report_lines.append("")
            report_lines.append("Top 10 Brands:")
            for i, (brand, count) in enumerate(list(brand_analysis['top_brands'].items())[:10], 1):
                report_lines.append(f"  {i}. {brand}: {count:,} reviews")
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
            report_lines.append("")
            
            if rating_analysis['rating_categories']:
                report_lines.append("Rating Categories:")
                for category, count in rating_analysis['rating_categories'].items():
                    report_lines.append(f"  {category}: {count:,} reviews")
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
            
            if 'verified_purchases' in review_analysis:
                verified = review_analysis['verified_purchases']
                report_lines.append(f"Verified Purchases: {verified['verified_count']:,} ({verified['verified_percentage']:.1f}%)")
                report_lines.append("")
        
        # Specifications Analysis
        spec_analysis = self.analyze_specifications()
        if spec_analysis:
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
        if 'price_numeric' in self.df.columns:
            price_quartiles = self.df['price_numeric'].quantile([0.25, 0.5, 0.75])
            insights['price_insights'] = {
                'budget_threshold': price_quartiles[0.25],
                'mid_range_threshold': price_quartiles[0.5],
                'high_end_threshold': price_quartiles[0.75]
            }
        
        # Rating insights
        if 'rating' in self.df.columns:
            insights['rating_insights'] = {
                'highly_rated_threshold': self.df['rating'].quantile(0.8),
                'average_rating': self.df['rating'].mean(),
                'rating_distribution': self.df['rating'].value_counts().sort_index().to_dict()
            }
        
        # Brand insights
        if 'brand' in self.df.columns:
            top_brands = self.df['brand'].value_counts().head(10)
            insights['brand_insights'] = {
                'top_brands': top_brands.to_dict(),
                'brand_count': len(self.df['brand'].unique())
            }
        
        # Feature importance insights
        if 'helpful_vote' in self.df.columns and 'rating' in self.df.columns:
            insights['feature_insights'] = {
                'helpful_reviews_threshold': self.df['helpful_vote'].quantile(0.8),
                'correlation_rating_helpful': self.df['rating'].corr(self.df['helpful_vote'])
            }
        
        return insights


def main():
    """
    Main function to run comprehensive data exploration.
    """
    print("Starting Laptop Dataset Exploration...")
    
    # Initialize explorer
    explorer = LaptopDataExplorer()
    
    # Generate comprehensive analysis
    print("\n1. Generating basic information...")
    basic_info = explorer.get_basic_info()
    print(f"Dataset loaded: {basic_info['shape'][0]} records, {basic_info['shape'][1]} features")
    
    print("\n2. Analyzing price distribution...")
    price_analysis = explorer.analyze_price_distribution()
    if 'error' not in price_analysis:
        print(f"Price range: ${price_analysis['price_statistics']['min']:.2f} - ${price_analysis['price_statistics']['max']:.2f}")
    else:
        print(f"Error analyzing price: {price_analysis['error']}")
    
    print("\n3. Analyzing brands...")
    brand_analysis = explorer.analyze_brands()
    if 'error' not in brand_analysis:
        print(f"Most popular brand: {brand_analysis['brand_stats']['most_popular']} with {brand_analysis['brand_stats']['most_popular_count']} reviews")
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
    
    print("\n7. Generating visualizations...")
    explorer.create_visualizations()
    
    print("\n8. Generating analysis report...")
    explorer.generate_report()
    
    print("\n9. Getting recommendation insights...")
    insights = explorer.get_recommendation_insights()
    print("Recommendation strategies identified:")
    if 'price_insights' in insights:
        print(f"  - Price-based filtering for budget-conscious users (threshold: ${insights['price_insights']['budget_threshold']:.2f})")
    if 'rating_insights' in insights:
        print(f"  - Performance-based recommendations for power users (highly rated threshold: {insights['rating_insights']['highly_rated_threshold']:.2f})")
    if 'brand_insights' in insights:
        print(f"  - Brand-based suggestions for brand-loyal customers (top brands: {insights['brand_insights']['top_brands']})")
    if 'feature_insights' in insights:
        print(f"  - Value-for-money recommendations using helpful reviews (threshold: {insights['feature_insights']['helpful_reviews_threshold']:.2f})")
    
    print("\nData exploration completed successfully!")
    print("Check the 'data/visualizations/' folder for charts and 'data/analysis_report.txt' for detailed report.")


if __name__ == "__main__":
    main()
