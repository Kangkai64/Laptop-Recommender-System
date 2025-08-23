"""
Data Explorer for Laptop Recommender System
Provides comprehensive data analysis and visualization capabilities
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
    Provides analysis, visualization, and insights from the processed data.
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
            self.df = preprocessor.preprocess_pipeline()
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
        price_col = 'latest_price_myr' if 'latest_price_myr' in self.df.columns else 'latest_price'
        
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
            'price_categories': self.df['price_category'].value_counts().to_dict(),
            'price_range_analysis': {
                'budget_count': len(self.df[self.df['price_category'] == 'Budget']),
                'mid_range_count': len(self.df[self.df['price_category'] == 'Mid-range']),
                'high_end_count': len(self.df[self.df['price_category'] == 'High-end']),
                'premium_count': len(self.df[self.df['price_category'] == 'Premium'])
            }
        }
        return analysis
    
    def analyze_brands(self) -> Dict:
        """
        Analyze brand distribution and performance.
        
        Returns:
            Dict: Brand analysis results
        """
        brand_analysis = {
            'brand_counts': self.df['brand'].value_counts().to_dict(),
            'top_brands': self.df['brand'].value_counts().head(10).to_dict(),
            'brand_price_stats': self.df.groupby('brand')['latest_price_myr'].agg([
                'count', 'mean', 'min', 'max', 'std'
            ]).round(2).to_dict('index'),
            'brand_performance': self.df.groupby('brand')['star_rating'].agg([
                'mean', 'count'
            ]).round(3).to_dict('index')
        }
        return brand_analysis
    
    def analyze_specifications(self) -> Dict:
        """
        Analyze laptop specifications distribution.
        
        Returns:
            Dict: Specifications analysis results
        """
        specs_analysis = {
            'ram_distribution': self.df['ram_gb'].value_counts().sort_index().to_dict(),
            'storage_types': self.df['storage_type'].value_counts().to_dict(),
            'processor_brands': self.df['processor_brand'].value_counts().to_dict(),
            'performance_categories': self.df['performance_category'].value_counts().to_dict(),
            'display_sizes': self.df['display_size'].value_counts().sort_index().to_dict(),
            'graphics_distribution': self.df['graphic_card_gb'].value_counts().sort_index().to_dict()
        }
        return specs_analysis
    
    def analyze_correlations(self) -> pd.DataFrame:
        """
        Analyze correlations between numerical features.
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        return correlation_matrix
    
    def get_value_analysis(self) -> Dict:
        """
        Analyze value for money aspects.
        
        Returns:
            Dict: Value analysis results
        """
        value_analysis = {
            'value_score_stats': {
                'mean': self.df['value_score'].mean(),
                'median': self.df['value_score'].median(),
                'std': self.df['value_score'].std(),
                'max': self.df['value_score'].max(),
                'min': self.df['value_score'].min()
            },
            'best_value_laptops': self.df.nlargest(10, 'value_score')[
                ['brand', 'model', 'latest_price_myr', 'star_rating', 'value_score']
            ].to_dict('records'),
            'price_performance_ratio': self.df.groupby('performance_category')['latest_price_myr'].agg([
                'mean', 'count'
            ]).round(2).to_dict('index')
        }
        return value_analysis
    
    def create_visualizations(self, save_path: str = "data/visualizations/"):
        """
        Create comprehensive visualizations of the dataset.
        
        Args:
            save_path (str): Path to save visualization files
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Set figure size for better quality
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Price Distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        price_col = 'latest_price_myr' if 'latest_price_myr' in self.df.columns else 'latest_price'
        plt.hist(self.df[price_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Price Distribution (MYR)')
        plt.xlabel('Price (MYR)')
        plt.ylabel('Frequency')
        
        # 2. Brand Distribution
        plt.subplot(2, 3, 2)
        top_brands = self.df['brand'].value_counts().head(10)
        plt.barh(range(len(top_brands)), top_brands.values)
        plt.yticks(range(len(top_brands)), top_brands.index)
        plt.title('Top 10 Brands')
        plt.xlabel('Number of Models')
        
        # 3. RAM Distribution
        plt.subplot(2, 3, 3)
        ram_counts = self.df['ram_gb'].value_counts().sort_index()
        plt.bar(ram_counts.index, ram_counts.values, color='lightgreen')
        plt.title('RAM Distribution')
        plt.xlabel('RAM (GB)')
        plt.ylabel('Count')
        
        # 4. Storage Types
        plt.subplot(2, 3, 4)
        storage_counts = self.df['storage_type'].value_counts()
        plt.pie(storage_counts.values, labels=storage_counts.index, autopct='%1.1f%%')
        plt.title('Storage Types Distribution')
        
        # 5. Performance Categories
        plt.subplot(2, 3, 5)
        perf_counts = self.df['performance_category'].value_counts()
        plt.bar(perf_counts.index, perf_counts.values, color='orange')
        plt.title('Performance Categories')
        plt.ylabel('Count')
        
        # 6. Price Categories
        plt.subplot(2, 3, 6)
        price_counts = self.df['price_category'].value_counts()
        plt.bar(price_counts.index, price_counts.values, color='lightcoral')
        plt.title('Price Categories')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}overview_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation Heatmap
        plt.figure(figsize=(12, 10))
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f"{save_path}correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Price vs Performance Analysis
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        for category in self.df['performance_category'].unique():
            subset = self.df[self.df['performance_category'] == category]
            plt.scatter(subset['latest_price_myr'], subset['star_rating'], 
                       alpha=0.6, label=category, s=50)
        plt.xlabel('Price (MYR)')
        plt.ylabel('Star Rating')
        plt.title('Price vs Rating by Performance')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        for storage_type in self.df['storage_type'].unique():
            subset = self.df[self.df['storage_type'] == storage_type]
            plt.scatter(subset['latest_price_myr'], subset['total_storage_gb'], 
                       alpha=0.6, label=storage_type, s=50)
        plt.xlabel('Price (MYR)')
        plt.ylabel('Total Storage (GB)')
        plt.title('Price vs Storage by Type')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        for brand in self.df['brand'].value_counts().head(5).index:
            subset = self.df[self.df['brand'] == brand]
            plt.scatter(subset['latest_price_myr'], subset['value_score'], 
                       alpha=0.6, label=brand, s=50)
        plt.xlabel('Price (MYR)')
        plt.ylabel('Value Score')
        plt.title('Price vs Value by Top Brands')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_path}price_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, output_path: str = "data/analysis_report.txt"):
        """
        Generate a comprehensive analysis report.
        
        Args:
            output_path (str): Path to save the report
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("LAPTOP DATASET ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Basic Information
        basic_info = self.get_basic_info()
        report_lines.append("BASIC INFORMATION:")
        report_lines.append(f"Dataset Shape: {basic_info['shape']}")
        report_lines.append(f"Total Features: {basic_info['shape'][1]}")
        report_lines.append(f"Memory Usage: {basic_info['memory_usage']:.2f} MB")
        report_lines.append("")
        
        # Price Analysis
        price_analysis = self.analyze_price_distribution()
        report_lines.append("PRICE ANALYSIS:")
        report_lines.append(f"Price Range: RM {price_analysis['price_statistics']['min']:.2f} - RM {price_analysis['price_statistics']['max']:.2f}")
        report_lines.append(f"Average Price: RM {price_analysis['price_statistics']['mean']:.2f}")
        report_lines.append(f"Median Price: RM {price_analysis['price_statistics']['median']:.2f}")
        report_lines.append("")
        
        report_lines.append("Price Categories:")
        for category, count in price_analysis['price_categories'].items():
            report_lines.append(f"  {category}: {count} laptops")
        report_lines.append("")
        
        # Brand Analysis
        brand_analysis = self.analyze_brands()
        report_lines.append("BRAND ANALYSIS:")
        report_lines.append("Top 5 Brands:")
        for i, (brand, count) in enumerate(list(brand_analysis['top_brands'].items())[:5], 1):
            report_lines.append(f"  {i}. {brand}: {count} models")
        report_lines.append("")
        
        # Specifications Analysis
        specs_analysis = self.analyze_specifications()
        report_lines.append("SPECIFICATIONS ANALYSIS:")
        report_lines.append(f"RAM Options: {list(specs_analysis['ram_distribution'].keys())}")
        report_lines.append(f"Storage Types: {list(specs_analysis['storage_types'].keys())}")
        report_lines.append(f"Processor Brands: {list(specs_analysis['processor_brands'].keys())}")
        report_lines.append("")
        
        # Value Analysis
        value_analysis = self.get_value_analysis()
        report_lines.append("VALUE ANALYSIS:")
        report_lines.append(f"Average Value Score: {value_analysis['value_score_stats']['mean']:.4f}")
        report_lines.append("Best Value Laptops:")
        for i, laptop in enumerate(value_analysis['best_value_laptops'][:5], 1):
            report_lines.append(f"  {i}. {laptop['brand']} {laptop['model']} - RM {laptop['latest_price_myr']:.2f} (Score: {laptop['value_score']:.4f})")
        report_lines.append("")
        
        # Missing Values
        missing_values = basic_info['missing_values']
        if any(missing_values.values()):
            report_lines.append("MISSING VALUES:")
            for col, count in missing_values.items():
                if count > 0:
                    report_lines.append(f"  {col}: {count} missing values")
        else:
            report_lines.append("No missing values found in the dataset.")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 60)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Analysis report saved to: {output_path}")
    
    def get_recommendation_insights(self) -> Dict:
        """
        Get insights useful for recommendation system development.
        
        Returns:
            Dict: Recommendation insights
        """
        insights = {
            'feature_importance_hints': {
                'price_sensitivity': 'Price is strongly correlated with performance features',
                'brand_preference': 'Top brands show consistent quality ratings',
                'storage_impact': 'SSD laptops generally have higher ratings',
                'ram_importance': 'RAM capacity significantly affects performance category'
            },
            'segmentation_opportunities': {
                'budget_segment': len(self.df[self.df['price_category'] == 'Budget']),
                'performance_segment': len(self.df[self.df['performance_category'] == 'High']),
                'brand_loyalty': self.df['brand'].value_counts().head(5).to_dict()
            },
            'recommendation_strategies': [
                'Price-based filtering for budget-conscious users',
                'Performance-based recommendations for power users',
                'Brand-based suggestions for brand-loyal customers',
                'Value-for-money recommendations using value_score'
            ]
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
    print(f"Price range: RM {price_analysis['price_statistics']['min']:.2f} - RM {price_analysis['price_statistics']['max']:.2f}")
    
    print("\n3. Analyzing brands...")
    brand_analysis = explorer.analyze_brands()
    print(f"Top brand: {list(brand_analysis['top_brands'].keys())[0]} with {list(brand_analysis['top_brands'].values())[0]} models")
    
    print("\n4. Analyzing specifications...")
    specs_analysis = explorer.analyze_specifications()
    print(f"Most common RAM: {max(specs_analysis['ram_distribution'], key=specs_analysis['ram_distribution'].get)} GB")
    
    print("\n5. Generating visualizations...")
    explorer.create_visualizations()
    
    print("\n6. Generating analysis report...")
    explorer.generate_report()
    
    print("\n7. Getting recommendation insights...")
    insights = explorer.get_recommendation_insights()
    print("Recommendation strategies identified:")
    for strategy in insights['recommendation_strategies']:
        print(f"  - {strategy}")
    
    print("\nData exploration completed successfully!")
    print("Check the 'data/visualizations/' folder for charts and 'data/analysis_report.txt' for detailed report.")


if __name__ == "__main__":
    main()
